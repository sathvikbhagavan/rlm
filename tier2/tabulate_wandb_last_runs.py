import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional

import wandb

PROJECT_PATTERN = re.compile(r'project\s*=\s*"([^"]+)"')
SAMPLE_CORRECT_PATTERN = re.compile(r"^sample/(\d+)/is_correct$")
SAMPLE_COST_PATTERN = re.compile(r"^sample/(\d+)/(?:final_total_cost_usd|total_cost_usd)$")
METRICS = ("accuracy",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tabulate tier2 W&B runs.")
    parser.add_argument("--entity", type=str, required=True, help="W&B entity/username.")
    parser.add_argument("--tier2-dir", type=str, default="/home/bhagavan/rlms/rlm/tier2")
    parser.add_argument("--last-n-per-model", type=int, default=3)
    parser.add_argument("--max-models", type=int, default=2)
    parser.add_argument("--models", type=str, default="")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/tier2/wandb_last_runs_report_tables",
    )
    parser.add_argument(
        "--context-sizes",
        type=str,
        default="100",
        help="CodeAct context sizes (default 100). Use 'all' to disable filter.",
    )
    parser.add_argument("--min-task-number", type=int, default=None)
    parser.add_argument("--max-task-number", type=int, default=None)
    return parser.parse_args()


def discover_task_projects(tier2_dir: Path) -> list[str]:
    projects: set[str] = set()
    for pattern in ("rlm_task*.py", "codeact_task*.py"):
        for path in sorted(tier2_dir.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            for match in PROJECT_PATTERN.finditer(text):
                projects.add(match.group(1))
    return sorted(projects, key=parse_task_number)


def parse_task_family(project: str) -> str:
    m = re.match(r"^(.*?)-Task\d+$", project)
    return m.group(1) if m else "unknown"


def parse_task_number(project: str) -> int:
    m = re.search(r"Task(\d+)$", project)
    return int(m.group(1)) if m else -1


def normalize_model_name(run) -> str:
    config = dict(run.config or {})
    for key in ("MODEL_NAME", "model_name", "model", "llm_model"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown_model"


def normalize_context_size(run) -> Optional[int]:
    config = dict(run.config or {})
    for key in ("CONTEXT_SIZE", "context_size"):
        value = config.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                return int(value.strip())
            except ValueError:
                continue
    return None


def extract_sample_correct(summary: dict) -> dict[int, float]:
    result: dict[int, float] = {}
    for key, value in summary.items():
        m = SAMPLE_CORRECT_PATTERN.match(key)
        if not m or not isinstance(value, (int, float)):
            continue
        result[int(m.group(1))] = float(value)
    return result


def extract_sample_costs(summary: dict) -> dict[int, float]:
    result: dict[int, float] = {}
    for key, value in summary.items():
        m = SAMPLE_COST_PATTERN.match(key)
        if not m or not isinstance(value, (int, float)):
            continue
        result[int(m.group(1))] = float(value)
    return result


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tier2_dir = Path(args.tier2_dir)

    context_sizes = [] if args.context_sizes.strip().lower() == "all" else [
        int(v.strip()) for v in args.context_sizes.split(",") if v.strip()
    ]
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]

    projects = discover_task_projects(tier2_dir)
    projects = [
        p for p in projects
        if (args.min_task_number is None or parse_task_number(p) >= args.min_task_number)
        and (args.max_task_number is None or parse_task_number(p) <= args.max_task_number)
    ]
    if not projects:
        raise RuntimeError("No tier2 projects found after filtering.")

    api = wandb.Api()
    sample_rows: list[dict[str, object]] = []
    run_cost_rows: list[dict[str, object]] = []

    for project in projects:
        use_context = "codeact" in parse_task_family(project).lower()
        runs = api.runs(f"{args.entity}/{project}", order="-created_at")
        grouped_counts: dict[tuple[str, int], int] = defaultdict(int)
        seen_models: list[str] = requested_models[:]

        for run in runs:
            summary = dict(run.summary)
            sample_correct = extract_sample_correct(summary)
            if not sample_correct:
                continue

            model_name = normalize_model_name(run)
            if requested_models:
                if model_name not in requested_models:
                    continue
            else:
                if model_name not in seen_models:
                    if len(seen_models) >= args.max_models:
                        continue
                    seen_models.append(model_name)

            context_size = normalize_context_size(run) if use_context else 0
            if context_size is None:
                continue
            if use_context and context_sizes and context_size not in context_sizes:
                continue

            key = (model_name, context_size)
            if grouped_counts[key] >= args.last_n_per_model:
                continue
            grouped_counts[key] += 1

            sample_costs = extract_sample_costs(summary)
            total_cost = summary.get("total_cost_usd")
            total_cost_usd = float(total_cost) if isinstance(total_cost, (int, float)) else (
                sum(sample_costs.values()) if sample_costs else None
            )
            avg_cost = summary.get("avg_cost_per_sample_usd")
            avg_cost_per_sample = float(avg_cost) if isinstance(avg_cost, (int, float)) else (
                (total_cost_usd / len(sample_costs)) if (total_cost_usd is not None and sample_costs) else None
            )

            run_cost_rows.append(
                {
                    "project": project,
                    "task_family": parse_task_family(project),
                    "task_number": parse_task_number(project),
                    "model_name": model_name,
                    "context_size": context_size,
                    "run_id": run.id,
                    "run_name": run.name or run.id,
                    "created_at": run.created_at or "",
                    "sample_count": len(sample_correct),
                    "samples_with_cost": len(sample_costs),
                    "total_cost_usd": total_cost_usd,
                    "avg_cost_per_sample_usd": avg_cost_per_sample,
                }
            )

            for sample_id, value in sample_correct.items():
                for metric in METRICS:
                    sample_rows.append(
                        {
                            "project": project,
                            "task_family": parse_task_family(project),
                            "task_number": parse_task_number(project),
                            "model_name": model_name,
                            "context_size": context_size,
                            "run_id": run.id,
                            "run_name": run.name or run.id,
                            "created_at": run.created_at or "",
                            "sample_id": sample_id,
                            "metric": metric,
                            "value": value,
                            "sample_cost_usd": sample_costs.get(sample_id),
                        }
                    )

    if not sample_rows:
        print("No qualifying runs with sample/*/is_correct found.")
        return

    task_grouped: dict[tuple[str, str, int, str, int, str], list[float]] = defaultdict(list)
    for row in sample_rows:
        key = (
            str(row["project"]),
            str(row["task_family"]),
            int(row["task_number"]),
            str(row["model_name"]),
            int(row["context_size"]),
            str(row["metric"]),
        )
        task_grouped[key].append(float(row["value"]))

    task_rows: list[dict[str, object]] = []
    for key, values in sorted(task_grouped.items()):
        project, task_family, task_number, model_name, context_size, metric = key
        task_rows.append(
            {
                "project": project,
                "task_family": task_family,
                "task_number": task_number,
                "model_name": model_name,
                "context_size": context_size,
                "metric": metric,
                "sample_count": len(values),
                "task_avg_over_samples": safe_mean(values),
            }
        )

    task_cost_grouped: dict[tuple[str, str, int, str, int], list[float]] = defaultdict(list)
    task_avg_cost_grouped: dict[tuple[str, str, int, str, int], list[float]] = defaultdict(list)
    task_run_counts: dict[tuple[str, str, int, str, int], int] = defaultdict(int)
    task_runs_with_cost: dict[tuple[str, str, int, str, int], int] = defaultdict(int)
    for row in run_cost_rows:
        key = (
            str(row["project"]),
            str(row["task_family"]),
            int(row["task_number"]),
            str(row["model_name"]),
            int(row["context_size"]),
        )
        task_run_counts[key] += 1
        if isinstance(row.get("total_cost_usd"), (int, float)):
            task_cost_grouped[key].append(float(row["total_cost_usd"]))
            task_runs_with_cost[key] += 1
        if isinstance(row.get("avg_cost_per_sample_usd"), (int, float)):
            task_avg_cost_grouped[key].append(float(row["avg_cost_per_sample_usd"]))

    task_cost_rows: list[dict[str, object]] = []
    for key in sorted(task_run_counts):
        project, task_family, task_number, model_name, context_size = key
        task_cost_rows.append(
            {
                "project": project,
                "task_family": task_family,
                "task_number": task_number,
                "model_name": model_name,
                "context_size": context_size,
                "run_count": task_run_counts[key],
                "runs_with_cost": task_runs_with_cost.get(key, 0),
                "avg_total_cost_per_run_usd": safe_mean(task_cost_grouped.get(key, [])),
                "avg_cost_per_sample_usd": safe_mean(task_avg_cost_grouped.get(key, [])),
            }
        )

    write_csv(
        output_dir / "sample_points_selected_runs.csv",
        [
            "project", "task_family", "task_number", "model_name", "context_size",
            "run_id", "run_name", "created_at", "sample_id", "metric", "value", "sample_cost_usd",
        ],
        sample_rows,
    )
    write_csv(
        output_dir / "task_averages.csv",
        [
            "project", "task_family", "task_number", "model_name", "context_size",
            "metric", "sample_count", "task_avg_over_samples",
        ],
        task_rows,
    )
    write_csv(
        output_dir / "task_cost_averages.csv",
        [
            "project", "task_family", "task_number", "model_name", "context_size",
            "run_count", "runs_with_cost", "avg_total_cost_per_run_usd", "avg_cost_per_sample_usd",
        ],
        task_cost_rows,
    )
    print(f"Wrote tier2 tables to: {output_dir}")


if __name__ == "__main__":
    main()
