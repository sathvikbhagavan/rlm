import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

try:
    import wandb
except ImportError:  # pragma: no cover - runtime environment dependent
    wandb = None


METRICS = ("precision", "recall", "f1")
SAMPLE_METRIC_PATTERN = re.compile(r"^sample/(\d+)/(precision|recall|f1)$")
PROJECT_PATTERN = re.compile(r'project\s*=\s*"([^"]+)"')


@dataclass
class RunSamples:
    project: str
    task_family: str
    task_number: int
    model_name: str
    run_id: str
    run_name: str
    created_at: str
    sample_metrics: dict[int, dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create tabular summaries for last N W&B runs per model "
            "for each tier3 task project."
        )
    )
    parser.add_argument(
        "--entity",
        type=str,
        required=True,
        help="W&B entity/username that owns the projects.",
    )
    parser.add_argument(
        "--tier3-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/tier3",
        help="Directory containing rlm_task*.py and codeact_task*.py files.",
    )
    parser.add_argument(
        "--last-n-per-model",
        type=int,
        default=5,
        help="Number of latest runs to include per model per project.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=2,
        help="Number of models to include per project when --models is omitted.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Optional comma-separated model names to include "
            "(e.g. 'openai/gpt-5-mini,anthropic/claude-sonnet-4')."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/tier3/wandb_last_runs_report_tables",
        help="Directory where CSV table outputs will be written.",
    )
    return parser.parse_args()


def discover_task_projects(tier3_dir: Path) -> list[str]:
    projects: set[str] = set()
    for pattern in ("rlm_task*.py", "codeact_task*.py"):
        for path in sorted(tier3_dir.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            for match in PROJECT_PATTERN.finditer(text):
                projects.add(match.group(1))

    def task_key(name: str) -> tuple[int, str]:
        m = re.search(r"Task(\d+)$", name)
        if m:
            return (int(m.group(1)), name)
        return (10**9, name)

    return sorted(projects, key=task_key)


def parse_task_family(project: str) -> str:
    m = re.match(r"^(.*?)-Task\d+$", project)
    if m:
        return m.group(1)
    return "unknown"


def parse_task_number(project: str) -> int:
    m = re.search(r"Task(\d+)$", project)
    if m:
        return int(m.group(1))
    return -1


def extract_sample_metrics(summary: dict) -> dict[int, dict[str, float]]:
    per_sample: dict[int, dict[str, float]] = {}
    for key, value in summary.items():
        m = SAMPLE_METRIC_PATTERN.match(key)
        if not m or not isinstance(value, (int, float)):
            continue
        sample_i = int(m.group(1))
        metric = m.group(2)
        per_sample.setdefault(sample_i, {})[metric] = float(value)
    return per_sample


def normalize_model_name(run) -> str:
    config = dict(run.config or {})
    for key in ("MODEL_NAME", "model_name", "model", "llm_model"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown_model"


def fetch_last_runs_by_model(
    entity: str,
    project: str,
    last_n_per_model: int,
    max_models: int,
    requested_models: list[str],
) -> dict[str, list[RunSamples]]:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    grouped: dict[str, list[RunSamples]] = defaultdict(list)
    selected_models: list[str] = requested_models[:]
    target_model_count = len(requested_models) if requested_models else max_models

    for run in runs:
        summary = dict(run.summary)
        sample_metrics = extract_sample_metrics(summary)
        if not sample_metrics:
            continue

        model_name = normalize_model_name(run)
        if requested_models:
            if model_name not in requested_models:
                continue
        else:
            if model_name not in selected_models:
                if len(selected_models) >= max_models:
                    continue
                selected_models.append(model_name)

        if len(grouped[model_name]) >= last_n_per_model:
            if _all_models_saturated(
                grouped=grouped,
                selected_models=selected_models,
                last_n_per_model=last_n_per_model,
                target_model_count=target_model_count,
            ):
                break
            continue

        grouped[model_name].append(
            RunSamples(
                project=project,
                task_family=parse_task_family(project),
                task_number=parse_task_number(project),
                model_name=model_name,
                run_id=run.id,
                run_name=run.name or run.id,
                created_at=run.created_at or "",
                sample_metrics=sample_metrics,
            )
        )

        if _all_models_saturated(
            grouped=grouped,
            selected_models=selected_models,
            last_n_per_model=last_n_per_model,
            target_model_count=target_model_count,
        ):
            break

    return {model: grouped.get(model, []) for model in selected_models}


def _all_models_saturated(
    grouped: dict[str, list[RunSamples]],
    selected_models: list[str],
    last_n_per_model: int,
    target_model_count: int,
) -> bool:
    # Do not stop early before we have discovered the target number of models.
    if not selected_models or len(selected_models) < target_model_count:
        return False
    return all(len(grouped.get(model, [])) >= last_n_per_model for model in selected_models)


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def is_num(value: object) -> bool:
    return isinstance(value, (int, float))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_sample_point_rows(runs: list[RunSamples]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for sample_id, sample_metrics in run.sample_metrics.items():
            for metric in METRICS:
                value = sample_metrics.get(metric)
                if not is_num(value):
                    continue
                rows.append(
                    {
                        "project": run.project,
                        "task_family": run.task_family,
                        "task_number": run.task_number,
                        "model_name": run.model_name,
                        "run_id": run.run_id,
                        "run_name": run.run_name,
                        "created_at": run.created_at,
                        "sample_id": sample_id,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return rows


def build_run_level_rows(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str, str, str, str, str], list[float]] = defaultdict(list)
    for row in sample_rows:
        value = row.get("value")
        if not is_num(value):
            continue
        key = (
            str(row["project"]),
            str(row["task_family"]),
            int(row["task_number"]),
            str(row["model_name"]),
            str(row["run_id"]),
            str(row["run_name"]),
            str(row["created_at"]),
            str(row["metric"]),
        )
        grouped[key].append(float(value))

    rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        project, task_family, task_number, model_name, run_id, run_name, created_at, metric = key
        rows.append(
            {
                "project": project,
                "task_family": task_family,
                "task_number": task_number,
                "model_name": model_name,
                "run_id": run_id,
                "run_name": run_name,
                "created_at": created_at,
                "metric": metric,
                "sample_count": len(values),
                "run_metric_mean": safe_mean(values),
            }
        )
    return rows


def build_sample_avg_rows(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str, int, str], list[float]] = defaultdict(list)
    run_sets: dict[tuple[str, str, int, str, int, str], set[str]] = defaultdict(set)
    for row in sample_rows:
        value = row.get("value")
        sample_id = row.get("sample_id")
        if not is_num(value) or not isinstance(sample_id, int):
            continue
        key = (
            str(row["project"]),
            str(row["task_family"]),
            int(row["task_number"]),
            str(row["model_name"]),
            sample_id,
            str(row["metric"]),
        )
        grouped[key].append(float(value))
        run_sets[key].add(str(row["run_id"]))

    rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        project, task_family, task_number, model_name, sample_id, metric = key
        rows.append(
            {
                "project": project,
                "task_family": task_family,
                "task_number": task_number,
                "model_name": model_name,
                "sample_id": sample_id,
                "metric": metric,
                "run_count": len(run_sets[key]),
                "sample_avg_over_runs": safe_mean(values),
            }
        )
    return rows


def build_task_avg_rows(sample_avg_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(list)
    for row in sample_avg_rows:
        value = row.get("sample_avg_over_runs")
        if not is_num(value):
            continue
        key = (
            str(row["project"]),
            str(row["task_family"]),
            int(row["task_number"]),
            str(row["model_name"]),
            str(row["metric"]),
        )
        grouped[key].append(float(value))

    rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        project, task_family, task_number, model_name, metric = key
        rows.append(
            {
                "project": project,
                "task_family": task_family,
                "task_number": task_number,
                "model_name": model_name,
                "metric": metric,
                "sample_count": len(values),
                "task_avg_over_samples": safe_mean(values),
            }
        )
    return rows


def build_overall_rows(task_avg_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_model_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    all_models_grouped: dict[str, list[float]] = defaultdict(list)

    for row in task_avg_rows:
        value = row.get("task_avg_over_samples")
        if not is_num(value):
            continue
        model_name = str(row["model_name"])
        metric = str(row["metric"])
        project = str(row["project"])
        by_model_grouped[(model_name, metric)].append(float(value))
        all_models_grouped[metric].append(float(value))
        all_models_grouped[f"{project}::{metric}"].append(float(value))

    by_model_rows: list[dict[str, object]] = []
    for key, values in sorted(by_model_grouped.items()):
        model_name, metric = key
        by_model_rows.append(
            {
                "model_name": model_name,
                "metric": metric,
                "task_count": len(values),
                "overall_avg_over_tasks": safe_mean(values),
            }
        )

    all_rows: list[dict[str, object]] = []
    for metric in METRICS:
        values = all_models_grouped.get(metric, [])
        all_rows.append(
            {
                "metric": metric,
                "task_count_all_models": len(values),
                "overall_avg_all_models": safe_mean(values),
            }
        )
    return by_model_rows, all_rows


def build_family_comparison_rows(
    task_avg_rows: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    family_metric_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    family_model_metric_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    task_pair_values: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in task_avg_rows:
        value = row.get("task_avg_over_samples")
        if not is_num(value):
            continue
        family = str(row.get("task_family", "unknown"))
        metric = str(row["metric"])
        model_name = str(row["model_name"])
        task_number = int(row.get("task_number", -1))
        numeric_value = float(value)

        family_metric_values[(family, metric)].append(numeric_value)
        family_model_metric_values[(family, model_name, metric)].append(numeric_value)
        task_pair_values[(task_number, metric)][family].append(numeric_value)

    family_metric_rows: list[dict[str, object]] = []
    for key, values in sorted(family_metric_values.items()):
        family, metric = key
        family_metric_rows.append(
            {
                "task_family": family,
                "metric": metric,
                "task_count": len(values),
                "family_avg_over_tasks": safe_mean(values),
            }
        )

    family_model_rows: list[dict[str, object]] = []
    for key, values in sorted(family_model_metric_values.items()):
        family, model_name, metric = key
        family_model_rows.append(
            {
                "task_family": family,
                "model_name": model_name,
                "metric": metric,
                "task_count": len(values),
                "family_model_avg_over_tasks": safe_mean(values),
            }
        )

    # Pairwise comparison only when both families exist for a task number/metric.
    family_task_pair_rows: list[dict[str, object]] = []
    shared_model_rows: list[dict[str, object]] = []
    families = sorted({family for family, _metric in family_metric_values})
    if len(families) >= 2:
        left_family = families[0]
        right_family = families[1]
        for (task_number, metric), family_map in sorted(task_pair_values.items()):
            left_values = family_map.get(left_family, [])
            right_values = family_map.get(right_family, [])
            if not left_values or not right_values:
                continue
            left_avg = safe_mean(left_values)
            right_avg = safe_mean(right_values)
            family_task_pair_rows.append(
                {
                    "task_number": task_number,
                    "metric": metric,
                    "left_family": left_family,
                    "left_avg": left_avg,
                    "right_family": right_family,
                    "right_avg": right_avg,
                    "delta_left_minus_right": left_avg - right_avg,
                }
            )

        # Shared-model comparison: compare only models present in both families.
        shared_grouped: dict[tuple[int, str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in task_avg_rows:
            value = row.get("task_avg_over_samples")
            if not is_num(value):
                continue
            family = str(row.get("task_family", "unknown"))
            if family not in (left_family, right_family):
                continue
            task_number = int(row.get("task_number", -1))
            metric = str(row.get("metric"))
            model_name = str(row.get("model_name"))
            shared_grouped[(task_number, metric, model_name)][family].append(float(value))

        for (task_number, metric, model_name), fam_map in sorted(shared_grouped.items()):
            left_values = fam_map.get(left_family, [])
            right_values = fam_map.get(right_family, [])
            if not left_values or not right_values:
                continue
            left_avg = safe_mean(left_values)
            right_avg = safe_mean(right_values)
            shared_model_rows.append(
                {
                    "task_number": task_number,
                    "metric": metric,
                    "model_name": model_name,
                    "left_family": left_family,
                    "left_avg": left_avg,
                    "right_family": right_family,
                    "right_avg": right_avg,
                    "delta_left_minus_right": left_avg - right_avg,
                }
            )

    return family_metric_rows, family_model_rows, family_task_pair_rows, shared_model_rows


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    if wandb is None:
        raise RuntimeError("wandb is not installed. Install it with: pip install wandb")

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    tier3_dir = Path(args.tier3_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = discover_task_projects(tier3_dir)
    if not projects:
        raise RuntimeError(
            f"No W&B projects discovered from {tier3_dir}/rlm_task*.py or codeact_task*.py"
        )

    selected_runs: list[RunSamples] = []
    for project in projects:
        model_runs = fetch_last_runs_by_model(
            entity=args.entity,
            project=project,
            last_n_per_model=args.last_n_per_model,
            max_models=args.max_models,
            requested_models=requested_models,
        )
        if not model_runs:
            print(f"[{project}] No runs found with sample precision/recall/f1.")
            continue

        print(f"\n[{project}]")
        for model_name, runs in model_runs.items():
            print(f"  model={model_name} runs={len(runs)}")
            for run in runs:
                print(f"    - {run.run_name} ({run.run_id}) created_at={run.created_at}")
            selected_runs.extend(runs)

    if not selected_runs:
        print("\nNo qualifying runs found.")
        return

    sample_rows = build_sample_point_rows(selected_runs)
    run_level_rows = build_run_level_rows(sample_rows)
    sample_avg_rows = build_sample_avg_rows(sample_rows)
    task_avg_rows = build_task_avg_rows(sample_avg_rows)
    overall_by_model_rows, overall_all_models_rows = build_overall_rows(task_avg_rows)
    (
        family_metric_rows,
        family_model_rows,
        family_task_pair_rows,
        family_shared_model_rows,
    ) = build_family_comparison_rows(task_avg_rows)

    write_csv(
        output_dir / "selected_runs.csv",
        [
            "project",
            "task_family",
            "task_number",
            "model_name",
            "run_id",
            "run_name",
            "created_at",
            "metric",
            "sample_count",
            "run_metric_mean",
        ],
        run_level_rows,
    )
    write_csv(
        output_dir / "sample_points_selected_runs.csv",
        [
            "project",
            "task_family",
            "task_number",
            "model_name",
            "run_id",
            "run_name",
            "created_at",
            "sample_id",
            "metric",
            "value",
        ],
        sample_rows,
    )
    write_csv(
        output_dir / "sample_averages.csv",
        [
            "project",
            "task_family",
            "task_number",
            "model_name",
            "sample_id",
            "metric",
            "run_count",
            "sample_avg_over_runs",
        ],
        sample_avg_rows,
    )
    write_csv(
        output_dir / "task_averages.csv",
        [
            "project",
            "task_family",
            "task_number",
            "model_name",
            "metric",
            "sample_count",
            "task_avg_over_samples",
        ],
        task_avg_rows,
    )
    write_csv(
        output_dir / "overall_averages_by_model.csv",
        ["model_name", "metric", "task_count", "overall_avg_over_tasks"],
        overall_by_model_rows,
    )
    write_csv(
        output_dir / "overall_averages_all_models.csv",
        ["metric", "task_count_all_models", "overall_avg_all_models"],
        overall_all_models_rows,
    )
    write_csv(
        output_dir / "family_averages.csv",
        [
            "task_family",
            "metric",
            "task_count",
            "family_avg_over_tasks",
        ],
        family_metric_rows,
    )
    write_csv(
        output_dir / "family_task_pair_comparison.csv",
        [
            "task_number",
            "metric",
            "left_family",
            "left_avg",
            "right_family",
            "right_avg",
            "delta_left_minus_right",
        ],
        family_task_pair_rows,
    )
    write_csv(
        output_dir / "family_shared_model_comparison.csv",
        [
            "task_number",
            "metric",
            "model_name",
            "left_family",
            "left_avg",
            "right_family",
            "right_avg",
            "delta_left_minus_right",
        ],
        family_shared_model_rows,
    )
    write_csv(
        output_dir / "family_model_averages.csv",
        [
            "task_family",
            "model_name",
            "metric",
            "task_count",
            "family_model_avg_over_tasks",
        ],
        family_model_rows,
    )

    print(f"\nWrote tables to: {output_dir}")
    for row in overall_by_model_rows:
        print(
            "  overall_by_model "
            f"model={row['model_name']} metric={row['metric']} "
            f"avg={fmt(row['overall_avg_over_tasks'])} tasks={row['task_count']}"
        )
    if family_metric_rows:
        print("\nFamily averages:")
        for row in family_metric_rows:
            print(
                "  "
                f"family={row['task_family']} metric={row['metric']} "
                f"avg={fmt(row['family_avg_over_tasks'])} tasks={row['task_count']}"
            )
    print(
        "\nWrote comparison tables:"
        f"\n  - {output_dir / 'family_task_pair_comparison.csv'}"
        f"\n  - {output_dir / 'family_shared_model_comparison.csv'}"
    )


if __name__ == "__main__":
    main()
