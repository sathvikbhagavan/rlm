import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional

try:
    import wandb
except ImportError:  # pragma: no cover - runtime environment dependent
    wandb = None


METRICS = ("precision", "recall", "f1")
SAMPLE_METRIC_PATTERN = re.compile(r"^sample/(\d+)/(precision|recall|f1)$")
SAMPLE_COST_PATTERN = re.compile(r"^sample/(\d+)/(?:final_total_cost_usd|total_cost_usd)$")
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
    context_size: int
    sample_metrics: dict[int, dict[str, float]]
    sample_costs_usd: dict[int, float]
    samples_with_cost: int
    total_cost_usd: Optional[float]
    avg_cost_per_sample_usd: Optional[float]


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
    parser.add_argument(
        "--context-sizes",
        type=str,
        default="100,500",
        help=(
            "Comma-separated CONTEXT_SIZE values to include "
            "(e.g. '100,500'). Use 'all' to disable filtering."
        ),
    )
    parser.add_argument(
        "--min-task-number",
        type=int,
        default=None,
        help="Optional minimum task number to include (e.g. 6).",
    )
    parser.add_argument(
        "--max-task-number",
        type=int,
        default=None,
        help="Optional maximum task number to include (e.g. 10).",
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


def project_in_task_range(project: str, min_task: Optional[int], max_task: Optional[int]) -> bool:
    task_number = parse_task_number(project)
    if task_number < 0:
        return False
    if min_task is not None and task_number < min_task:
        return False
    if max_task is not None and task_number > max_task:
        return False
    return True


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


def extract_sample_costs(summary: dict) -> dict[int, float]:
    per_sample: dict[int, float] = {}
    for key, value in summary.items():
        m = SAMPLE_COST_PATTERN.match(key)
        if not m or not isinstance(value, (int, float)):
            continue
        per_sample[int(m.group(1))] = float(value)
    return per_sample


def extract_run_cost_stats(summary: dict) -> tuple[int, Optional[float], Optional[float]]:
    sample_costs = extract_sample_costs(summary)
    samples_with_cost_raw = summary.get("samples_with_cost")
    if isinstance(samples_with_cost_raw, (int, float)) and not isinstance(samples_with_cost_raw, bool):
        samples_with_cost = int(samples_with_cost_raw)
    else:
        samples_with_cost = len(sample_costs)

    total_cost_raw = summary.get("total_cost_usd")
    if isinstance(total_cost_raw, (int, float)):
        total_cost_usd: Optional[float] = float(total_cost_raw)
    elif sample_costs:
        total_cost_usd = float(sum(sample_costs.values()))
    else:
        total_cost_usd = None

    avg_cost_raw = summary.get("avg_cost_per_sample_usd")
    if isinstance(avg_cost_raw, (int, float)):
        avg_cost_per_sample_usd: Optional[float] = float(avg_cost_raw)
    elif total_cost_usd is not None and samples_with_cost > 0:
        avg_cost_per_sample_usd = total_cost_usd / samples_with_cost
    else:
        avg_cost_per_sample_usd = None

    if samples_with_cost == 0 and sample_costs:
        samples_with_cost = len(sample_costs)

    return samples_with_cost, total_cost_usd, avg_cost_per_sample_usd


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


def fetch_last_runs_by_model(
    entity: str,
    project: str,
    last_n_per_model: int,
    max_models: int,
    requested_models: list[str],
    context_sizes: list[int],
    use_context_size: bool,
) -> dict[tuple[str, int], list[RunSamples]]:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    grouped: dict[tuple[str, int], list[RunSamples]] = defaultdict(list)
    selected_models: list[str] = requested_models[:]
    target_model_count = len(requested_models) if requested_models else max_models
    saturation_context_sizes = context_sizes if use_context_size else [0]

    for run in runs:
        summary = dict(run.summary)
        sample_metrics = extract_sample_metrics(summary)
        sample_costs_usd = extract_sample_costs(summary)
        samples_with_cost, total_cost_usd, avg_cost_per_sample_usd = extract_run_cost_stats(summary)
        if not sample_metrics:
            continue

        if use_context_size:
            context_size = normalize_context_size(run)
            if context_size is None:
                continue
            if context_sizes and context_size not in context_sizes:
                continue
        else:
            context_size = 0

        model_name = normalize_model_name(run)
        if requested_models:
            if model_name not in requested_models:
                continue
        else:
            if model_name not in selected_models:
                if len(selected_models) >= max_models:
                    continue
                selected_models.append(model_name)

        group_key = (model_name, context_size)
        if len(grouped[group_key]) >= last_n_per_model:
            if _all_models_saturated(
                grouped=grouped,
                selected_models=selected_models,
                last_n_per_model=last_n_per_model,
                target_model_count=target_model_count,
                context_sizes=saturation_context_sizes,
            ):
                break
            continue

        grouped[group_key].append(
            RunSamples(
                project=project,
                task_family=parse_task_family(project),
                task_number=parse_task_number(project),
                model_name=model_name,
                run_id=run.id,
                run_name=run.name or run.id,
                created_at=run.created_at or "",
                context_size=context_size,
                sample_metrics=sample_metrics,
                sample_costs_usd=sample_costs_usd,
                samples_with_cost=samples_with_cost,
                total_cost_usd=total_cost_usd,
                avg_cost_per_sample_usd=avg_cost_per_sample_usd,
            )
        )

        if _all_models_saturated(
            grouped=grouped,
            selected_models=selected_models,
            last_n_per_model=last_n_per_model,
            target_model_count=target_model_count,
            context_sizes=saturation_context_sizes,
        ):
            break

    output: dict[tuple[str, int], list[RunSamples]] = {}
    target_context_sizes = saturation_context_sizes[:] if saturation_context_sizes else sorted(
        {ctx for (_model, ctx) in grouped}
    )
    for model in selected_models:
        for context_size in target_context_sizes:
            output[(model, context_size)] = grouped.get((model, context_size), [])
    return output


def _all_models_saturated(
    grouped: dict[tuple[str, int], list[RunSamples]],
    selected_models: list[str],
    last_n_per_model: int,
    target_model_count: int,
    context_sizes: list[int],
) -> bool:
    # Do not stop early before we have discovered the target number of models.
    if not selected_models or len(selected_models) < target_model_count:
        return False
    if not context_sizes:
        return all(
            sum(
                len(runs)
                for (model_name, _context_size), runs in grouped.items()
                if model_name == model
            )
            >= last_n_per_model
            for model in selected_models
        )
    return all(
        len(grouped.get((model, context_size), [])) >= last_n_per_model
        for model in selected_models
        for context_size in context_sizes
    )


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
                        "context_size": run.context_size,
                        "run_id": run.run_id,
                        "run_name": run.run_name,
                        "created_at": run.created_at,
                        "sample_id": sample_id,
                        "metric": metric,
                        "value": float(value),
                        "sample_cost_usd": run.sample_costs_usd.get(sample_id),
                        "run_samples_with_cost": run.samples_with_cost,
                        "run_total_cost_usd": run.total_cost_usd,
                        "run_avg_cost_per_sample_usd": run.avg_cost_per_sample_usd,
                    }
                )
    return rows


def build_run_level_rows(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str, int, str, str, str, str], list[float]] = defaultdict(list)
    grouped_costs: dict[tuple[str, str, int, str, int, str, str, str], dict[str, Optional[float]]] = {}
    for row in sample_rows:
        value = row.get("value")
        if not is_num(value):
            continue
        key = (
            str(row["project"]),
            str(row["task_family"]),
            int(row["task_number"]),
            str(row["model_name"]),
            int(row["context_size"]),
            str(row["run_id"]),
            str(row["run_name"]),
            str(row["created_at"]),
            str(row["metric"]),
        )
        grouped[key].append(float(value))
        run_key = key[:-1]
        if run_key not in grouped_costs:
            grouped_costs[run_key] = {
                "run_samples_with_cost": (
                    float(row["run_samples_with_cost"])
                    if is_num(row.get("run_samples_with_cost"))
                    else None
                ),
                "run_total_cost_usd": (
                    float(row["run_total_cost_usd"])
                    if is_num(row.get("run_total_cost_usd"))
                    else None
                ),
                "run_avg_cost_per_sample_usd": (
                    float(row["run_avg_cost_per_sample_usd"])
                    if is_num(row.get("run_avg_cost_per_sample_usd"))
                    else None
                ),
            }

    rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        project, task_family, task_number, model_name, context_size, run_id, run_name, created_at, metric = key
        cost_row = grouped_costs.get(key[:-1], {})
        rows.append(
            {
                "project": project,
                "task_family": task_family,
                "task_number": task_number,
                "model_name": model_name,
                "context_size": context_size,
                "run_id": run_id,
                "run_name": run_name,
                "created_at": created_at,
                "metric": metric,
                "sample_count": len(values),
                "run_metric_mean": safe_mean(values),
                "run_samples_with_cost": int(cost_row["run_samples_with_cost"])
                if isinstance(cost_row.get("run_samples_with_cost"), float)
                else 0,
                "run_total_cost_usd": cost_row.get("run_total_cost_usd"),
                "run_avg_cost_per_sample_usd": cost_row.get("run_avg_cost_per_sample_usd"),
            }
        )
    return rows


def build_run_cost_rows(runs: list[RunSamples]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in sorted(
        runs,
        key=lambda r: (r.project, r.task_family, r.task_number, r.model_name, r.context_size, r.created_at, r.run_id),
    ):
        rows.append(
            {
                "project": run.project,
                "task_family": run.task_family,
                "task_number": run.task_number,
                "model_name": run.model_name,
                "context_size": run.context_size,
                "run_id": run.run_id,
                "run_name": run.run_name,
                "created_at": run.created_at,
                "sample_count": len(run.sample_metrics),
                "samples_with_cost": run.samples_with_cost,
                "total_cost_usd": run.total_cost_usd,
                "avg_cost_per_sample_usd": run.avg_cost_per_sample_usd,
            }
        )
    return rows


def build_task_cost_avg_rows(run_cost_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped_total_costs: dict[tuple[str, str, int, str, int], list[float]] = defaultdict(list)
    grouped_avg_costs: dict[tuple[str, str, int, str, int], list[float]] = defaultdict(list)
    run_counts: dict[tuple[str, str, int, str, int], int] = defaultdict(int)
    runs_with_cost_counts: dict[tuple[str, str, int, str, int], int] = defaultdict(int)

    for row in run_cost_rows:
        key = (
            str(row["project"]),
            str(row["task_family"]),
            int(row["task_number"]),
            str(row["model_name"]),
            int(row["context_size"]),
        )
        run_counts[key] += 1

        total_cost = row.get("total_cost_usd")
        if is_num(total_cost):
            grouped_total_costs[key].append(float(total_cost))
            runs_with_cost_counts[key] += 1

        avg_cost = row.get("avg_cost_per_sample_usd")
        if is_num(avg_cost):
            grouped_avg_costs[key].append(float(avg_cost))

    rows: list[dict[str, object]] = []
    for key in sorted(run_counts):
        project, task_family, task_number, model_name, context_size = key
        rows.append(
            {
                "project": project,
                "task_family": task_family,
                "task_number": task_number,
                "model_name": model_name,
                "context_size": context_size,
                "run_count": run_counts[key],
                "runs_with_cost": runs_with_cost_counts.get(key, 0),
                "avg_total_cost_per_run_usd": safe_mean(grouped_total_costs.get(key, [])),
                "avg_cost_per_sample_usd": safe_mean(grouped_avg_costs.get(key, [])),
            }
        )
    return rows


def build_overall_cost_rows(
    run_cost_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_model_total_costs: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_model_avg_costs: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_model_run_counts: dict[tuple[str, int], int] = defaultdict(int)
    by_model_runs_with_cost: dict[tuple[str, int], int] = defaultdict(int)

    all_model_total_costs: dict[int, list[float]] = defaultdict(list)
    all_model_avg_costs: dict[int, list[float]] = defaultdict(list)
    all_model_run_counts: dict[int, int] = defaultdict(int)
    all_model_runs_with_cost: dict[int, int] = defaultdict(int)

    for row in run_cost_rows:
        model_name = str(row["model_name"])
        context_size = int(row["context_size"])
        key = (model_name, context_size)

        by_model_run_counts[key] += 1
        all_model_run_counts[context_size] += 1

        total_cost = row.get("total_cost_usd")
        if is_num(total_cost):
            by_model_total_costs[key].append(float(total_cost))
            by_model_runs_with_cost[key] += 1
            all_model_total_costs[context_size].append(float(total_cost))
            all_model_runs_with_cost[context_size] += 1

        avg_cost = row.get("avg_cost_per_sample_usd")
        if is_num(avg_cost):
            by_model_avg_costs[key].append(float(avg_cost))
            all_model_avg_costs[context_size].append(float(avg_cost))

    by_model_rows: list[dict[str, object]] = []
    for model_name, context_size in sorted(by_model_run_counts):
        key = (model_name, context_size)
        by_model_rows.append(
            {
                "model_name": model_name,
                "context_size": context_size,
                "run_count": by_model_run_counts[key],
                "runs_with_cost": by_model_runs_with_cost.get(key, 0),
                "avg_total_cost_per_run_usd": safe_mean(by_model_total_costs.get(key, [])),
                "avg_cost_per_sample_usd": safe_mean(by_model_avg_costs.get(key, [])),
            }
        )

    all_rows: list[dict[str, object]] = []
    for context_size in sorted(all_model_run_counts):
        all_rows.append(
            {
                "context_size": context_size,
                "run_count_all_models": all_model_run_counts[context_size],
                "runs_with_cost_all_models": all_model_runs_with_cost.get(context_size, 0),
                "avg_total_cost_per_run_all_models_usd": safe_mean(all_model_total_costs.get(context_size, [])),
                "avg_cost_per_sample_all_models_usd": safe_mean(all_model_avg_costs.get(context_size, [])),
            }
        )
    return by_model_rows, all_rows


def build_sample_avg_rows(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str, int, int, str], list[float]] = defaultdict(list)
    run_sets: dict[tuple[str, str, int, str, int, int, str], set[str]] = defaultdict(set)
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
            int(row["context_size"]),
            sample_id,
            str(row["metric"]),
        )
        grouped[key].append(float(value))
        run_sets[key].add(str(row["run_id"]))

    rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        project, task_family, task_number, model_name, context_size, sample_id, metric = key
        rows.append(
            {
                "project": project,
                "task_family": task_family,
                "task_number": task_number,
                "model_name": model_name,
                "context_size": context_size,
                "sample_id": sample_id,
                "metric": metric,
                "run_count": len(run_sets[key]),
                "sample_avg_over_runs": safe_mean(values),
            }
        )
    return rows


def build_task_avg_rows(sample_avg_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str, int, str], list[float]] = defaultdict(list)
    for row in sample_avg_rows:
        value = row.get("sample_avg_over_runs")
        if not is_num(value):
            continue
        key = (
            str(row["project"]),
            str(row["task_family"]),
            int(row["task_number"]),
            str(row["model_name"]),
            int(row["context_size"]),
            str(row["metric"]),
        )
        grouped[key].append(float(value))

    rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        project, task_family, task_number, model_name, context_size, metric = key
        rows.append(
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
    return rows


def build_overall_rows(task_avg_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_model_grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    all_models_grouped: dict[tuple[int, str], list[float]] = defaultdict(list)

    for row in task_avg_rows:
        value = row.get("task_avg_over_samples")
        if not is_num(value):
            continue
        model_name = str(row["model_name"])
        context_size = int(row["context_size"])
        metric = str(row["metric"])
        by_model_grouped[(model_name, context_size, metric)].append(float(value))
        all_models_grouped[(context_size, metric)].append(float(value))

    by_model_rows: list[dict[str, object]] = []
    for key, values in sorted(by_model_grouped.items()):
        model_name, context_size, metric = key
        by_model_rows.append(
            {
                "model_name": model_name,
                "context_size": context_size,
                "metric": metric,
                "task_count": len(values),
                "overall_avg_over_tasks": safe_mean(values),
            }
        )

    all_rows: list[dict[str, object]] = []
    for context_size, metric in sorted(all_models_grouped):
        values = all_models_grouped.get((context_size, metric), [])
        all_rows.append(
            {
                "context_size": context_size,
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
    family_metric_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    family_model_metric_values: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    task_pair_values: dict[tuple[int, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in task_avg_rows:
        value = row.get("task_avg_over_samples")
        if not is_num(value):
            continue
        family = str(row.get("task_family", "unknown"))
        context_size = int(row.get("context_size", -1))
        metric = str(row["metric"])
        model_name = str(row["model_name"])
        task_number = int(row.get("task_number", -1))
        numeric_value = float(value)

        family_metric_values[(family, context_size, metric)].append(numeric_value)
        family_model_metric_values[(family, model_name, context_size, metric)].append(numeric_value)
        task_pair_values[(task_number, context_size, metric)][family].append(numeric_value)

    family_metric_rows: list[dict[str, object]] = []
    for key, values in sorted(family_metric_values.items()):
        family, context_size, metric = key
        family_metric_rows.append(
            {
                "task_family": family,
                "context_size": context_size,
                "metric": metric,
                "task_count": len(values),
                "family_avg_over_tasks": safe_mean(values),
            }
        )

    family_model_rows: list[dict[str, object]] = []
    for key, values in sorted(family_model_metric_values.items()):
        family, model_name, context_size, metric = key
        family_model_rows.append(
            {
                "task_family": family,
                "model_name": model_name,
                "context_size": context_size,
                "metric": metric,
                "task_count": len(values),
                "family_model_avg_over_tasks": safe_mean(values),
            }
        )

    # Pairwise comparison only when both families exist for a task number/metric.
    family_task_pair_rows: list[dict[str, object]] = []
    shared_model_rows: list[dict[str, object]] = []
    families = sorted({family for family, _context_size, _metric in family_metric_values})
    if len(families) >= 2:
        left_family = families[0]
        right_family = families[1]
        for (task_number, context_size, metric), family_map in sorted(task_pair_values.items()):
            left_values = family_map.get(left_family, [])
            right_values = family_map.get(right_family, [])
            if not left_values or not right_values:
                continue
            left_avg = safe_mean(left_values)
            right_avg = safe_mean(right_values)
            family_task_pair_rows.append(
                {
                    "task_number": task_number,
                    "context_size": context_size,
                    "metric": metric,
                    "left_family": left_family,
                    "left_avg": left_avg,
                    "right_family": right_family,
                    "right_avg": right_avg,
                    "delta_left_minus_right": left_avg - right_avg,
                }
            )

        # Shared-model comparison: compare only models present in both families.
        shared_grouped: dict[tuple[int, int, str, str], dict[str, list[float]]] = defaultdict(
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
            context_size = int(row.get("context_size", -1))
            metric = str(row.get("metric"))
            model_name = str(row.get("model_name"))
            shared_grouped[(task_number, context_size, metric, model_name)][family].append(float(value))

        for (task_number, context_size, metric, model_name), fam_map in sorted(shared_grouped.items()):
            left_values = fam_map.get(left_family, [])
            right_values = fam_map.get(right_family, [])
            if not left_values or not right_values:
                continue
            left_avg = safe_mean(left_values)
            right_avg = safe_mean(right_values)
            shared_model_rows.append(
                {
                    "task_number": task_number,
                    "context_size": context_size,
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
    context_sizes_raw = args.context_sizes.strip().lower()
    if context_sizes_raw == "all":
        context_sizes: list[int] = []
    else:
        context_sizes = [int(v.strip()) for v in args.context_sizes.split(",") if v.strip()]
    tier3_dir = Path(args.tier3_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = discover_task_projects(tier3_dir)
    if args.min_task_number is not None or args.max_task_number is not None:
        projects = [
            project
            for project in projects
            if project_in_task_range(project, args.min_task_number, args.max_task_number)
        ]
    if not projects:
        raise RuntimeError(
            f"No W&B projects discovered from {tier3_dir}/rlm_task*.py or codeact_task*.py"
        )

    selected_runs: list[RunSamples] = []
    for project in projects:
        use_context_size = "codeact" in parse_task_family(project).lower()
        model_runs = fetch_last_runs_by_model(
            entity=args.entity,
            project=project,
            last_n_per_model=args.last_n_per_model,
            max_models=args.max_models,
            requested_models=requested_models,
            context_sizes=context_sizes,
            use_context_size=use_context_size,
        )
        if not model_runs:
            print(f"[{project}] No runs found with sample precision/recall/f1.")
            continue

        print(f"\n[{project}]")
        for (model_name, context_size), runs in model_runs.items():
            if use_context_size:
                print(f"  model={model_name} context_size={context_size} runs={len(runs)}")
            else:
                print(f"  model={model_name} runs={len(runs)}")
            for run in runs:
                print(f"    - {run.run_name} ({run.run_id}) created_at={run.created_at}")
            selected_runs.extend(runs)

    if not selected_runs:
        print("\nNo qualifying runs found.")
        return

    sample_rows = build_sample_point_rows(selected_runs)
    run_level_rows = build_run_level_rows(sample_rows)
    run_cost_rows = build_run_cost_rows(selected_runs)
    sample_avg_rows = build_sample_avg_rows(sample_rows)
    task_avg_rows = build_task_avg_rows(sample_avg_rows)
    overall_by_model_rows, overall_all_models_rows = build_overall_rows(task_avg_rows)
    task_cost_avg_rows = build_task_cost_avg_rows(run_cost_rows)
    overall_cost_by_model_rows, overall_cost_all_models_rows = build_overall_cost_rows(run_cost_rows)
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
            "context_size",
            "run_id",
            "run_name",
            "created_at",
            "metric",
            "sample_count",
            "run_metric_mean",
            "run_samples_with_cost",
            "run_total_cost_usd",
            "run_avg_cost_per_sample_usd",
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
            "context_size",
            "run_id",
            "run_name",
            "created_at",
            "sample_id",
            "metric",
            "value",
            "sample_cost_usd",
            "run_samples_with_cost",
            "run_total_cost_usd",
            "run_avg_cost_per_sample_usd",
        ],
        sample_rows,
    )
    write_csv(
        output_dir / "run_costs.csv",
        [
            "project",
            "task_family",
            "task_number",
            "model_name",
            "context_size",
            "run_id",
            "run_name",
            "created_at",
            "sample_count",
            "samples_with_cost",
            "total_cost_usd",
            "avg_cost_per_sample_usd",
        ],
        run_cost_rows,
    )
    write_csv(
        output_dir / "sample_averages.csv",
        [
            "project",
            "task_family",
            "task_number",
            "model_name",
            "context_size",
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
            "context_size",
            "metric",
            "sample_count",
            "task_avg_over_samples",
        ],
        task_avg_rows,
    )
    write_csv(
        output_dir / "overall_averages_by_model.csv",
        ["model_name", "context_size", "metric", "task_count", "overall_avg_over_tasks"],
        overall_by_model_rows,
    )
    write_csv(
        output_dir / "overall_averages_all_models.csv",
        ["context_size", "metric", "task_count_all_models", "overall_avg_all_models"],
        overall_all_models_rows,
    )
    write_csv(
        output_dir / "task_cost_averages.csv",
        [
            "project",
            "task_family",
            "task_number",
            "model_name",
            "context_size",
            "run_count",
            "runs_with_cost",
            "avg_total_cost_per_run_usd",
            "avg_cost_per_sample_usd",
        ],
        task_cost_avg_rows,
    )
    write_csv(
        output_dir / "overall_cost_averages_by_model.csv",
        [
            "model_name",
            "context_size",
            "run_count",
            "runs_with_cost",
            "avg_total_cost_per_run_usd",
            "avg_cost_per_sample_usd",
        ],
        overall_cost_by_model_rows,
    )
    write_csv(
        output_dir / "overall_cost_averages_all_models.csv",
        [
            "context_size",
            "run_count_all_models",
            "runs_with_cost_all_models",
            "avg_total_cost_per_run_all_models_usd",
            "avg_cost_per_sample_all_models_usd",
        ],
        overall_cost_all_models_rows,
    )
    write_csv(
        output_dir / "family_averages.csv",
        [
            "task_family",
            "context_size",
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
            "context_size",
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
            "context_size",
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
            "context_size",
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
            f"model={row['model_name']} context_size={row['context_size']} metric={row['metric']} "
            f"avg={fmt(row['overall_avg_over_tasks'])} tasks={row['task_count']}"
        )
    if family_metric_rows:
        print("\nFamily averages:")
        for row in family_metric_rows:
            print(
                "  "
                f"family={row['task_family']} context_size={row['context_size']} metric={row['metric']} "
                f"avg={fmt(row['family_avg_over_tasks'])} tasks={row['task_count']}"
            )
    print(
        "\nWrote comparison tables:"
        f"\n  - {output_dir / 'family_task_pair_comparison.csv'}"
        f"\n  - {output_dir / 'family_shared_model_comparison.csv'}"
    )


if __name__ == "__main__":
    main()
