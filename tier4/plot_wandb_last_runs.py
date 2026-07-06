"""Plot tier4 overall F1/cost/token trends from latest W&B runs.

Averages the last N completed runs per task/family/model/context (N=5 by default),
then aggregates across tasks with question-count-weighted F1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Optional

import wandb

try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    plt = None


TASK_IDS = ("11", "12", "12b", "13", "14", "15", "16", "17", "17b")
SUBGROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Mechanical Graph", ("11", "12", "12b")),
    ("Chemically Constrained", ("13", "14", "15")),
    ("Prospective / Multi-Constraint", ("16", "17", "17b")),
)
FAMILY_PREFIX = {"LLM": "LLM", "CodeAct": "CodeAct", "RLM": "RLMs"}
FAMILY_COLORS = {"LLM": "#1f77b4", "CodeAct": "#d62728", "RLM": "#2ca02c"}
PROJECT_PATTERN = re.compile(r'project\s*=\s*"([^"]+)"')
TASK_SCRIPT_PATTERN = re.compile(r"(?:rlm|llm|codeact)_task(\d+b?)\.py$")
CONFIG_FIELD_PATTERN = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)

# Fallback question counts when a run summary omits `total`.
TASK_QUESTION_COUNTS: dict[str, int] = {
    "11": 2,
    "12": 2,
    "12b": 1,
    "13": 4,
    "14": 2,
    "15": 4,
    "16": 10,
    "17": 5,
    "17b": 5,
}

# Task-specific overrides take precedence (e.g. RLM task16 @ full context).
MIN_RUNS_OVERRIDES: dict[tuple[str, str, int], int] = {
    ("RLM", "16", -1): 3,
}


@dataclass
class RunRecord:
    family: str
    task_id: str
    model_name: str
    context_size: int
    question_count: float
    f1: float
    cost_per_sample_usd: Optional[float]
    input_tokens_per_sample: Optional[float]
    output_tokens_per_sample: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot tier4 overall F1/cost/token trends vs context from latest W&B runs."
    )
    parser.add_argument("--entity", type=str, default=os.getenv("WANDB_ENTITY", ""))
    parser.add_argument(
        "--tier4-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/tier4",
        help="Directory containing tier4 task scripts (used to discover W&B project names).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="openai/gpt-5-mini",
        help="Comma-separated model names, or 'all'.",
    )
    parser.add_argument(
        "--last-n-runs",
        type=int,
        default=5,
        help="Take last N completed runs for each task/family/model/context.",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=("api", "local"),
        default="local",
        help="Read runs from W&B API or local tier4/wandb offline runs.",
    )
    parser.add_argument(
        "--local-wandb-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/tier4/wandb",
        help="Directory of local offline W&B run folders when --source local.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/tier4/plots_last5",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        default=True,
        help="Plot with available runs instead of failing when a group has < last-n-runs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any group has fewer completed runs than required.",
    )
    return parser.parse_args()


def task_sort_key(task_id: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)(.*)$", task_id)
    if not match:
        return (10**9, task_id)
    return (int(match.group(1)), match.group(2))


def parse_task_id(project: str) -> Optional[str]:
    match = re.match(r"^(?:LLM|CodeAct|RLMs)-Task(\d+b?)$", project)
    if match:
        return match.group(1)
    return None


def parse_family(project: str) -> Optional[str]:
    if project.startswith("LLM-"):
        return "LLM"
    if project.startswith("CodeAct-"):
        return "CodeAct"
    if project.startswith("RLMs-"):
        return "RLM"
    return None


def project_name(family: str, task_id: str) -> str:
    return f"{FAMILY_PREFIX[family]}-Task{task_id}"


def required_runs_for_group(
    family: str,
    task_id: str,
    context_size: int,
    *,
    default: int,
) -> int:
    if (family, task_id, context_size) in MIN_RUNS_OVERRIDES:
        return MIN_RUNS_OVERRIDES[(family, task_id, context_size)]
    return default


def discover_projects(tier4_dir: Path, task_ids: tuple[str, ...]) -> dict[str, dict[str, str]]:
    discovered: dict[str, dict[str, str]] = defaultdict(dict)
    wanted = set(task_ids)
    for pattern in ("rlm_task*.py", "llm_task*.py", "codeact_task*.py"):
        for path in sorted(tier4_dir.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            for match in PROJECT_PATTERN.finditer(text):
                project = match.group(1)
                family = parse_family(project)
                task_id = parse_task_id(project)
                if family is None or task_id is None or task_id not in wanted:
                    continue
                discovered[task_id][family] = project
    return discovered


def normalize_model_name(config: dict) -> str:
    for key in ("MODEL_NAME", "model_name", "model", "llm_model"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown_model"


def normalize_context_size(config: dict) -> Optional[int]:
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


def safe_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        out = float(value)
        if not math.isnan(out):
            return out
    return None


def question_count_from_run(config: dict, summary: dict, task_id: str) -> Optional[float]:
    for source in (summary, config):
        value = safe_float(source.get("total"))
        if value is not None and value > 0:
            return value
    value = safe_float(config.get("num_questions"))
    if value is not None and value > 0:
        return value
    fallback = TASK_QUESTION_COUNTS.get(task_id)
    if fallback is not None and fallback > 0:
        return float(fallback)
    return None


def context_allowed(family: str, context_size: int) -> bool:
    if family in {"LLM", "CodeAct"}:
        return context_size in {100, 500}
    if family == "RLM":
        return context_size in {100, 500, -1}
    return False


def context_sort_key(context_size: int) -> int:
    if context_size == -1:
        return 10_000
    return context_size


def context_label(context_size: int) -> str:
    return "full" if context_size == -1 else str(context_size)


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
        }
    )


def weighted_mean(values: list[tuple[float, float]]) -> Optional[float]:
    if not values:
        return None
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in values) / total_weight


def mean_or_nan(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def bounded_f1_yerr(
    means: list[float], stds: list[float], lower_bound: float = 0.0, upper_bound: float = 1.0
) -> list[list[float]]:
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for m, s in zip(means, stds):
        lower_errors.append(min(s, max(0.0, m - lower_bound)))
        upper_errors.append(min(s, max(0.0, upper_bound - m)))
    return [lower_errors, upper_errors]


def parse_flat_config_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    config: dict[str, object] = {}

    for match in re.finditer(r"^([\w_]+):\s*\n\s*value:\s*(.+)$", text, re.MULTILINE):
        key = match.group(1)
        raw = match.group(2).strip()
        if raw.startswith('"') and raw.endswith('"'):
            config[key] = raw[1:-1]
        elif raw.lower() in {"true", "false"}:
            config[key] = raw.lower() == "true"
        else:
            try:
                if "." in raw:
                    config[key] = float(raw)
                else:
                    config[key] = int(raw)
            except ValueError:
                config[key] = raw

    if config:
        return config

    for match in CONFIG_FIELD_PATTERN.finditer(text):
        key = match.group(1)
        raw = match.group(2).strip()
        if key == "_wandb":
            continue
        if raw.startswith('"') and raw.endswith('"'):
            config[key] = raw[1:-1]
        elif raw.lower() in {"true", "false"}:
            config[key] = raw.lower() == "true"
        else:
            try:
                if "." in raw:
                    config[key] = float(raw)
                else:
                    config[key] = int(raw)
            except ValueError:
                config[key] = raw
    return config


def family_from_script_name(script_name: str) -> Optional[str]:
    if script_name.startswith("rlm_task"):
        return "RLM"
    if script_name.startswith("llm_task"):
        return "LLM"
    if script_name.startswith("codeact_task"):
        return "CodeAct"
    return None


def task_id_from_script_name(script_name: str) -> Optional[str]:
    match = TASK_SCRIPT_PATTERN.search(script_name)
    if match:
        return match.group(1)
    return None


def run_sort_key(run_dir: Path) -> str:
    return run_dir.name


def append_record(
    records: list[RunRecord],
    seen_counts: dict[tuple[str, str, str, int], int],
    *,
    family: str,
    task_id: str,
    model_name: str,
    context_size: int,
    question_count: float,
    f1: float,
    cost_per_sample_usd: Optional[float],
    input_tokens_per_sample: Optional[float],
    output_tokens_per_sample: Optional[float],
    last_n_runs: int,
    models_filter: set[str],
) -> bool:
    if task_id not in TASK_IDS:
        return False
    if models_filter and model_name not in models_filter:
        return False
    if not context_allowed(family, context_size):
        return False
    key = (family, task_id, model_name, context_size)
    required = required_runs_for_group(
        family, task_id, context_size, default=last_n_runs
    )
    if seen_counts[key] >= required:
        return False
    seen_counts[key] += 1
    records.append(
        RunRecord(
            family=family,
            task_id=task_id,
            model_name=model_name,
            context_size=context_size,
            question_count=question_count,
            f1=f1,
            cost_per_sample_usd=cost_per_sample_usd,
            input_tokens_per_sample=input_tokens_per_sample,
            output_tokens_per_sample=output_tokens_per_sample,
        )
    )
    return True


def finalize_collection(
    seen_counts: dict[tuple[str, str, str, int], int],
    last_n_runs: int,
    allow_partial: bool,
) -> None:
    if allow_partial:
        return
    insufficient_groups = []
    for key, count in seen_counts.items():
        family, task_id, model, context = key
        required = required_runs_for_group(
            family, task_id, context, default=last_n_runs
        )
        if count < required:
            insufficient_groups.append(key)
    if insufficient_groups:
        formatted = ", ".join(
            [
                (
                    f"{family}/task{task}/model={model}/ctx={context} "
                    f"(found {seen_counts[(family, task, model, context)]}, "
                    f"need {required_runs_for_group(family, task, context, default=last_n_runs)})"
                )
                for (family, task, model, context) in sorted(
                    insufficient_groups, key=lambda x: (x[0], task_sort_key(x[1]), x[2], x[3])
                )
            ]
        )
        raise RuntimeError(
            "Not enough completed runs for strict averaging. Missing groups: "
            f"{formatted}"
        )


def collect_records(
    entity: str,
    projects_by_task: dict[str, dict[str, str]],
    models_filter: set[str],
    last_n_runs: int,
    allow_partial: bool,
) -> list[RunRecord]:
    api = wandb.Api()
    records: list[RunRecord] = []
    seen_counts: dict[tuple[str, str, str, int], int] = defaultdict(int)

    for task_id in TASK_IDS:
        family_projects = projects_by_task.get(task_id, {})
        for family in ("LLM", "CodeAct", "RLM"):
            project = family_projects.get(family) or project_name(family, task_id)
            runs = api.runs(f"{entity}/{project}", order="-created_at")
            for run in runs:
                config = dict(run.config or {})
                summary = dict(run.summary or {})
                model_name = normalize_model_name(config)
                context_size = normalize_context_size(config)
                if context_size is None:
                    continue
                question_count = question_count_from_run(config, summary, task_id)
                f1 = safe_float(summary.get("macro_f1"))
                if question_count is None or f1 is None:
                    continue
                append_record(
                    records,
                    seen_counts,
                    family=family,
                    task_id=task_id,
                    model_name=model_name,
                    context_size=context_size,
                    question_count=question_count,
                    f1=f1,
                    cost_per_sample_usd=safe_float(summary.get("avg_cost_per_sample_usd")),
                    input_tokens_per_sample=safe_float(
                        summary.get("avg_total_input_tokens_per_sample")
                    ),
                    output_tokens_per_sample=safe_float(
                        summary.get("avg_total_output_tokens_per_sample")
                    ),
                    last_n_runs=last_n_runs,
                    models_filter=models_filter,
                )

    finalize_collection(seen_counts, last_n_runs, allow_partial)
    return records


def collect_records_from_local(
    local_wandb_dir: Path,
    models_filter: set[str],
    last_n_runs: int,
    allow_partial: bool,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    seen_counts: dict[tuple[str, str, str, int], int] = defaultdict(int)
    run_dirs = sorted(
        [path for path in local_wandb_dir.glob("run-*") if path.is_dir()],
        key=run_sort_key,
        reverse=True,
    )

    for run_dir in run_dirs:
        metadata_path = run_dir / "files" / "wandb-metadata.json"
        summary_path = run_dir / "files" / "wandb-summary.json"
        config_path = run_dir / "files" / "config.yaml"
        if not metadata_path.exists() or not summary_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = parse_flat_config_yaml(config_path)

        code_path = str(metadata.get("codePathLocal") or metadata.get("codePath") or "")
        script_name = Path(code_path).name
        family = family_from_script_name(script_name)
        task_id = task_id_from_script_name(script_name)
        if family is None or task_id is None:
            continue

        model_name = normalize_model_name(config)
        context_size = normalize_context_size(config)
        if context_size is None:
            continue
        question_count = question_count_from_run(config, summary, task_id)
        f1 = safe_float(summary.get("macro_f1"))
        if question_count is None or f1 is None:
            continue

        append_record(
            records,
            seen_counts,
            family=family,
            task_id=task_id,
            model_name=model_name,
            context_size=context_size,
            question_count=question_count,
            f1=f1,
            cost_per_sample_usd=safe_float(summary.get("avg_cost_per_sample_usd")),
            input_tokens_per_sample=safe_float(summary.get("avg_total_input_tokens_per_sample")),
            output_tokens_per_sample=safe_float(summary.get("avg_total_output_tokens_per_sample")),
            last_n_runs=last_n_runs,
            models_filter=models_filter,
        )

    finalize_collection(seen_counts, last_n_runs, allow_partial)
    return records


def aggregate_task(records: list[RunRecord]) -> dict[tuple[str, str, int], dict[str, float]]:
    grouped: dict[tuple[str, str, int], list[RunRecord]] = defaultdict(list)
    for row in records:
        grouped[(row.family, row.task_id, row.context_size)].append(row)

    out: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, rows in grouped.items():
        f1_values = [r.f1 for r in rows]
        out[key] = {
            "question_count": mean([r.question_count for r in rows]),
            "run_count": len(rows),
            "f1": mean(f1_values),
            "f1_std": pstdev(f1_values) if len(f1_values) > 1 else 0.0,
            "cost": mean_or_nan(
                [r.cost_per_sample_usd for r in rows if r.cost_per_sample_usd is not None]
            ),
            "input_tokens": mean_or_nan(
                [r.input_tokens_per_sample for r in rows if r.input_tokens_per_sample is not None]
            ),
            "output_tokens": mean_or_nan(
                [r.output_tokens_per_sample for r in rows if r.output_tokens_per_sample is not None]
            ),
        }
    return out


def aggregate_overall(
    task_agg: dict[tuple[str, str, int], dict[str, float]],
    *,
    task_filter: Optional[frozenset[str]] = None,
) -> dict[tuple[str, int], dict[str, Optional[float]]]:
    grouped: dict[tuple[str, int], list[tuple[str, dict[str, float]]]] = defaultdict(list)
    for (family, task_id, context), stats in task_agg.items():
        if task_filter is not None and task_id not in task_filter:
            continue
        grouped[(family, context)].append((task_id, stats))

    out: dict[tuple[str, int], dict[str, Optional[float]]] = {}
    for key, rows in grouped.items():
        f1_pairs = [(stats["f1"], stats["question_count"]) for _, stats in rows]
        total_weight = sum(stats["question_count"] for _, stats in rows)
        f1_var = 0.0
        if total_weight > 0:
            for _, stats in rows:
                w_norm = stats["question_count"] / total_weight
                f1_var += (w_norm**2) * (float(stats.get("f1_std", 0.0)) ** 2)
        cost_pairs = [
            (stats["cost"], stats["question_count"])
            for _, stats in rows
            if not math.isnan(stats["cost"])
        ]
        input_pairs = [
            (stats["input_tokens"], stats["question_count"])
            for _, stats in rows
            if not math.isnan(stats["input_tokens"])
        ]
        output_pairs = [
            (stats["output_tokens"], stats["question_count"])
            for _, stats in rows
            if not math.isnan(stats["output_tokens"])
        ]
        out[key] = {
            "f1": weighted_mean(f1_pairs),
            "f1_std": math.sqrt(f1_var) if total_weight > 0 else None,
            "cost": weighted_mean(cost_pairs),
            "input_tokens": weighted_mean(input_pairs),
            "output_tokens": weighted_mean(output_pairs),
            "task_count": len(rows),
            "question_count": total_weight,
        }
    return out


def question_count_for_tasks(
    task_agg: dict[tuple[str, str, int], dict[str, float]], task_ids: tuple[str, ...]
) -> int:
    per_task: dict[str, float] = {}
    for (_, task_id, _), stats in task_agg.items():
        if task_id in task_ids and task_id not in per_task:
            per_task[task_id] = stats["question_count"]
    if per_task:
        return int(round(sum(per_task.values())))
    return int(round(sum(TASK_QUESTION_COUNTS.get(task_id, 0) for task_id in task_ids)))


def series_for_task(
    task_agg: dict[tuple[str, str, int], dict[str, float]], task_id: str
) -> dict[tuple[str, int], dict[str, Optional[float]]]:
    out: dict[tuple[str, int], dict[str, Optional[float]]] = {}
    for (family, tid, context), stats in task_agg.items():
        if tid != task_id:
            continue
        out[(family, context)] = {
            "f1": stats["f1"],
            "f1_std": stats.get("f1_std"),
            "cost": stats["cost"] if not math.isnan(stats["cost"]) else None,
            "input_tokens": stats["input_tokens"]
            if not math.isnan(stats["input_tokens"])
            else None,
            "output_tokens": stats["output_tokens"]
            if not math.isnan(stats["output_tokens"])
            else None,
        }
    return out


def plot_f1_lines_on_ax(
    ax,
    series: dict[tuple[str, int], dict[str, Optional[float]]],
    *,
    title: str,
    subtitle: Optional[str] = None,
    show_legend: bool = False,
    show_ylabel: bool = True,
) -> None:
    for family in ("LLM", "CodeAct", "RLM"):
        points = sorted(
            [
                (context, stats["f1"], stats.get("f1_std"))
                for (fam, context), stats in series.items()
                if fam == family and stats.get("f1") is not None
            ],
            key=lambda x: context_sort_key(x[0]),
        )
        if not points:
            continue
        xs = [context_label(c) for c, _, _ in points]
        ys = [float(v) for _, v, _ in points]
        ystd = [float(e) if e is not None else 0.0 for _, _, e in points]
        yerr = bounded_f1_yerr(ys, ystd)
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            linewidth=1.6,
            capsize=2.5,
            label=family,
            color=FAMILY_COLORS[family],
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([i / 10 for i in range(0, 11)])
    ax.grid(which="major", axis="both", alpha=0.35)
    ax.minorticks_on()
    ax.grid(which="minor", axis="y", alpha=0.18, linestyle=":")
    ax.set_xlabel("Context")
    if show_ylabel:
        ax.set_ylabel("F1")
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=10, pad=8)
    if show_legend:
        ax.legend()


def shared_family_legend(
    fig, *, y: float = 0.01, loc: str = "lower center"
) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color=FAMILY_COLORS[family],
            marker="o",
            linewidth=1.6,
            label=family,
        )
        for family in ("LLM", "CodeAct", "RLM")
    ]
    fig.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=(0.5, y),
        ncol=3,
        frameon=True,
        framealpha=0.95,
    )


def plot_subcategory_f1_panels(
    out_dir: Path,
    model_name: str,
    task_agg: dict[tuple[str, str, int], dict[str, float]],
    subcategories: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    tier_label: str,
    nrows: int,
    ncols: int,
    suptitle_text: Optional[str] = None,
) -> None:
    if nrows == 1:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 6.2))
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows))
    axes_flat = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    for ax, (name, task_ids) in zip(axes_flat, subcategories):
        subset = aggregate_overall(task_agg, task_filter=frozenset(task_ids))
        qcount = question_count_for_tasks(task_agg, task_ids)
        panel_title = f"{name} ({qcount}Q)" if qcount else name
        plot_f1_lines_on_ax(ax, subset, title=panel_title, show_ylabel=True)
    for ax in axes_flat[len(subcategories) :]:
        ax.set_visible(False)
    title = suptitle_text or f"{tier_label} sub-group F1 vs context ({model_name})"
    if nrows == 1:
        fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.16, wspace=0.28)
        fig.suptitle(title, y=0.94, fontsize=14)
        shared_family_legend(fig, y=0.03)
    else:
        fig.subplots_adjust(
            left=0.08, right=0.98, top=0.92, bottom=0.12, hspace=0.42, wspace=0.28
        )
        fig.suptitle(title, y=0.98, fontsize=14)
        shared_family_legend(fig)
    save_plot(fig, out_dir, f"{model_slug(model_name)}_2_subcategory_f1_vs_context")


def plot_task_f1_panels(
    out_dir: Path,
    model_name: str,
    task_agg: dict[tuple[str, str, int], dict[str, float]],
    task_ids: tuple[str, ...],
    *,
    tier_label: str,
    nrows: int,
    ncols: int,
) -> None:
    sorted_tasks = sorted(task_ids, key=task_sort_key)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), constrained_layout=True
    )
    axes_flat = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    for ax, task_id in zip(axes_flat, sorted_tasks):
        series = series_for_task(task_agg, task_id)
        qcount = question_count_for_tasks(task_agg, (task_id,))
        subtitle = f"{qcount}Q" if qcount else None
        plot_f1_lines_on_ax(ax, series, title=f"Task {task_id}", subtitle=subtitle)
    for ax in axes_flat[len(sorted_tasks) :]:
        ax.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"{tier_label} per-task F1 vs context ({model_name})",
        y=1.04,
        fontsize=14,
    )
    save_plot(fig, out_dir, f"{model_slug(model_name)}_4_task_f1_vs_context")


def save_plot(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overall_f1_line(
    out_dir: Path, model_name: str, overall: dict[tuple[str, int], dict[str, Optional[float]]]
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    plot_f1_lines_on_ax(
        ax,
        overall,
        title=f"Tier4 overall F1 vs context ({model_name})",
        show_legend=True,
    )
    ax.set_xlabel("Context length")
    ax.set_ylabel("Question-weighted average F1")
    save_plot(fig, out_dir, f"{model_slug(model_name)}_1_overall_f1_vs_context")


def plot_overall_cost_bar(
    out_dir: Path, model_name: str, overall: dict[tuple[str, int], dict[str, Optional[float]]]
) -> None:
    context_values = sorted({ctx for (_, ctx) in overall.keys()}, key=context_sort_key)
    families = ("LLM", "CodeAct", "RLM")
    width = 0.24
    x = list(range(len(context_values)))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i, family in enumerate(families):
        ys = []
        for context in context_values:
            metric = overall.get((family, context), {}).get("cost")
            ys.append(float(metric) if metric is not None else math.nan)
        positions = [v + (i - 1) * width for v in x]
        ax.bar(positions, ys, width=width, label=family, color=FAMILY_COLORS[family], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([context_label(c) for c in context_values])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Question-weighted avg cost per sample (USD)")
    ax.set_title(f"Tier4 overall cost vs context ({model_name})")
    ax.legend()
    save_plot(fig, out_dir, f"{model_slug(model_name)}_3_overall_cost_vs_context")


def plot_overall_tokens_bar(
    out_dir: Path, model_name: str, overall: dict[tuple[str, int], dict[str, Optional[float]]]
) -> None:
    contexts = sorted({ctx for (_, ctx) in overall}, key=context_sort_key)
    families = ("LLM", "CodeAct", "RLM")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    metrics = [("input_tokens", "Input tokens"), ("output_tokens", "Output tokens")]
    width = 0.24
    x = list(range(len(contexts)))
    for ax, (metric_key, title) in zip(axes, metrics):
        for i, family in enumerate(families):
            ys = []
            for context in contexts:
                metric = overall.get((family, context), {}).get(metric_key)
                ys.append(float(metric) if metric is not None else math.nan)
            ax.bar(
                [v + (i - 1) * width for v in x],
                ys,
                width=width,
                label=family,
                color=FAMILY_COLORS[family],
                alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([context_label(c) for c in contexts])
        ax.set_xlabel("Context length")
        ax.set_ylabel(f"Question-weighted avg {title.lower()} / sample")
        ax.set_title(f"{title} ({model_name})")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    save_plot(fig, out_dir, f"{model_slug(model_name)}_5_overall_tokens_vs_context")


def write_summary_json(
    out_dir: Path,
    model_name: str,
    records: list[RunRecord],
    task_agg: dict[tuple[str, str, int], dict[str, float]],
    overall: dict[tuple[str, int], dict[str, Optional[float]]],
) -> None:
    payload = {
        "model_name": model_name,
        "records": [
            {
                "family": r.family,
                "task_id": r.task_id,
                "context_size": r.context_size,
                "question_count": r.question_count,
                "f1": r.f1,
                "cost_per_sample_usd": r.cost_per_sample_usd,
                "input_tokens_per_sample": r.input_tokens_per_sample,
                "output_tokens_per_sample": r.output_tokens_per_sample,
            }
            for r in records
        ],
        "task_aggregates": {
            f"{family}/task{task_id}/ctx={context}": stats
            for (family, task_id, context), stats in sorted(
                task_agg.items(),
                key=lambda item: (item[0][0], task_sort_key(item[0][1]), item[0][2]),
            )
        },
        "overall": {
            f"{family}/ctx={context_label(context)}": stats
            for (family, context), stats in sorted(
                overall.items(), key=lambda item: (item[0][0], context_sort_key(item[0][1]))
            )
        },
        "subcategories": {
            name: {
                f"{family}/ctx={context_label(context)}": stats
                for (family, context), stats in sorted(
                    aggregate_overall(task_agg, task_filter=frozenset(task_ids)).items(),
                    key=lambda item: (item[0][0], context_sort_key(item[0][1])),
                )
            }
            for name, task_ids in SUBGROUPS
        },
    }
    out_path = out_dir / f"{model_slug(model_name)}_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install it with: pip install matplotlib")
    if args.source == "api" and not args.entity:
        raise ValueError("Missing W&B entity. Pass --entity or set WANDB_ENTITY.")

    requested_models = set()
    if args.models.strip().lower() != "all":
        requested_models = {m.strip() for m in args.models.split(",") if m.strip()}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    allow_partial = args.allow_partial and not args.strict

    if args.source == "local":
        records = collect_records_from_local(
            local_wandb_dir=Path(args.local_wandb_dir),
            models_filter=requested_models,
            last_n_runs=args.last_n_runs,
            allow_partial=allow_partial,
        )
    else:
        tier4_dir = Path(args.tier4_dir)
        projects_by_task = discover_projects(tier4_dir, TASK_IDS)
        missing_tasks = [task_id for task_id in TASK_IDS if not projects_by_task.get(task_id)]
        if missing_tasks:
            print(f"Warning: no project names discovered for tasks: {', '.join(missing_tasks)}")
        records = collect_records(
            entity=args.entity,
            projects_by_task=projects_by_task,
            models_filter=requested_models,
            last_n_runs=args.last_n_runs,
            allow_partial=allow_partial,
        )
    if not records:
        raise RuntimeError("No matching completed runs found for the requested filters.")

    by_model: dict[str, list[RunRecord]] = defaultdict(list)
    for row in records:
        by_model[row.model_name].append(row)

    for model_name, model_rows in sorted(by_model.items()):
        task_agg = aggregate_task(model_rows)
        overall = aggregate_overall(task_agg)
        plot_overall_f1_line(out_dir, model_name, overall)
        plot_subcategory_f1_panels(
            out_dir,
            model_name,
            task_agg,
            SUBGROUPS,
            tier_label="Tier4",
            nrows=1,
            ncols=3,
        )
        plot_task_f1_panels(
            out_dir,
            model_name,
            task_agg,
            TASK_IDS,
            tier_label="Tier4",
            nrows=3,
            ncols=3,
        )
        plot_overall_cost_bar(out_dir, model_name, overall)
        plot_overall_tokens_bar(out_dir, model_name, overall)
        write_summary_json(out_dir, model_name, model_rows, task_agg, overall)
        print(
            f"[{model_name}] tasks={len({r.task_id for r in model_rows})} "
            f"records={len(model_rows)} contexts="
            f"{sorted({r.context_size for r in model_rows}, key=context_sort_key)}"
        )

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
