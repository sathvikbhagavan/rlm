"""Plot question-weighted overall F1 vs context across tiers 1–4 (100 questions).

Task lists and per-task question weights mirror each tier's plot_wandb_last_runs.py.
Tier 3 skips tasks 11, 12, 16, 19. Tier 4 RLM @ full context uses the last 3 runs.
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

try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    plt = None

FAMILY_COLORS = {"LLM": "#1f77b4", "CodeAct": "#d62728", "RLM": "#2ca02c"}
TASK_SCRIPT_PATTERN = re.compile(r"(?:rlm|llm|codeact)_task(\d+b?)\.py$")
CONFIG_FIELD_PATTERN = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)

TIER1_TASK_IDS = ("1",)
TIER2_TASK_IDS = ("2", "3", "4", "5")
TIER3_TASK_IDS = (
    "6",
    "7",
    "8",
    "9",
    "10",
    "10b",
    "13",
    "14",
    "15",
    "17",
    "18",
    "20",
    "21",
    "22",
    "23",
    "24",
)
TIER4_TASK_IDS = ("11", "12", "12b", "13", "14", "15", "16", "17", "17b")

TIER1_QUESTION_COUNTS: dict[str, int] = {"1": 10}
TIER2_QUESTION_COUNTS: dict[str, int] = {"2": 6, "3": 5, "4": 5, "5": 4}
# Matches evaluated reaction keys in tier3 scripts (skips excluded at runtime).
TIER3_QUESTION_COUNTS: dict[str, int] = {
    "6": 4,
    "7": 5,
    "8": 2,
    "9": 4,
    "10": 5,
    "10b": 5,
    "13": 1,
    "14": 1,
    "15": 1,
    "17": 1,
    "18": 1,
    "20": 1,
    "21": 1,
    "22": 1,
    "23": 1,
    "24": 1,
}
# Mirrors tier4/plot_wandb_last_runs.py; subgroup totals 5 + 10 + 20 = 35.
TIER4_QUESTION_COUNTS: dict[str, int] = {
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

TIER_SPECS: tuple[tuple[str, Path, tuple[str, ...], dict[str, int]], ...] = (
    ("tier1", Path("/home/bhagavan/rlms/rlm/tier1/wandb"), TIER1_TASK_IDS, TIER1_QUESTION_COUNTS),
    ("tier2", Path("/home/bhagavan/rlms/rlm/tier2/wandb"), TIER2_TASK_IDS, TIER2_QUESTION_COUNTS),
    ("tier3", Path("/home/bhagavan/rlms/rlm/tier3/wandb"), TIER3_TASK_IDS, TIER3_QUESTION_COUNTS),
    ("tier4", Path("/home/bhagavan/rlms/rlm/tier4/wandb"), TIER4_TASK_IDS, TIER4_QUESTION_COUNTS),
)

MIN_RUNS_OVERRIDES: dict[tuple[str, str, int], int] = {
    ("RLM", "16", -1): 3,
}


@dataclass
class RunRecord:
    tier: str
    family: str
    task_id: str
    model_name: str
    context_size: int
    question_count: float
    f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all-tiers overall F1 vs context from local W&B runs."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="openai/gpt-5-mini",
        help="Comma-separated model names, or 'all'.",
    )
    parser.add_argument("--last-n-runs", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/plots_all_tiers",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        default=True,
        help="Plot with available runs instead of requiring last-n-runs per group.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def task_sort_key(task_id: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)(.*)$", task_id)
    if not match:
        return (10**9, task_id)
    return (int(match.group(1)), match.group(2))


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


def context_allowed(family: str, context_size: int) -> bool:
    if family in {"LLM", "CodeAct"}:
        return context_size in {100, 500}
    if family == "RLM":
        return context_size in {100, 500, -1}
    return False


def context_sort_key(context_size: int) -> int:
    return 10_000 if context_size == -1 else context_size


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
                config[key] = float(raw) if "." in raw else int(raw)
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
                config[key] = float(raw) if "." in raw else int(raw)
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
    return match.group(1) if match else None


def question_count_from_run(
    config: dict, summary: dict, task_id: str, fallbacks: dict[str, int]
) -> Optional[float]:
    for source in (summary, config):
        value = safe_float(source.get("total"))
        if value is not None and value > 0:
            return value
    value = safe_float(config.get("num_questions"))
    if value is not None and value > 0:
        return value
    fallback = fallbacks.get(task_id)
    if fallback is not None and fallback > 0:
        return float(fallback)
    return None


def required_runs_for_group(
    family: str, task_id: str, context_size: int, *, default: int
) -> int:
    if (family, task_id, context_size) in MIN_RUNS_OVERRIDES:
        return MIN_RUNS_OVERRIDES[(family, task_id, context_size)]
    return default


def collect_records_from_local(
    *,
    tier: str,
    local_wandb_dir: Path,
    allowed_tasks: frozenset[str],
    fallbacks: dict[str, int],
    models_filter: set[str],
    last_n_runs: int,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    seen_counts: dict[tuple[str, str, str, str, int], int] = defaultdict(int)
    run_dirs = sorted(
        [path for path in local_wandb_dir.glob("run-*") if path.is_dir()],
        key=lambda path: path.name,
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
        if family is None or task_id is None or task_id not in allowed_tasks:
            continue

        model_name = normalize_model_name(config)
        if models_filter and model_name not in models_filter:
            continue

        context_size = normalize_context_size(config)
        if context_size is None or not context_allowed(family, context_size):
            continue

        question_count = question_count_from_run(config, summary, task_id, fallbacks)
        f1 = safe_float(summary.get("macro_f1"))
        if question_count is None or f1 is None:
            continue

        key = (tier, family, task_id, model_name, context_size)
        required = required_runs_for_group(
            family, task_id, context_size, default=last_n_runs
        )
        if seen_counts[key] >= required:
            continue
        seen_counts[key] += 1
        records.append(
            RunRecord(
                tier=tier,
                family=family,
                task_id=task_id,
                model_name=model_name,
                context_size=context_size,
                question_count=question_count,
                f1=f1,
            )
        )
    return records


def aggregate_task(
    records: list[RunRecord],
) -> dict[tuple[str, str, str, int], dict[str, float]]:
    grouped: dict[tuple[str, str, str, int], list[RunRecord]] = defaultdict(list)
    for row in records:
        grouped[(row.family, row.tier, row.task_id, row.context_size)].append(row)

    out: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for key, rows in grouped.items():
        f1_values = [r.f1 for r in rows]
        out[key] = {
            "question_count": mean([r.question_count for r in rows]),
            "run_count": len(rows),
            "f1": mean(f1_values),
            "f1_std": pstdev(f1_values) if len(f1_values) > 1 else 0.0,
        }
    return out


def aggregate_overall(
    task_agg: dict[tuple[str, str, str, int], dict[str, float]],
) -> dict[tuple[str, int], dict[str, Optional[float]]]:
    grouped: dict[tuple[str, int], list[tuple[str, str, dict[str, float]]]] = defaultdict(list)
    for (family, tier, task_id, context), stats in task_agg.items():
        grouped[(family, context)].append((tier, task_id, stats))

    out: dict[tuple[str, int], dict[str, Optional[float]]] = {}
    for key, rows in grouped.items():
        f1_pairs = [(stats["f1"], stats["question_count"]) for _, _, stats in rows]
        total_weight = sum(stats["question_count"] for _, _, stats in rows)
        f1_var = 0.0
        if total_weight > 0:
            for _, _, stats in rows:
                w_norm = stats["question_count"] / total_weight
                f1_var += (w_norm**2) * (float(stats.get("f1_std", 0.0)) ** 2)
        out[key] = {
            "f1": weighted_mean(f1_pairs),
            "f1_std": math.sqrt(f1_var) if total_weight > 0 else None,
            "task_count": len(rows),
            "question_count": total_weight,
        }
    return out


def save_plot(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overall_f1_line(
    out_dir: Path,
    model_name: str,
    overall: dict[tuple[str, int], dict[str, Optional[float]]],
    *,
    benchmark_questions: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for family in ("LLM", "CodeAct", "RLM"):
        points = sorted(
            [
                (context, stats["f1"], stats.get("f1_std"))
                for (fam, context), stats in overall.items()
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
            linewidth=2,
            capsize=3,
            label=family,
            color=FAMILY_COLORS[family],
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([i / 10 for i in range(0, 11)])
    ax.grid(which="major", axis="both", alpha=0.35)
    ax.minorticks_on()
    ax.grid(which="minor", axis="y", alpha=0.18, linestyle=":")
    ax.set_xlabel("Context length")
    ax.set_ylabel("Question-weighted average F1")
    ax.set_title(
        f"All tiers overall F1 vs context ({model_name}, {benchmark_questions}Q)"
    )
    ax.legend()
    save_plot(fig, out_dir, f"{model_slug(model_name)}_all_tiers_overall_f1_vs_context")


def write_summary_json(
    out_dir: Path,
    model_name: str,
    records: list[RunRecord],
    task_agg: dict[tuple[str, str, str, int], dict[str, float]],
    overall: dict[tuple[str, int], dict[str, Optional[float]]],
    *,
    benchmark_questions: int,
) -> None:
    payload = {
        "model_name": model_name,
        "benchmark_questions": benchmark_questions,
        "records": [
            {
                "tier": r.tier,
                "family": r.family,
                "task_id": r.task_id,
                "context_size": r.context_size,
                "question_count": r.question_count,
                "f1": r.f1,
            }
            for r in records
        ],
        "task_aggregates": {
            f"{family}/{tier}/task{task_id}/ctx={context_label(context)}": stats
            for (family, tier, task_id, context), stats in sorted(
                task_agg.items(),
                key=lambda item: (item[0][0], item[0][1], task_sort_key(item[0][2]), item[0][3]),
            )
        },
        "overall": {
            f"{family}/ctx={context_label(context)}": stats
            for (family, context), stats in sorted(
                overall.items(), key=lambda item: (item[0][0], context_sort_key(item[0][1]))
            )
        },
    }
    out_path = out_dir / f"{model_slug(model_name)}_all_tiers_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def benchmark_question_total() -> int:
    return sum(
        sum(counts.values())
        for _, _, _, counts in TIER_SPECS
    )


def main() -> None:
    args = parse_args()
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install it with: pip install matplotlib")

    requested_models = set()
    if args.models.strip().lower() != "all":
        requested_models = {m.strip() for m in args.models.split(",") if m.strip()}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    allow_partial = args.allow_partial and not args.strict
    benchmark_questions = benchmark_question_total()

    records: list[RunRecord] = []
    for tier, wandb_dir, task_ids, fallbacks in TIER_SPECS:
        if not wandb_dir.exists():
            print(f"Warning: missing wandb dir for {tier}: {wandb_dir}")
            continue
        tier_records = collect_records_from_local(
            tier=tier,
            local_wandb_dir=wandb_dir,
            allowed_tasks=frozenset(task_ids),
            fallbacks=fallbacks,
            models_filter=requested_models,
            last_n_runs=args.last_n_runs,
        )
        records.extend(tier_records)
        print(
            f"[{tier}] tasks={len({r.task_id for r in tier_records})} "
            f"records={len(tier_records)}"
        )

    if not records:
        raise RuntimeError("No matching completed runs found for the requested filters.")

    by_model: dict[str, list[RunRecord]] = defaultdict(list)
    for row in records:
        by_model[row.model_name].append(row)

    for model_name, model_rows in sorted(by_model.items()):
        task_agg = aggregate_task(model_rows)
        overall = aggregate_overall(task_agg)
        plot_overall_f1_line(
            out_dir,
            model_name,
            overall,
            benchmark_questions=benchmark_questions,
        )
        write_summary_json(
            out_dir,
            model_name,
            model_rows,
            task_agg,
            overall,
            benchmark_questions=benchmark_questions,
        )
        contexts = sorted({r.context_size for r in model_rows}, key=context_sort_key)
        tasks = len({(r.tier, r.task_id) for r in model_rows})
        print(
            f"[{model_name}] tasks={tasks} records={len(model_rows)} "
            f"contexts={contexts} benchmark_questions={benchmark_questions}"
        )

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
