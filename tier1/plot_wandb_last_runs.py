import argparse
import math
import os
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


PROJECT_BY_FAMILY = {
    "LLM": "LLM-Task1",
    "CodeAct": "CodeAct-Task1",
    "RLM": "RLMs-Task1",
}
FAMILY_COLORS = {"LLM": "#1f77b4", "CodeAct": "#d62728", "RLM": "#2ca02c"}


@dataclass
class RunRecord:
    family: str
    model_name: str
    context_size: int
    question_count: float
    f1: float
    cost_per_sample_usd: Optional[float]
    input_tokens_per_sample: Optional[float]
    output_tokens_per_sample: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot tier1 F1/cost/token trends vs context size from latest W&B runs."
    )
    parser.add_argument("--entity", type=str, default=os.getenv("WANDB_ENTITY", ""))
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
        help="Take last N runs for each family/model/context.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/tier1/plots_last5",
    )
    return parser.parse_args()


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
    if context_size == -1:
        return 10_000
    return context_size


def context_label(context_size: int) -> str:
    return "full" if context_size == -1 else str(context_size)


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def mean_or_nan(values: list[float]) -> float:
    return mean(values) if values else float("nan")


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


def extract_avg_tokens_from_summary(summary: dict, token_kind: str) -> Optional[float]:
    direct_key = f"avg_total_{token_kind}_tokens_per_sample"
    direct = safe_float(summary.get(direct_key))
    if direct is not None:
        return direct

    values: list[float] = []
    suffix = f"/final_total_{token_kind}_tokens"
    for key, value in summary.items():
        if key.startswith("sample/") and key.endswith(suffix):
            parsed = safe_float(value)
            if parsed is not None:
                values.append(parsed)
    if values:
        return mean(values)
    return None


def collect_records(entity: str, models_filter: set[str], last_n_runs: int) -> list[RunRecord]:
    api = wandb.Api()
    records: list[RunRecord] = []
    seen_counts: dict[tuple[str, str, int], int] = defaultdict(int)

    for family, project in PROJECT_BY_FAMILY.items():
        runs = api.runs(f"{entity}/{project}", order="-created_at")
        for run in runs:
            config = dict(run.config or {})
            summary = dict(run.summary or {})
            model_name = normalize_model_name(config)
            if models_filter and model_name not in models_filter:
                continue

            context_size = normalize_context_size(config)
            if context_size is None or not context_allowed(family, context_size):
                continue

            key = (family, model_name, context_size)
            if seen_counts[key] >= last_n_runs:
                continue

            question_count = safe_float(summary.get("total"))
            f1 = safe_float(summary.get("macro_f1"))
            if question_count is None or f1 is None:
                continue

            seen_counts[key] += 1

            avg_cost_per_sample = safe_float(summary.get("avg_cost_per_sample_usd"))
            if avg_cost_per_sample is None:
                total_cost = safe_float(summary.get("total_cost_usd"))
                avg_cost_per_sample = (
                    total_cost / question_count
                    if total_cost is not None and question_count > 0
                    else None
                )

            records.append(
                RunRecord(
                    family=family,
                    model_name=model_name,
                    context_size=context_size,
                    question_count=question_count,
                    f1=f1,
                    cost_per_sample_usd=avg_cost_per_sample,
                    input_tokens_per_sample=extract_avg_tokens_from_summary(summary, "input"),
                    output_tokens_per_sample=extract_avg_tokens_from_summary(summary, "output"),
                )
            )

    insufficient_groups = [key for key, count in seen_counts.items() if count < last_n_runs]
    if insufficient_groups:
        formatted = ", ".join(
            [
                (
                    f"{family}/model={model}/ctx={context} "
                    f"(found {seen_counts[(family, model, context)]}, need {last_n_runs})"
                )
                for (family, model, context) in sorted(insufficient_groups)
            ]
        )
        raise RuntimeError(
            "Not enough runs for strict averaging across last "
            f"{last_n_runs} runs. Missing groups: {formatted}"
        )
    return records


def aggregate_runs(records: list[RunRecord]) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], list[RunRecord]] = defaultdict(list)
    for row in records:
        grouped[(row.family, row.context_size)].append(row)

    out: dict[tuple[str, int], dict[str, float]] = {}
    for key, rows in grouped.items():
        f1_values = [r.f1 for r in rows]
        out[key] = {
            "question_count": mean([r.question_count for r in rows]),
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


def save_plot(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_f1_line(
    out_dir: Path, model_name: str, agg: dict[tuple[str, int], dict[str, float]], stem: str, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for family in ("LLM", "CodeAct", "RLM"):
        points = sorted(
            [
                (context, stats["f1"], stats.get("f1_std", 0.0))
                for (fam, context), stats in agg.items()
                if fam == family
            ],
            key=lambda x: context_sort_key(x[0]),
        )
        if not points:
            continue
        xs = [context_label(c) for c, _, _ in points]
        ys = [float(v) for _, v, _ in points]
        ystd = [float(e) for _, _, e in points]
        ax.errorbar(
            xs,
            ys,
            yerr=bounded_f1_yerr(ys, ystd),
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
    ax.set_ylabel("Average F1")
    ax.set_title(f"{title} ({model_name})")
    ax.legend()
    save_plot(fig, out_dir, f"{model_slug(model_name)}_{stem}")


def plot_cost_bar(
    out_dir: Path, model_name: str, agg: dict[tuple[str, int], dict[str, float]], stem: str, title: str
) -> None:
    contexts = sorted({ctx for (_, ctx) in agg.keys()}, key=context_sort_key)
    families = ("LLM", "CodeAct", "RLM")
    width = 0.24
    x = list(range(len(contexts)))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i, family in enumerate(families):
        ys = []
        for context in contexts:
            metric = agg.get((family, context), {}).get("cost")
            ys.append(float(metric) if metric is not None and not math.isnan(metric) else math.nan)
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
    ax.set_ylabel("Average cost per sample (USD)")
    ax.set_title(f"{title} ({model_name})")
    ax.legend()
    save_plot(fig, out_dir, f"{model_slug(model_name)}_{stem}")


def plot_tokens_bar(
    out_dir: Path, model_name: str, agg: dict[tuple[str, int], dict[str, float]], stem: str, title: str
) -> None:
    contexts = sorted({ctx for (_, ctx) in agg.keys()}, key=context_sort_key)
    families = ("LLM", "CodeAct", "RLM")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=False)
    metrics = [("input_tokens", "Input tokens"), ("output_tokens", "Output tokens")]
    width = 0.24
    x = list(range(len(contexts)))
    for ax, (metric_key, metric_title) in zip(axes, metrics):
        for i, family in enumerate(families):
            ys = []
            for context in contexts:
                metric = agg.get((family, context), {}).get(metric_key)
                ys.append(float(metric) if metric is not None and not math.isnan(metric) else math.nan)
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
        ax.set_ylabel(f"Average {metric_title.lower()} / sample")
        ax.set_title(metric_title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            title="Family",
            borderaxespad=0.0,
        )
    fig.suptitle(f"{title} ({model_name})", y=0.97)
    # Reserve fixed space for legend and titles so it never overlaps axes labels.
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.23, wspace=0.16)
    save_plot(fig, out_dir, f"{model_slug(model_name)}_{stem}")


def main() -> None:
    args = parse_args()
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install it with: pip install matplotlib")
    if not args.entity:
        raise ValueError("Missing W&B entity. Pass --entity or set WANDB_ENTITY.")

    requested_models = set()
    if args.models.strip().lower() != "all":
        requested_models = {m.strip() for m in args.models.split(",") if m.strip()}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    records = collect_records(args.entity, requested_models, args.last_n_runs)
    if not records:
        raise RuntimeError("No matching runs found for the requested filters.")

    by_model: dict[str, list[RunRecord]] = defaultdict(list)
    for row in records:
        by_model[row.model_name].append(row)

    for model_name, model_rows in sorted(by_model.items()):
        agg = aggregate_runs(model_rows)
        # 1) Overall average F1 vs context
        plot_f1_line(
            out_dir,
            model_name,
            agg,
            stem="1_overall_f1_vs_context",
            title="Average F1 vs context",
        )
        # 2) Overall average cost vs context
        plot_cost_bar(
            out_dir,
            model_name,
            agg,
            stem="2_overall_cost_vs_context",
            title="Average cost vs context",
        )
        # 3) Overall average input/output tokens vs context
        plot_tokens_bar(
            out_dir,
            model_name,
            agg,
            stem="3_overall_tokens_vs_context",
            title="Average input/output tokens vs context",
        )

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
