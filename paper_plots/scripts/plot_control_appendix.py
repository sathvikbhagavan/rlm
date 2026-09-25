#!/usr/bin/env python3
"""Plot detailed matched-cardinality, chemistry-rule, and executor controls."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import HEATMAP_CMAP, apply_paper_style

CONDITIONS = (
    "scale-x100-k1",
    "scale-x500-k1",
    "scale-x5000-k1",
    "scale-x50000-k1",
    "scale-xfull-k1",
    "cardinality-x5000-k5",
    "cardinality-x5000-k20",
)
CONDITION_LABELS = ("100/1", "500/1", "5k/1", "50k/1", "Full/1", "5k/5", "5k/20")
CONTEXTS = ("100", "500", "full")
CONTEXT_LABELS = ("100", "500", "Full")
TASK_LABELS = {
    "tier1/task1": "T1 Exact lookup",
    "tier2/task2": "T2 Weight change",
    "tier2/task3": "T2 Ring-count change",
    "tier2/task4": "T2 Aromatic rings",
    "tier2/task5": "T2 Weight + rings",
    "tier3/task6": "T3 Amide classes",
    "tier3/task7": "T3 Group transformations",
    "tier3/task8": "T3 Protecting groups",
    "tier3/task9": "T3 Named reactions",
    "tier3/task10": "T3 Mechanisms I",
    "tier3/task10b": "T3 Mechanisms II",
    "tier3/task13": "T3 New nitrogen",
    "tier3/task14": "T3 C--N / C--O",
    "tier3/task15": "T3 C--C formation",
    "tier3/task17": "T3 Fused heterocycle",
    "tier3/task18": "T3 New ring system",
    "tier3/task20": "T3 Fused-ring construction",
    "tier3/task21": "T3 Metal reagent",
    "tier3/task22": "T3 HATU / T3P",
    "tier3/task23": "T3 New stereocenter",
    "tier3/task24": "T3 E-alkene",
    "tier4/task13": "T4 Group-constrained paths",
    "tier4/task14": "T4 Protecting-group paths",
}
ORACLE_TASKS = ("tier3/task6", "tier3/task10", "tier3/task23", "tier4/task13", "tier4/task14")
ARM_COLORS = {"ordinary": "#5302A3", "predicate": "#CB4679"}
TIER_COLORS = {1: "#7E03A8", 2: "#5302A3", 3: "#CB4679"}
apply_paper_style()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(figure: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Creator": "RxnHaystack gold plotting pipeline",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03, metadata=metadata)
    figure.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def matched_task_heatmap(rows: list[dict[str, str]]) -> plt.Figure:
    rows = [row for row in rows if row["study"] == "matched_cardinality"]
    tasks = sorted({row["task"] for row in rows}, key=lambda task: list(TASK_LABELS).index(task))
    lookup: dict[tuple[str, str], float] = {}
    for task in tasks:
        for condition in CONDITIONS:
            values = [
                float(row["f1"])
                for row in rows
                if row["task"] == task and row["condition"] == condition
            ]
            lookup[(task, condition)] = mean(values)
    data = np.asarray([[lookup[(task, condition)] for condition in CONDITIONS] for task in tasks])
    figure, axis = plt.subplots(figsize=(7.25, 7.2))
    image = axis.imshow(data, cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            axis.text(
                j,
                i,
                "---" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="#666666" if np.isnan(value) else ("white" if value < 0.58 else "#111111"),
            )
    axis.set_xticks(range(len(CONDITIONS)), CONDITION_LABELS)
    axis.set_yticks(range(len(tasks)), [TASK_LABELS[task] for task in tasks])
    axis.xaxis.tick_top()
    axis.set_xlabel("Corpus size $N$ / positive reactions $K$")
    axis.xaxis.set_label_position("top")
    axis.axvline(4.5, color="white", linewidth=2.2)
    tiers = [int(task.removeprefix("tier").split("/", 1)[0]) for task in tasks]
    for i in range(1, len(tasks)):
        if tiers[i] != tiers[i - 1]:
            axis.axhline(i - 0.5, color="white", linewidth=2.2)
    axis.tick_params(length=0, labelsize=7)
    for spine in axis.spines.values():
        spine.set_visible(False)
    cbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.025)
    cbar.set_label("Macro F1")
    figure.subplots_adjust(left=0.30, right=0.94, top=0.91, bottom=0.02)
    return figure


def aggregate_condition_metric(
    rows: list[dict[str, str]], condition: str, tier: int, metric: str
) -> float:
    group = [row for row in rows if row["condition"] == condition and int(row["tier"]) == tier]
    if metric == "peak_combined_memory_mib":
        return mean([float(row[metric]) for row in group if row[metric] != ""])
    numerator = sum(float(row[metric]) for row in group if row[metric] != "")
    denominator = sum(int(row["question_count"]) for row in group if row[metric] != "")
    return numerator / denominator if denominator else float("nan")


def matched_resource_page(rows: list[dict[str, str]]) -> plt.Figure:
    rows = [row for row in rows if row["study"] == "matched_cardinality"]
    metrics = (
        ("calls", "Model calls / trajectory", "log"),
        ("input_tokens", "Input tokens / trajectory", "log"),
        ("output_tokens", "Output tokens / trajectory", "log"),
        ("cost_usd", "Recorded cost (USD) / trajectory", "symlog"),
        ("latency_seconds", "Model latency (s) / trajectory", "log"),
        ("tool_time_seconds", "Tool time (s) / trajectory", "symlog"),
        ("process_wall_time_seconds", "Process wall time (s) / trajectory", "log"),
        ("peak_combined_memory_mib", "Peak memory (MiB) / job", "log"),
    )
    figure, axes = plt.subplots(4, 2, figsize=(7.25, 9.0), squeeze=False)
    x = np.arange(len(CONDITIONS))
    for axis, (metric, label, scale) in zip(axes.ravel(), metrics, strict=True):
        for tier in (1, 2, 3):
            values = [
                aggregate_condition_metric(rows, condition, tier, metric)
                for condition in CONDITIONS
            ]
            axis.plot(
                x,
                values,
                color=TIER_COLORS[tier],
                marker=("o", "s", "D")[tier - 1],
                label=f"Tier {tier}",
            )
        axis.axvline(4.5, color="#BDBDBD", linewidth=0.8, linestyle="--")
        axis.set_xticks(x, CONDITION_LABELS, rotation=25, ha="right")
        axis.set_ylabel(label)
        if scale == "log":
            axis.set_yscale("log")
        else:
            axis.set_yscale("symlog", linthresh=1e-4)
            axis.set_ylim(bottom=0)
        axis.grid(axis="y", which="both", color="#D8DDE2", linewidth=0.45)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=3)
    figure.supxlabel("Corpus size $N$ / positive reactions $K$", y=0.01)
    figure.subplots_adjust(left=0.11, right=0.99, top=0.98, bottom=0.09, hspace=0.43, wspace=0.32)
    return figure


def qwen_snapshot(rows: list[dict[str, str]]) -> plt.Figure:
    """Plot the scientific result from the available Qwen matched runs.

    Execution coverage belongs in the experiment ledger and dashboard, not in
    the scientific figure.  Keeping it out also prevents transient job state
    from competing visually with the measured score.
    """
    figure, axis = plt.subplots(figsize=(7.25, 2.45))
    x = np.arange(len(CONDITIONS))
    for tier in (2, 3):
        values = []
        for condition in CONDITIONS:
            group = [
                row
                for row in rows
                if row["condition"] == condition
                and int(row["tier"]) == tier
                and row["status"] == "succeeded"
            ]
            numerator = sum(float(row["f1"]) * int(row["question_count"]) for row in group)
            denominator = sum(int(row["question_count"]) for row in group)
            values.append(numerator / denominator if denominator else np.nan)
        axis.plot(
            x,
            values,
            color=TIER_COLORS[tier],
            marker=("s" if tier == 2 else "D"),
            label=f"Tier {tier}",
        )
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Macro F1")
    axis.set_xticks(x, CONDITION_LABELS)
    axis.set_xlabel("Corpus size $N$ / positive reactions $K$")
    axis.axvline(4.5, color="#BDBDBD", linewidth=0.8, linestyle="--")
    axis.grid(axis="y", color="#D8DDE2", linewidth=0.45)
    axis.spines[["top", "right"]].set_visible(False)
    figure.legend(
        *axis.get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.subplots_adjust(left=0.08, right=0.995, top=0.83, bottom=0.23)
    return figure


def oracle_heatmaps(rows: list[dict[str, str]]) -> plt.Figure:
    oracle = [row for row in rows if row["study"] == "oracle_predicate"]
    columns = [
        (context, arm) for context in CONTEXTS for arm in ("ordinary", "predicate", "executor")
    ]
    labels = [
        f"{context.title()}\n{ {'ordinary': 'Ord.', 'predicate': 'Rule', 'executor': 'Exec.'}[arm] }"
        for context, arm in columns
    ]
    figure, axes = plt.subplots(2, 1, figsize=(7.25, 4.7), squeeze=False)
    for axis, model in zip(axes.ravel(), ("Qwen 3.5", "Claude Haiku 4.5"), strict=True):
        data = []
        for task in ORACLE_TASKS:
            values = []
            for context, arm in columns:
                model_label = "Deterministic executor" if arm == "executor" else model
                group = [
                    row
                    for row in oracle
                    if row["task"] == task
                    and row["context"] == context
                    and row["arm"] == arm
                    and row["model_label"] == model_label
                ]
                values.append(mean([float(row["f1"]) for row in group]))
            data.append(values)
        array = np.asarray(data)
        image = axis.imshow(array, cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto")
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                axis.text(
                    j,
                    i,
                    f"{array[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="white" if array[i, j] < 0.58 else "#111111",
                )
        axis.set_yticks(range(len(ORACLE_TASKS)), [TASK_LABELS[task] for task in ORACLE_TASKS])
        axis.set_xticks(range(len(columns)), labels)
        axis.set_title(model, loc="left")
        axis.tick_params(length=0, labelsize=7)
        axis.axvline(2.5, color="white", linewidth=2)
        axis.axvline(5.5, color="white", linewidth=2)
        for spine in axis.spines.values():
            spine.set_visible(False)
    cbar_axis = figure.add_axes([0.92, 0.18, 0.018, 0.64])
    cbar = figure.colorbar(image, cax=cbar_axis)
    cbar.set_label("Macro F1")
    figure.subplots_adjust(left=0.29, right=0.89, top=0.96, bottom=0.05, hspace=0.36)
    return figure


def oracle_model_page(rows: list[dict[str, str]], model: str) -> plt.Figure:
    rows = [
        row
        for row in rows
        if row["study"] == "oracle_predicate"
        and row["model_label"] == model
        and row["arm"] in {"ordinary", "predicate"}
    ]
    metrics = (
        ("f1", "Macro F1", "linear", True),
        ("calls", "Model calls / trajectory", "log", False),
        ("input_tokens", "Input tokens / trajectory", "log", False),
        ("output_tokens", "Output tokens / trajectory", "log", False),
        ("cost_usd", "Recorded cost (USD) / trajectory", "symlog", False),
        ("latency_seconds", "Model latency (s) / trajectory", "log", False),
        ("tool_time_seconds", "Tool time (s) / trajectory", "symlog", False),
        ("process_wall_time_seconds", "Process wall time (s) / trajectory", "log", False),
        ("peak_combined_memory_mib", "Peak memory (MiB) / job", "log", False),
    )
    figure, axes = plt.subplots(5, 2, figsize=(7.25, 9.8), squeeze=False)
    x = np.arange(3)
    for axis, (metric, label, scale, is_score) in zip(
        axes.ravel()[: len(metrics)], metrics, strict=True
    ):
        for arm in ("ordinary", "predicate"):
            values = []
            for context in CONTEXTS:
                group = [row for row in rows if row["context"] == context and row["arm"] == arm]
                if is_score:
                    numerator = sum(
                        float(row[metric]) * int(row["question_count"]) for row in group
                    )
                    denominator = sum(int(row["question_count"]) for row in group)
                    values.append(numerator / denominator)
                elif metric == "peak_combined_memory_mib":
                    values.append(mean([float(row[metric]) for row in group if row[metric] != ""]))
                else:
                    measured = [row for row in group if row[metric] != ""]
                    values.append(
                        sum(float(row[metric]) for row in measured)
                        / sum(int(row["question_count"]) for row in measured)
                    )
            axis.plot(
                x,
                values,
                color=ARM_COLORS[arm],
                marker=("o" if arm == "ordinary" else "D"),
                label=("Ordinary RLM" if arm == "ordinary" else "Chemistry rule supplied"),
            )
        axis.set_xticks(x, CONTEXT_LABELS)
        axis.set_ylabel(label)
        if scale == "log":
            axis.set_yscale("log")
        elif scale == "symlog":
            axis.set_yscale("symlog", linthresh=1e-4)
            axis.set_ylim(bottom=0)
        else:
            axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", which="both", color="#D8DDE2", linewidth=0.45)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1, -1].axis("off")
    axes[0, 0].legend(frameon=False)
    figure.supxlabel("Context size (reactions)", y=0.01)
    figure.subplots_adjust(left=0.105, right=0.995, top=0.99, bottom=0.06, hspace=0.43, wspace=0.34)
    return figure


def executor_heatmaps(rows: list[dict[str, str]]) -> plt.Figure:
    rows = [row for row in rows if row["study"] == "oracle_predicate" and row["arm"] == "executor"]
    metrics = (
        ("process_wall_time_seconds", "Runtime (s)"),
        ("peak_combined_memory_mib", "Peak memory (MiB)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.1))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        lookup = {(row["task"], row["context"]): float(row[metric]) for row in rows}
        array = np.asarray(
            [[lookup[(task, context)] for context in CONTEXTS] for task in ORACLE_TASKS]
        )
        image = axis.imshow(array, cmap=HEATMAP_CMAP, aspect="auto")
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                axis.text(
                    j,
                    i,
                    f"{array[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if array[i, j] < np.nanmedian(array) else "#111111",
                )
        axis.set_xticks(range(3), CONTEXT_LABELS)
        axis.set_yticks(
            range(5), [TASK_LABELS[task] for task in ORACLE_TASKS] if axis is axes[0] else []
        )
        axis.set_title(title)
        axis.tick_params(length=0, labelsize=7)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    figure.subplots_adjust(left=0.25, right=0.98, top=0.91, bottom=0.13, wspace=0.32)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("paper_plots/gold/iclr2027/causal_controls")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper_plots/figures/gold/control_appendix")
    )
    args = parser.parse_args()
    records = read_csv(args.data_root / "records.csv")
    qwen = read_csv(args.data_root / "matched_qwen_provisional.csv")
    save(matched_task_heatmap(records), args.output, "matched_gpt_task_heatmap")
    save(matched_resource_page(records), args.output, "matched_gpt_resources")
    save(qwen_snapshot(qwen), args.output, "matched_qwen_snapshot")
    save(oracle_heatmaps(records), args.output, "oracle_task_context_heatmaps")
    save(oracle_model_page(records, "Qwen 3.5"), args.output, "oracle_qwen_profile")
    save(oracle_model_page(records, "Claude Haiku 4.5"), args.output, "oracle_claude_profile")
    save(executor_heatmaps(records), args.output, "oracle_executor_runtime")
    print(f"Wrote detailed control figures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
