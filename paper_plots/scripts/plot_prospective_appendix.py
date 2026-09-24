#!/usr/bin/env python3
"""Plot aggregate, target-level, and efficiency views of the Task-16 control."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import HEATMAP_CMAP, apply_paper_style

CONDITIONS = ("name_only", "structure_only", "structure_plus_class")
CONDITION_LABELS = ("Target name", "Target structure", "Structure +\nfinal-step class")
MODELS = ("Qwen", "Claude")
MODEL_COLORS = {"Qwen": "#5302A3", "Claude": "#FCA636"}
MODEL_MARKERS = {"Qwen": "o", "Claude": "D"}
TARGETS = ("pyrimidine_piperazine", "lactam_dipeptide", "benzamide_pyrazole")
TARGET_LABELS = {
    "pyrimidine_piperazine": "Pyrimidine--piperazine",
    "lactam_dipeptide": "Lactam dipeptide",
    "benzamide_pyrazole": "Benzamide--pyrazole",
}

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


def aggregate_performance(rows: list[dict[str, str]]) -> plt.Figure:
    metrics = (
        ("mean_macro_f1", "Macro F1"),
        ("mean_precision", "Macro precision"),
        ("mean_recall", "Macro recall"),
        ("mean_exact_match_accuracy", "Exact-route accuracy"),
    )
    lookup = {(row["model_label"], row["condition"]): row for row in rows}
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 5.1), squeeze=False)
    x = np.arange(3)
    for axis, (metric, label) in zip(axes.ravel(), metrics, strict=True):
        for model in MODELS:
            values = [float(lookup[(model, condition)][metric]) for condition in CONDITIONS]
            errors = (
                [float(lookup[(model, condition)]["sd_macro_f1"]) for condition in CONDITIONS]
                if metric == "mean_macro_f1"
                else None
            )
            axis.errorbar(
                x,
                values,
                yerr=errors,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                markersize=5,
                capsize=2.5,
                label=model,
            )
        axis.set_xticks(x, CONDITION_LABELS)
        axis.set_ylabel(label)
        axis.set_ylim(-0.025, 0.62 if metric != "mean_exact_match_accuracy" else 0.42)
        axis.grid(axis="y", color="#D8DDE2", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    figure.subplots_adjust(left=0.10, right=0.99, top=0.98, bottom=0.11, hspace=0.35, wspace=0.28)
    return figure


def target_heatmaps(rows: list[dict[str, str]]) -> plt.Figure:
    figure, axes = plt.subplots(2, 1, figsize=(7.25, 3.9))
    for axis, model in zip(axes, MODELS, strict=True):
        array = np.asarray(
            [
                [
                    statistics.fmean(
                        float(row["f1"])
                        for row in rows
                        if row["model_label"] == model
                        and row["target"] == target
                        and row["condition"] == condition
                    )
                    for condition in CONDITIONS
                ]
                for target in TARGETS
            ]
        )
        image = axis.imshow(array, cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto")
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                value = array[i, j]
                axis.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < 0.58 else "#111111",
                )
        axis.set_yticks(range(3), [TARGET_LABELS[target] for target in TARGETS])
        axis.set_xticks(range(3), CONDITION_LABELS)
        axis.set_title(model, loc="left")
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    cbar_axis = figure.add_axes([0.91, 0.18, 0.018, 0.64])
    cbar = figure.colorbar(image, cax=cbar_axis)
    cbar.set_label("Macro F1")
    figure.subplots_adjust(left=0.24, right=0.88, top=0.95, bottom=0.08, hspace=0.43)
    return figure


def resource_page(rows: list[dict[str, str]]) -> plt.Figure:
    metrics = (
        ("calls", "Model calls / trajectory", "log"),
        ("total_tokens", "Total tokens / trajectory", "log"),
        ("cost_usd", "Recorded cost (USD) / trajectory", "symlog"),
        ("latency_seconds", "Model latency (s) / trajectory", "log"),
        ("tool_time_seconds", "Tool time (s) / trajectory", "symlog"),
        ("process_wall_time_seconds", "Process wall time (s) / trajectory", "log"),
        ("peak_combined_memory_mib", "Peak memory (MiB) / job", "log"),
    )
    figure, axes = plt.subplots(4, 2, figsize=(7.25, 8.7), squeeze=False)
    x = np.arange(3)
    for index, (metric, label, scale) in enumerate(metrics):
        axis = axes.ravel()[index]
        for model in MODELS:
            values = []
            errors = []
            for condition in CONDITIONS:
                group = [
                    row
                    for row in rows
                    if row["model_label"] == model and row["condition"] == condition
                ]
                per_run = [
                    float(row[metric])
                    if metric == "peak_combined_memory_mib"
                    else float(row[metric]) / int(row["question_count"])
                    for row in group
                    if row[metric] != ""
                ]
                values.append(statistics.fmean(per_run))
                errors.append(statistics.stdev(per_run) if len(per_run) > 1 else 0.0)
            axis.errorbar(
                x,
                values,
                yerr=errors,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                capsize=2,
                label=model,
            )
        axis.set_xticks(x, CONDITION_LABELS)
        axis.set_ylabel(label)
        if scale == "log":
            axis.set_yscale("log")
        else:
            axis.set_yscale("symlog", linthresh=1e-4)
        axis.grid(axis="y", which="both", color="#D8DDE2", linewidth=0.45)
        axis.spines[["top", "right"]].set_visible(False)
    axes.ravel()[0].legend(frameon=False)
    axes.ravel()[-1].axis("off")
    figure.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.055, hspace=0.48, wspace=0.35)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("paper_plots/gold/iclr2027/prospective_decomposition"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper_plots/figures/gold/prospective_decomposition")
    )
    args = parser.parse_args()
    aggregates = read_csv(args.data_root / "aggregates.csv")
    records = read_csv(args.data_root / "records.csv")
    targets = read_csv(args.data_root / "target_records.csv")
    save(aggregate_performance(aggregates), args.output, "task16_aggregate_metrics")
    save(target_heatmaps(targets), args.output, "task16_target_breakdown")
    save(resource_page(records), args.output, "task16_resource_diagnostics")
    print(f"Wrote Task-16 appendix figures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
