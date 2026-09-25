#!/usr/bin/env python3
"""Plot a complete per-model performance and efficiency appendix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_style import (
    HEATMAP_CMAP,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    apply_paper_style,
)

METHODS = ("llm", "codeact", "rlm")
CONTEXT_ORDER = {"100": 0, "500": 1, "1000": 2, "full": 3}
CONTEXT_LABELS = {"100": "100", "500": "500", "1000": "1000", "full": "Full"}
TIER_NAMES = {
    1: "Structural lookup",
    2: "Property aggregation",
    3: "Reaction-level reasoning",
    4: "Relational reasoning",
}
CORE_METRICS = (
    ("f1", "Macro F1", "linear"),
    ("total_tokens", "Total tokens / trajectory", "log"),
    ("cost_usd", "Recorded cost (USD) / trajectory", "symlog"),
    ("process_wall_time_seconds", "Process wall time (s) / trajectory", "log"),
)
DIAGNOSTIC_METRICS = (
    ("calls", "Model calls / trajectory", "log"),
    ("latency_seconds", "Recorded model latency (s) / trajectory", "log"),
    ("tool_time_seconds", "Tool execution time (s) / trajectory", "symlog"),
    ("peak_combined_memory_mib", "Peak memory (MiB) / job", "log"),
)
COLUMN_ORDER = (
    ("llm", "100"),
    ("codeact", "100"),
    ("rlm", "100"),
    ("llm", "500"),
    ("codeact", "500"),
    ("rlm", "500"),
    ("codeact", "1000"),
    ("rlm", "1000"),
    ("rlm", "full"),
)

apply_paper_style()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(figure: plt.Figure, output: Path, name: str) -> list[str]:
    metadata = {
        "Creator": "RxnHaystack gold plotting pipeline",
        "CreationDate": None,
        "ModDate": None,
    }
    pdf = output / f"{name}.pdf"
    png = output / f"{name}.png"
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.02, metadata=metadata)
    figure.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return [pdf.name, png.name]


def metric_page(
    rows: list[dict[str, str]],
    metrics: tuple[tuple[str, str, str], ...],
) -> plt.Figure:
    """Make four 1x4 tier rows on one appendix page."""
    contexts = tuple(
        sorted({row["context"] for row in rows}, key=CONTEXT_ORDER.__getitem__)
    )
    context_positions = {context: index for index, context in enumerate(contexts)}
    has_positive = {
        metric: any(float(row[f"{metric}_mean"] or 0.0) > 0 for row in rows)
        for metric, _, _ in metrics
    }
    figure, axes = plt.subplots(4, 4, figsize=(7.25, 9.0), squeeze=False)
    for column, tier in enumerate(range(1, 5)):
        axes[0, column].set_title(
            f"Tier {tier}\n{TIER_NAMES[tier]}", fontsize=9, fontweight="normal", pad=6
        )

    for row_index, (metric, label, scale) in enumerate(metrics):
        for column, tier in enumerate(range(1, 5)):
            axis = axes[row_index, column]
            if "full" in context_positions:
                full_position = context_positions["full"]
                axis.axvspan(
                    full_position - 0.28, full_position + 0.28, color="#F2F2F2", zorder=0
                )
            for method in METHODS:
                subset = sorted(
                    (
                        row
                        for row in rows
                        if int(row["tier"]) == tier
                        and row["method"] == method
                        and row[f"{metric}_mean"] != ""
                    ),
                    key=lambda row: context_positions[row["context"]],
                )
                if not subset:
                    continue
                x_values = [context_positions[row["context"]] for row in subset]
                means = [float(row[f"{metric}_mean"]) for row in subset]
                errors = [float(row[f"{metric}_std"]) for row in subset]
                lower_errors = [
                    min(error, mean * 0.95) if mean > 0 else 0.0
                    for mean, error in zip(means, errors, strict=True)
                ]
                axis.errorbar(
                    x_values,
                    means,
                    yerr=np.asarray([lower_errors, errors]),
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    markersize=4.0,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    capsize=2.0,
                    elinewidth=0.75,
                    zorder=3,
                )
            if scale == "log" and has_positive[metric]:
                axis.set_yscale("log")
            elif scale == "log":
                axis.set_ylim(-0.05, 1.05)
                axis.set_yticks((0, 0.5, 1.0))
            elif scale == "symlog":
                axis.set_yscale("symlog", linthresh=1e-3)
            else:
                axis.set_ylim(-0.03, 1.05)
                axis.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
            axis.set_xlim(-0.18, len(contexts) - 0.82)
            axis.set_xticks(
                range(len(contexts)), [CONTEXT_LABELS[context] for context in contexts]
            )
            axis.grid(axis="y", which="both", color="#D8DDE2", linewidth=0.45)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(length=2.2, width=0.55, labelsize=6.7)
            if column == 0:
                axis.set_ylabel(label, fontsize=7.2)
            if row_index == len(metrics) - 1:
                axis.set_xlabel("Context size", fontsize=7.2)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=4.5,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.subplots_adjust(
        left=0.105, right=0.995, top=0.925, bottom=0.055, hspace=0.42, wspace=0.36
    )
    return figure


def task_heatmap(rows: list[dict[str, str]]) -> plt.Figure:
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    labels = {row["task"]: row["task_label"] for row in rows}
    lookup = {
        (row["task"], row["method"], row["context"]): float(row["f1_mean"])
        for row in rows
        if row["f1_mean"] != ""
    }
    available_keys = {(row["method"], row["context"]) for row in rows}
    column_keys = tuple(key for key in COLUMN_ORDER if key in available_keys)
    column_labels = tuple(
        f"{METHOD_LABELS[method]}\n{CONTEXT_LABELS[context]}"
        for method, context in column_keys
    )
    values = np.asarray(
        [
            [lookup.get((task, method, context), np.nan) for method, context in column_keys]
            for task in tasks
        ]
    )
    figure, axis = plt.subplots(figsize=(7.25, 9.5))
    image = axis.imshow(values, cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto")
    for row_index in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row_index, column]
            if np.isnan(value):
                continue
            axis.text(
                column,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.1,
                color="white" if value < 0.58 else "#111111",
            )
    axis.set_yticks(range(len(tasks)), [labels[task] for task in tasks])
    axis.set_xticks(range(len(column_keys)), column_labels)
    axis.xaxis.tick_top()
    axis.tick_params(axis="both", length=0, labelsize=6.8)
    for index in range(1, len(column_keys)):
        if column_keys[index][1] != column_keys[index - 1][1]:
            axis.axvline(index - 0.5, color="white", linewidth=2)
    tiers = [int(next(row["tier"] for row in rows if row["task"] == task)) for task in tasks]
    for index in range(1, len(tasks)):
        if tiers[index] != tiers[index - 1]:
            axis.axhline(index - 0.5, color="white", linewidth=2.5)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.subplots_adjust(left=0.43, right=0.91, top=0.94, bottom=0.02)
    colorbar_axis = figure.add_axes([0.93, 0.13, 0.016, 0.68])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Macro F1", rotation=90, labelpad=6)
    colorbar.set_ticks((0, 0.25, 0.5, 0.75, 1.0))
    return figure


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("paper_plots/gold/iclr2027/model_appendix")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper_plots/figures/gold/model_appendix")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.data_root / args.model
    tier_path = source / "tier_metrics.csv"
    task_path = source / "task_f1.csv"
    tier_rows = read_csv(tier_path)
    task_rows = read_csv(task_path)
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.model.replace(".", "_")
    figure_files: list[str] = []
    figure_files.extend(
        save_figure(
            metric_page(tier_rows, CORE_METRICS),
            args.output,
            f"{prefix}_core_metrics",
        )
    )
    figure_files.extend(
        save_figure(
            metric_page(tier_rows, DIAGNOSTIC_METRICS),
            args.output,
            f"{prefix}_resource_diagnostics",
        )
    )
    figure_files.extend(
        save_figure(
            task_heatmap(task_rows),
            args.output,
            f"{prefix}_task_heatmap",
        )
    )
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "model_label": args.label,
        "palette": "Plasma",
        "sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in (tier_path, task_path)
        ],
        "figures": [
            {"path": name, "sha256": sha256_file(args.output / name)} for name in figure_files
        ],
    }
    (args.output / f"{prefix}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(figure_files)} figure files to {args.output}")


if __name__ == "__main__":
    main()
