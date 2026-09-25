#!/usr/bin/env python3
"""Plot the main GPT-5-mini results directly from the frozen gold records."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    from plot_style import (
        HEATMAP_CMAP,
        METHOD_COLORS,
        METHOD_LABELS,
        METHOD_MARKERS,
        apply_paper_style,
    )
except ModuleNotFoundError:  # Imported as a package by tests.
    from paper_plots.scripts.plot_style import (
        HEATMAP_CMAP,
        METHOD_COLORS,
        METHOD_LABELS,
        METHOD_MARKERS,
        apply_paper_style,
    )

MODEL = "gpt-5-mini"
METHODS = ("llm", "codeact", "rlm")
CONTEXTS = ("100", "500", "full")
CONTEXT_LABELS = ("100", "500", "Full")
TIER_NAMES = {
    1: "Structural lookup",
    2: "Property aggregation",
    3: "Reaction-level reasoning",
    4: "Relational reasoning",
}
TIER_QUESTIONS = {1: 10, 2: 20, 3: 35, 4: 35}
CAPABILITY_GROUPS = {
    3: (
        ("Bond-level", ("tier3/task13", "tier3/task14", "tier3/task15")),
        ("Scaffold", ("tier3/task17", "tier3/task18", "tier3/task20")),
        ("Reagent", ("tier3/task21", "tier3/task22")),
        ("Stereochemistry", ("tier3/task23", "tier3/task24")),
        ("Functional group", ("tier3/task6", "tier3/task7", "tier3/task8")),
        ("Mechanism", ("tier3/task9", "tier3/task10", "tier3/task10b")),
    ),
    4: (
        ("Mechanical graph", ("tier4/task11", "tier4/task12", "tier4/task12b")),
        ("Chemical constraints", ("tier4/task13", "tier4/task14", "tier4/task15")),
        (
            "Route / multi-constraint",
            ("tier4/task16", "tier4/task17", "tier4/task17b"),
        ),
    ),
}
COLUMNS = (
    ("llm", "100"),
    ("codeact", "100"),
    ("rlm", "100"),
    ("llm", "500"),
    ("codeact", "500"),
    ("rlm", "500"),
    ("rlm", "full"),
)
COLUMN_LABELS = (
    "LLM\n100",
    "CodeAct\n100",
    "RLM\n100",
    "LLM\n500",
    "CodeAct\n500",
    "RLM\n500",
    "RLM\nFull",
)

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
    figure.savefig(
        output / f"{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata=metadata,
    )
    figure.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def plot_performance(rows: list[dict[str, str]]) -> plt.Figure:
    selected = [
        row
        for row in rows
        if row["model"] == MODEL and row["context"] in CONTEXTS and row["method"] in METHODS
    ]
    lookup = {(int(row["tier"]), row["method"], row["context"]): row for row in selected}
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.55), sharex=True, sharey=True)
    positions = np.arange(len(CONTEXTS))
    for tier, axis in enumerate(axes.ravel(), start=1):
        axis.axvspan(1.72, 2.28, color="#F1F3F5", zorder=0)
        for method in METHODS:
            contexts = [context for context in CONTEXTS if (tier, method, context) in lookup]
            points = [lookup[(tier, method, context)] for context in contexts]
            axis.errorbar(
                [CONTEXTS.index(context) for context in contexts],
                [float(row["f1"]) for row in points],
                yerr=[float(row["f1_std"]) for row in points],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=4.8,
                markeredgecolor="white",
                markeredgewidth=0.45,
                capsize=3.4,
                capthick=1.1,
                elinewidth=1.15,
                zorder=3,
            )
        axis.set_title(
            f"({chr(96 + tier)}) Tier {tier}: {TIER_NAMES[tier]} ($n={TIER_QUESTIONS[tier]}$)",
            loc="left",
            pad=5,
        )
        axis.set_xlim(-0.18, 2.2)
        axis.set_ylim(-0.02, 1.06)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.grid(axis="y", color="#D8DDE2", linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=0.6)
    for axis in axes[1]:
        axis.set_xticks(positions, CONTEXT_LABELS)
        axis.set_xlabel("Accessible reactions / RLM chunk size")
    for axis in axes[:, 0]:
        axis.set_ylabel("Macro F1")
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=2.2,
    )
    figure.subplots_adjust(top=0.89, hspace=0.35, wspace=0.20)
    return figure


def capability_matrix(records: list[dict[str, str]], tier: int) -> tuple[np.ndarray, list[str]]:
    selected = [
        row
        for row in records
        if row["model"] == MODEL and int(row["tier"]) == tier and row["context"] in CONTEXTS
    ]
    task_to_group = {task: label for label, tasks in CAPABILITY_GROUPS[tier] for task in tasks}
    by_replication: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        label = task_to_group.get(row["task"])
        if label is None:
            continue
        key = (label, row["method"], row["context"], int(row["repetition"]))
        by_replication[key].append(row)

    replication_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (label, method, context, _), members in by_replication.items():
        weight = sum(int(row["question_count"]) for row in members)
        score = sum(
            int(row["question_count"])
            * (float(row["f1"]) if row["status"] == "succeeded" and row["f1"] else 0.0)
            for row in members
        )
        replication_scores[(label, method, context)].append(score / weight)

    labels = [label for label, _ in CAPABILITY_GROUPS[tier]]
    values = np.full((len(labels), len(COLUMNS)), np.nan)
    for row_index, label in enumerate(labels):
        for column_index, (method, context) in enumerate(COLUMNS):
            scores = replication_scores.get((label, method, context))
            if scores:
                values[row_index, column_index] = statistics.fmean(scores)
    return values, labels


def plot_capabilities(records: list[dict[str, str]]) -> plt.Figure:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.15),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    image = None
    for axis, tier in zip(axes, (3, 4), strict=True):
        values, labels = capability_matrix(records, tier)
        image = axis.imshow(values, cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto")
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if np.isnan(value):
                    continue
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.1,
                    color="white" if value < 0.62 else "#111111",
                )
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xticks(range(len(COLUMNS)), COLUMN_LABELS)
        axis.tick_params(axis="both", length=0)
        axis.xaxis.tick_top()
        axis.axvline(2.5, color="white", linewidth=2)
        axis.axvline(5.5, color="white", linewidth=2)
        axis.set_title(f"Tier {tier}: {TIER_NAMES[tier]}", loc="left", fontsize=9, pad=11)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.subplots_adjust(left=0.245, right=0.91, top=0.91, bottom=0.04, hspace=0.52)
    colorbar_axis = figure.add_axes([0.93, 0.15, 0.015, 0.65])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Macro F1", rotation=90, labelpad=6)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("paper_plots/gold/iclr2027"))
    parser.add_argument("--output", type=Path, default=Path("paper_plots/figures/gold"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scaling = read_csv(args.gold / "tier_scaling.csv")
    records = read_csv(args.gold / "full_benchmark_records.csv")
    save(plot_performance(scaling), args.output, "performance_overview")
    save(plot_capabilities(records), args.output, "capability_map")
    print(f"Wrote main-text figures to {args.output}")


if __name__ == "__main__":
    main()
