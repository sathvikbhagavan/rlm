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
from matplotlib.ticker import LogLocator, NullFormatter

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
PAID_MODELS = ("gemini-3.7-flash", "gpt-5-mini", "claude-haiku-4.5")
PAID_MODEL_LABELS = {
    "gemini-3.7-flash": "Gemini 3.7 Flash",
    "gpt-5-mini": "GPT-5 mini",
    "claude-haiku-4.5": "Claude Haiku 4.5",
}
METHODS = ("llm", "codeact", "rlm")
CONTEXTS = ("100", "500", "1000", "full")
CONTEXT_LABELS = ("100", "500", "1000", "Full")
CONTEXT_MARKERS = {"100": "o", "500": "s", "1000": "^", "full": "D"}
EFFICIENCY_CONTEXTS = ("100", "500", "full")
EFFICIENCY_CONTEXT_LABELS = ("100", "500", "Full")
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
        axis.axvspan(2.72, 3.28, color="#F1F3F5", zorder=0)
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
        axis.set_xlim(-0.18, 3.2)
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


def human_tier_f1(rows: list[dict[str, str]]) -> dict[int, float]:
    """Return the frozen first-submission human F1 reference for each tier."""
    selected = [row for row in rows if row["population"] == "human_assigned_items"]
    result = {int(row["tier"]): float(row["mean_f1"]) for row in selected}
    if set(result) != set(TIER_NAMES) or len(selected) != len(result):
        raise ValueError("Expected exactly one human-assigned result for each tier")
    return result


def plot_aggregate_performance(
    rows: list[dict[str, str]],
    *,
    human_rows: list[dict[str, str]] | None = None,
    kind: str = "line",
) -> plt.Figure:
    """Plot the core benchmark mean across terminal model arms in four panels."""
    if kind not in {"line", "bar"}:
        raise ValueError(f"Unsupported aggregate plot kind: {kind}")
    selected = [row for row in rows if row["context"] in CONTEXTS and row["method"] in METHODS]
    lookup = {(int(row["tier"]), row["method"], row["context"]): row for row in selected}
    positions = np.arange(len(CONTEXTS), dtype=float)
    offsets = {"llm": -0.23, "codeact": 0.0, "rlm": 0.23}
    figure, axes = plt.subplots(1, 4, figsize=(7.35, 2.55), sharex=True, sharey=True)
    human_scores = human_tier_f1(human_rows) if human_rows is not None else {}

    for tier, axis in enumerate(axes, start=1):
        axis.axvspan(2.72, 3.28, color="#F1F3F5", zorder=0)
        if tier in human_scores:
            axis.axhline(
                human_scores[tier],
                color="#444444",
                linestyle=(0, (4, 2.4)),
                linewidth=1.15,
                zorder=2,
            )
        for method in METHODS:
            contexts = [context for context in CONTEXTS if (tier, method, context) in lookup]
            points = [lookup[(tier, method, context)] for context in contexts]
            x_values = np.asarray([CONTEXTS.index(context) for context in contexts], dtype=float)
            means = np.asarray([float(row["mean_f1"]) for row in points])
            errors = np.asarray(
                [float(row["model_sem"]) if row["model_sem"] else 0.0 for row in points]
            )
            if kind == "bar":
                bar_x = x_values + offsets[method]
                if contexts == ["full"]:
                    bar_x = x_values
                axis.bar(
                    bar_x,
                    means,
                    width=0.21,
                    yerr=errors,
                    color=METHOD_COLORS[method],
                    edgecolor="white",
                    linewidth=0.45,
                    capsize=2.0,
                    error_kw={"elinewidth": 0.8, "capthick": 0.8},
                    zorder=3,
                )
            else:
                axis.errorbar(
                    x_values,
                    means,
                    yerr=errors,
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    markersize=4.7,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    capsize=2.5,
                    elinewidth=0.9,
                    zorder=3,
                )
        axis.set_title(
            f"({chr(96 + tier)}) Tier {tier}\n{TIER_NAMES[tier]} ($n={TIER_QUESTIONS[tier]}$)",
            loc="left",
            pad=5,
            fontsize=8.5,
        )
        axis.set_xlim(-0.38, 3.38)
        axis.set_ylim(-0.02, 1.06)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_xticks(positions, CONTEXT_LABELS)
        axis.set_xlabel("Context size")
        axis.grid(axis="y", color="#D8DDE2", linewidth=0.5, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.3, width=0.55)
    axes[0].set_ylabel("Macro F1")

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method] if kind == "line" else "s",
            linewidth=1.5 if kind == "line" else 0,
            markersize=5.0,
            markeredgecolor="white",
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    if human_scores:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#444444",
                linestyle=(0, (4, 2.4)),
                linewidth=1.15,
                label="Human baseline",
            )
        )
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=2.0,
    )
    figure.subplots_adjust(left=0.07, right=0.995, top=0.76, bottom=0.19, wspace=0.17)
    return figure


def human_tier_summary(
    item_rows: list[dict[str, str]],
) -> dict[int, dict[str, float]]:
    """Aggregate frozen human accuracy and non-additive timing summaries by tier."""
    summary: dict[int, dict[str, float]] = {}
    for tier in TIER_NAMES:
        submitted = [
            row for row in item_rows if int(row["tier"]) == tier and row["submitted"] == "True"
        ]
        scored = [row for row in submitted if not row["abstention"]]
        summary[tier] = {
            "f1": statistics.fmean(float(row["f1"]) for row in scored),
            "exact_match": statistics.fmean(float(row["exact_match"]) for row in scored),
            "median_active_minutes": statistics.median(
                float(row["active"]) / 60 for row in submitted
            ),
            "median_offline_minutes": statistics.median(
                float(row["offline_minutes"] or 0) for row in submitted
            ),
        }
    return summary


def plot_human_validation(item_rows: list[dict[str, str]]) -> plt.Figure:
    """Plot per-tier human scores and separately recorded time summaries."""
    summary = human_tier_summary(item_rows)
    tiers = np.arange(1, 5, dtype=float)
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))

    axes[0].bar(
        tiers - width / 2,
        [summary[tier]["f1"] for tier in TIER_NAMES],
        width,
        color="#2A6F97",
        label="Macro F1",
    )
    axes[0].bar(
        tiers + width / 2,
        [summary[tier]["exact_match"] for tier in TIER_NAMES],
        width,
        color="#90BE6D",
        label="Exact set match",
    )
    axes[0].set_ylim(0, 1.06)
    axes[0].set_ylabel("Score")
    axes[0].set_title("(a) Human baseline by tier", loc="left")
    axes[0].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )

    axes[1].bar(
        tiers - width / 2,
        [summary[tier]["median_active_minutes"] for tier in TIER_NAMES],
        width,
        color="#577590",
        label="Active browser time",
    )
    axes[1].bar(
        tiers + width / 2,
        [summary[tier]["median_offline_minutes"] for tier in TIER_NAMES],
        width,
        color="#F9C74F",
        label="Self-reported offline time",
    )
    axes[1].set_ylabel("Median minutes / response")
    axes[1].set_title("(b) Time recorded separately", loc="left")
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )

    for axis in axes:
        axis.set_xticks(tiers, [f"Tier {tier}" for tier in TIER_NAMES])
        axis.grid(axis="y", color="#D8DDE2", linewidth=0.5, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.3, width=0.55)
    figure.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.29, wspace=0.30)
    return figure


def efficiency_points(
    records: list[dict[str, str]],
    *,
    model: str = MODEL,
) -> dict[tuple[int, str, str], tuple[float, float, float]]:
    """Aggregate one model's F1 and USD cost per question trajectory.

    Each repetition receives equal weight. Within a repetition, task scores are
    weighted by their question counts and terminal failures contribute zero.
    Recorded API cost is divided by the same number of question trajectories.
    """
    by_repetition: dict[tuple[int, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        if (
            row["model"] == model
            and row["method"] in METHODS
            and row["context"] in EFFICIENCY_CONTEXTS
        ):
            key = (
                int(row["tier"]),
                row["method"],
                row["context"],
                int(row["repetition"]),
            )
            by_repetition[key].append(row)

    repetition_values: dict[tuple[int, str, str], list[tuple[float, float]]] = defaultdict(list)
    for (tier, method, context, _), members in by_repetition.items():
        trajectories = sum(int(row["question_count"]) for row in members)
        if trajectories == 0:
            continue
        f1 = (
            sum(
                int(row["question_count"])
                * (float(row["f1"]) if row["status"] == "succeeded" and row["f1"] else 0.0)
                for row in members
            )
            / trajectories
        )
        cost = sum(float(row["cost_usd"] or 0.0) for row in members) / trajectories
        repetition_values[(tier, method, context)].append((f1, cost))

    points: dict[tuple[int, str, str], tuple[float, float, float]] = {}
    for key, values in repetition_values.items():
        f1_values = [value[0] for value in values]
        cost_values = [value[1] for value in values]
        points[key] = (
            statistics.fmean(cost_values),
            statistics.fmean(f1_values),
            statistics.pstdev(f1_values),
        )
    return points


def aggregate_efficiency_points(
    records: list[dict[str, str]],
) -> dict[tuple[int, str, str], tuple[float, float, float, float, int]]:
    """Aggregate accuracy and recorded API cost over the paid model arms."""
    by_model = {model: efficiency_points(records, model=model) for model in PAID_MODELS}
    keys = set.intersection(*(set(points) for points in by_model.values()))
    aggregates: dict[tuple[int, str, str], tuple[float, float, float, float, int]] = {}
    for key in keys:
        costs = [by_model[model][key][0] for model in PAID_MODELS]
        scores = [by_model[model][key][1] for model in PAID_MODELS]
        count = len(scores)
        aggregates[key] = (
            statistics.fmean(costs),
            statistics.pstdev(costs) / np.sqrt(count),
            statistics.fmean(scores),
            statistics.pstdev(scores) / np.sqrt(count),
            count,
        )
    return aggregates


def efficiency_legend(figure: plt.Figure, *, top: float = 1.01) -> None:
    """Add the shared method and context legend to an efficiency figure."""
    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        for method in METHODS
    ]
    context_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=CONTEXT_MARKERS[context],
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            label=label,
        )
        for context, label in zip(
            EFFICIENCY_CONTEXTS, EFFICIENCY_CONTEXT_LABELS, strict=True
        )
    ]
    figure.legend(
        handles=method_handles + context_handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, top),
    )


def plot_efficiency_frontier(records: list[dict[str, str]]) -> plt.Figure:
    """Plot the paid-model mean accuracy--cost frontier by tier."""
    points = aggregate_efficiency_points(records)
    figure, axes = plt.subplots(1, 4, figsize=(7.35, 2.55), sharey=True)
    for tier, axis in enumerate(axes, start=1):
        for method in METHODS:
            contexts = [
                context
                for context in EFFICIENCY_CONTEXTS
                if (tier, method, context) in points
            ]
            contexts.sort(key=EFFICIENCY_CONTEXTS.index)
            costs = [points[(tier, method, context)][0] for context in contexts]
            cost_errors = [points[(tier, method, context)][1] for context in contexts]
            scores = [points[(tier, method, context)][2] for context in contexts]
            score_errors = [points[(tier, method, context)][3] for context in contexts]
            axis.plot(costs, scores, color=METHOD_COLORS[method], linewidth=1.25, zorder=2)
            for context, cost, cost_error, score, score_error in zip(
                contexts, costs, cost_errors, scores, score_errors, strict=True
            ):
                axis.errorbar(
                    cost,
                    score,
                    xerr=cost_error,
                    yerr=score_error,
                    color=METHOD_COLORS[method],
                    marker=CONTEXT_MARKERS[context],
                    markersize=4.8,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    capsize=2.0,
                    elinewidth=0.75,
                    zorder=3,
                )
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_ylim(-0.02, 1.06)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_title(
            f"({chr(96 + tier)}) Tier {tier}\n{TIER_NAMES[tier]} ($n={TIER_QUESTIONS[tier]}$)",
            loc="left",
            pad=5,
            fontsize=8.3,
        )
        axis.grid(color="#D8DDE2", linewidth=0.5, which="both")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=0.6)
        axis.set_xlabel("USD / trajectory", fontsize=7.2)
        axis.tick_params(axis="x", labelsize=6.5)
    axes[0].set_ylabel("Macro F1")
    efficiency_legend(figure, top=1.01)
    figure.subplots_adjust(left=0.075, right=0.995, top=0.75, bottom=0.20, wspace=0.18)
    return figure


def plot_efficiency_frontier_by_model(records: list[dict[str, str]]) -> plt.Figure:
    """Plot the same frontier separately for every paid model and tier."""
    figure, axes = plt.subplots(3, 4, figsize=(7.35, 6.35), sharey=True)
    for row_index, model in enumerate(PAID_MODELS):
        points = efficiency_points(records, model=model)
        for column, tier in enumerate(range(1, 5)):
            axis = axes[row_index, column]
            for method in METHODS:
                contexts = [
                    context
                    for context in EFFICIENCY_CONTEXTS
                    if (tier, method, context) in points
                ]
                contexts.sort(key=EFFICIENCY_CONTEXTS.index)
                costs = [points[(tier, method, context)][0] for context in contexts]
                scores = [points[(tier, method, context)][1] for context in contexts]
                errors = [points[(tier, method, context)][2] for context in contexts]
                axis.plot(
                    costs,
                    scores,
                    color=METHOD_COLORS[method],
                    linewidth=1.1,
                    zorder=2,
                )
                for context, cost, score, error in zip(
                    contexts, costs, scores, errors, strict=True
                ):
                    axis.errorbar(
                        cost,
                        score,
                        yerr=error,
                        color=METHOD_COLORS[method],
                        marker=CONTEXT_MARKERS[context],
                        markersize=3.8,
                        markeredgecolor="white",
                        markeredgewidth=0.4,
                        capsize=1.8,
                        elinewidth=0.65,
                        zorder=3,
                    )
            axis.set_xscale("log")
            axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_ylim(-0.02, 1.06)
            axis.set_yticks(np.linspace(0, 1, 6))
            axis.grid(color="#D8DDE2", linewidth=0.45, which="both")
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(length=2.2, width=0.55, labelsize=6.5)
            if row_index == 0:
                axis.set_title(f"Tier {tier}\n{TIER_NAMES[tier]}", fontsize=8.3, pad=5)
            if column == 0:
                axis.set_ylabel(f"{PAID_MODEL_LABELS[model]}\nMacro F1", fontsize=7.2)
            if row_index == len(PAID_MODELS) - 1:
                axis.set_xlabel("USD / trajectory", fontsize=7.0)
            else:
                axis.tick_params(axis="x", labelbottom=False)
    efficiency_legend(figure, top=0.995)
    figure.subplots_adjust(left=0.105, right=0.995, top=0.90, bottom=0.07, hspace=0.35, wspace=0.26)
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
    aggregate_scaling = read_csv(args.gold / "tier_scaling_across_models.csv")
    records = read_csv(args.gold / "full_benchmark_records.csv")
    human_directory = args.gold / "human_validation"
    human_comparison = read_csv(human_directory / "human_model_tier_comparison.csv")
    human_items = read_csv(human_directory / "item_metrics.csv")
    save(
        plot_aggregate_performance(aggregate_scaling, human_rows=human_comparison),
        args.output,
        "performance_overview",
    )
    save(plot_human_validation(human_items), args.output, "human_validation_summary")
    save(plot_capabilities(records), args.output, "capability_map")
    save(plot_efficiency_frontier(records), args.output, "efficiency_frontier")
    save(
        plot_efficiency_frontier_by_model(records),
        args.output,
        "efficiency_frontier_by_model",
    )
    print(f"Wrote main-text figures to {args.output}")


if __name__ == "__main__":
    main()
