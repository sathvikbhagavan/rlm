#!/usr/bin/env python3
"""Plot controlled evidence separating scale, chemistry-rule inference, and execution."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from plot_style import apply_paper_style

TIER_COLORS = {2: "#5302A3", 3: "#CB4679"}
TIER_MARKERS = {2: "o", 3: "s"}
ARM_COLORS = {"ordinary": "#5302A3", "predicate": "#CB4679", "executor": "#8A8F98"}
ARM_MARKERS = {"ordinary": "o", "predicate": "D"}
MODEL_DISPLAY_NAMES = {
    "Qwen 3.5": "Qwen3.5-397B-A17B",
    "Claude Haiku 4.5": "Claude Haiku 4.5",
}
CONTEXTS = ("100", "500", "5000", "50000", "full")
CONTEXT_LABELS = ("100", "500", "5k", "50k", "Full")

apply_paper_style()


def read_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    **row,
                    "tier": int(row["tier"]),
                    "repetition": int(row["repetition"]),
                    "question_count": int(row["question_count"]),
                    "f1": float(row["f1"]),
                }
            )
    return rows


def weighted_replication_means(
    rows: Iterable[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> dict[tuple[Any, ...], list[float]]:
    accum: dict[tuple[Any, ...], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        key = tuple(row[field] for field in key_fields) + (row["repetition"],)
        weight = row["question_count"]
        accum[key][0] += row["f1"] * weight
        accum[key][1] += weight
    replications: dict[tuple[Any, ...], list[tuple[int, float]]] = defaultdict(list)
    for key, (weighted_sum, total_weight) in accum.items():
        group, repetition = key[:-1], key[-1]
        replications[group].append((int(repetition), weighted_sum / total_weight))
    return {group: [value for _, value in sorted(values)] for group, values in replications.items()}


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty control cell")
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matched = [row for row in records if row["study"] == "matched_cardinality"]
    if {row["model_label"] for row in matched} != {"GPT-5 mini"}:
        raise ValueError("Matched-cardinality figure must contain only completed GPT-5-mini data")
    matched_reps = weighted_replication_means(matched, ("condition", "context", "tier"))

    oracle = [
        row
        for row in records
        if row["study"] == "oracle_predicate" and row["arm"] in {"ordinary", "predicate"}
    ]
    oracle_reps = weighted_replication_means(oracle, ("model_label", "context", "arm"))

    aggregates: list[dict[str, Any]] = []
    for (condition, context, tier), values in sorted(matched_reps.items()):
        mean, sd = mean_sd(values)
        aggregates.append(
            {
                "study": "matched_cardinality",
                "model_label": "GPT-5 mini",
                "arm": "matched",
                "condition": condition,
                "context": context,
                "tier": tier,
                "mean_f1": mean,
                "sd_f1": sd,
                "replications": len(values),
            }
        )
    for (model_label, context, arm), values in sorted(oracle_reps.items()):
        mean, sd = mean_sd(values)
        aggregates.append(
            {
                "study": "oracle_predicate",
                "model_label": model_label,
                "arm": arm,
                "condition": arm,
                "context": context,
                "tier": "",
                "mean_f1": mean,
                "sd_f1": sd,
                "replications": len(values),
            }
        )
    return aggregates


def style_axis(axis: plt.Axes) -> None:
    axis.set_ylim(-0.02, 1.055)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.grid(axis="y", color="#D8DDE2", linewidth=0.55, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, width=0.6)


def matched_lookup(
    aggregates: list[dict[str, Any]], *, condition: str, tier: int
) -> dict[str, dict[str, Any]]:
    return {
        row["context"]: row
        for row in aggregates
        if row["study"] == "matched_cardinality"
        and row["condition"] == condition
        and row["tier"] == tier
    }


def plot_matched_scale(axis: plt.Axes, aggregates: list[dict[str, Any]]) -> None:
    positions = np.arange(len(CONTEXTS))
    for tier in (2, 3):
        values = {
            row["context"]: row
            for row in aggregates
            if row["study"] == "matched_cardinality"
            and str(row["condition"]).startswith("scale-")
            and row["tier"] == tier
        }
        axis.errorbar(
            positions,
            [values[context]["mean_f1"] for context in CONTEXTS],
            yerr=[values[context]["sd_f1"] for context in CONTEXTS],
            color=TIER_COLORS[tier],
            marker=TIER_MARKERS[tier],
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            capsize=2.2,
            elinewidth=0.8,
            label=f"Tier {tier}",
            zorder=3,
        )
    axis.set_xticks(positions, CONTEXT_LABELS)
    axis.set_xlabel("Corpus size $N$")
    axis.set_ylabel("Macro F1")
    axis.set_title("(a) Scale ($K=1$)", loc="left", pad=5)
    style_axis(axis)


def plot_matched_cardinality(axis: plt.Axes, aggregates: list[dict[str, Any]]) -> None:
    conditions = (
        ("scale-x5000-k1", "1"),
        ("cardinality-x5000-k5", "5"),
        ("cardinality-x5000-k20", "20"),
    )
    positions = np.arange(len(conditions))
    for tier in (2, 3):
        values = []
        errors = []
        for condition, _ in conditions:
            rows = [
                row
                for row in aggregates
                if row["study"] == "matched_cardinality"
                and row["condition"] == condition
                and row["tier"] == tier
            ]
            if len(rows) != 1:
                raise ValueError(f"Missing matched-cardinality aggregate: {condition}, Tier {tier}")
            values.append(rows[0]["mean_f1"])
            errors.append(rows[0]["sd_f1"])
        axis.errorbar(
            positions,
            values,
            yerr=errors,
            color=TIER_COLORS[tier],
            marker=TIER_MARKERS[tier],
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            capsize=2.2,
            elinewidth=0.8,
            zorder=3,
        )
    axis.set_xticks(positions, [label for _, label in conditions])
    axis.set_xlabel("Positive reactions $K$")
    axis.set_title("(b) Cardinality ($N=5{,}000$)", loc="left", pad=5)
    style_axis(axis)


def plot_oracle(
    axis: plt.Axes,
    aggregates: list[dict[str, Any]],
    *,
    model_label: str,
    panel: str,
) -> None:
    contexts = ("100", "500", "full")
    positions = np.arange(len(contexts))
    series: dict[str, list[float]] = {}
    for arm in ("ordinary", "predicate"):
        values = {
            row["context"]: row
            for row in aggregates
            if row["study"] == "oracle_predicate"
            and row["model_label"] == model_label
            and row["arm"] == arm
        }
        series[arm] = [values[context]["mean_f1"] for context in contexts]
        axis.errorbar(
            positions,
            series[arm],
            yerr=[values[context]["sd_f1"] for context in contexts],
            color=ARM_COLORS[arm],
            marker=ARM_MARKERS[arm],
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            capsize=2.2,
            elinewidth=0.8,
            label="Ordinary RLM" if arm == "ordinary" else "Rule-supplied RLM",
            zorder=3,
        )
    axis.axhline(
        1.0,
        color=ARM_COLORS["executor"],
        linestyle=(0, (3, 2)),
        linewidth=1.2,
        label="Deterministic executor",
        zorder=2,
    )
    delta = series["predicate"][-1] - series["ordinary"][-1]
    axis.annotate(
        f"{delta:+.3f}",
        xy=(2, series["predicate"][-1]),
        xytext=(1.78, min(0.92, series["predicate"][-1] + 0.14)),
        color=ARM_COLORS["predicate"],
        fontsize=7,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": ARM_COLORS["predicate"], "lw": 0.8},
    )
    axis.set_xticks(positions, ("100", "500", "Full"))
    axis.set_xlabel("Corpus size")
    display_name = MODEL_DISPLAY_NAMES.get(model_label, model_label)
    axis.set_title(
        f"({panel}) Rule supplied\n{display_name}",
        loc="left",
        pad=4,
        fontsize=8.0,
    )
    style_axis(axis)


def write_aggregates(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "study",
        "model_label",
        "arm",
        "condition",
        "context",
        "tier",
        "mean_f1",
        "sd_f1",
        "replications",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("paper_plots/gold/iclr2027/causal_controls/records.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper_plots/figures/gold"))
    args = parser.parse_args()

    records = read_records(args.records)
    aggregates = aggregate_rows(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_aggregates(args.records.with_name("aggregates.csv"), aggregates)

    figure, axes = plt.subplots(1, 4, figsize=(7.1, 2.45), sharey=True)
    plot_matched_scale(axes[0], aggregates)
    plot_matched_cardinality(axes[1], aggregates)
    plot_oracle(axes[2], aggregates, model_label="Qwen 3.5", panel="c")
    plot_oracle(axes[3], aggregates, model_label="Claude Haiku 4.5", panel="d")
    axes[1].set_ylabel("")

    tier_handles, tier_labels = axes[0].get_legend_handles_labels()
    rule_handles, rule_labels = axes[2].get_legend_handles_labels()
    figure.legend(
        tier_handles + rule_handles,
        tier_labels + rule_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.3,
    )
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.28, top=0.95, wspace=0.20)

    pdf_path = args.output_dir / "iclr_causal_controls.pdf"
    png_path = args.output_dir / "iclr_causal_controls.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(png_path, dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    print(f"Wrote {pdf_path} and {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
