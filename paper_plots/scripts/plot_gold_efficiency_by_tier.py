#!/usr/bin/env python3
"""Plot cost, token, and wall-time scaling from the gold benchmark records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

METHODS = ("llm", "codeact", "rlm")
METHOD_LABELS = {"llm": "LLM", "codeact": "CodeAct", "rlm": "RLM"}
COLORS = {"llm": "#0072B2", "codeact": "#D55E00", "rlm": "#009E73"}
MARKERS = {"llm": "o", "codeact": "s", "rlm": "D"}
TIER_NAMES = {
    1: "Structural lookup",
    2: "Property aggregation",
    3: "Reaction-level reasoning",
    4: "Relational reasoning",
}
TIER_QUESTIONS = {1: 10, 2: 20, 3: 35, 4: 35}
METRICS = {
    "cost": {
        "column": "cost_chf_per_trajectory",
        "ylabel": "Recorded cost per answered trajectory (CHF)",
        "title": "Mean recorded billed cost across paid models",
        "filename": "cost_by_tier_across_models",
        "n_column": "n_paid_models",
        "final_column": "paid_cost_is_final",
        "note": "Paid models only: Gemini, GPT-5 mini, and Claude. Free SwissAI models are "
        "excluded; failed-attempt cost metadata are unavailable.",
    },
    "tokens": {
        "column": "tokens_per_trajectory",
        "ylabel": "Recorded tokens per answered trajectory",
        "title": "Mean token use across models",
        "filename": "tokens_by_tier_across_models",
        "n_column": "n_models",
        "final_column": "is_final",
        "note": "Unweighted mean ± SEM across terminal model arms. Accounting covers answered "
        "trajectories; failed-attempt token metadata are unavailable.",
    },
    "wall_time": {
        "column": "wall_time_seconds_per_trajectory",
        "ylabel": "Wall time per successful trajectory (s)",
        "title": "Mean wall time for successful answers",
        "filename": "wall_time_by_tier_across_models",
        "n_column": "n_models",
        "final_column": "is_final",
        "note": "Unweighted mean ± SEM across terminal model arms. Only successful jobs enter "
        "the wall-time numerator and denominator; failed jobs are excluded.",
    },
}

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def as_bool(value: str) -> bool:
    return value.casefold() == "true"


def efficiency_figure(
    rows: list[dict[str, str]],
    *,
    column: str,
    ylabel: str,
    title: str,
    n_column: str,
    final_column: str,
    note: str,
) -> plt.Figure:
    contexts = ("100", "500", "1000", "full")
    positions = {context: index for index, context in enumerate(contexts)}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharex=True, sharey=True)
    axes = axes.ravel()
    mean_column = f"mean_{column}"
    sem_column = f"sem_{column}"

    for tier, axis in enumerate(axes, start=1):
        axis.axvspan(2.72, 3.28, color="#F1F3F5", zorder=0)
        for method in METHODS:
            subset = [
                row
                for row in rows
                if row["method"] == method
                and int(row["tier"]) == tier
                and row[mean_column]
                and float(row[mean_column]) > 0
            ]
            subset.sort(key=lambda row: positions[row["context"]])
            if not subset:
                continue
            x_values = [positions[row["context"]] for row in subset]
            y_values = [float(row[mean_column]) for row in subset]
            axis.plot(x_values, y_values, color=COLORS[method], linewidth=1.65, zorder=2)
            for x_value, y_value, row in zip(x_values, y_values, subset, strict=True):
                sem = float(row[sem_column]) if row[sem_column] else 0.0
                lower_error = min(sem, y_value * 0.95)
                final = as_bool(row[final_column])
                axis.errorbar(
                    [x_value],
                    [y_value],
                    yerr=np.array([[lower_error], [sem]]),
                    color=COLORS[method],
                    marker=MARKERS[method],
                    markerfacecolor=COLORS[method] if final else "white",
                    markersize=5.0,
                    markeredgewidth=1.0,
                    capsize=2.5,
                    elinewidth=0.9,
                    zorder=3,
                )
                if not final:
                    axis.annotate(
                        f"* n={row[n_column]}",
                        (x_value, y_value + sem),
                        xytext=(3, 4),
                        textcoords="offset points",
                        fontsize=6.5,
                        color=COLORS[method],
                    )
        axis.set_title(
            f"({chr(96 + tier)}) Tier {tier}: {TIER_NAMES[tier]} ($n={TIER_QUESTIONS[tier]}$)",
            loc="left",
            pad=5,
        )
        axis.set_yscale("log")
        axis.set_xlim(-0.18, 3.2)
        axis.grid(axis="y", which="both", color="#D8DDE2", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=0.6)

    for axis in axes[2:]:
        axis.set_xticks(range(len(contexts)), ("100", "500", "1000", "Full"))
        axis.set_xlabel("Context size (reactions)")
    fig.supylabel(ylabel, x=0.012, fontsize=8)

    provisional_methods = {
        method: any(row["method"] == method and not as_bool(row[final_column]) for row in rows)
        for method in METHODS
    }
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=5,
            label=("*" if provisional_methods[method] else "") + METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    fig.suptitle(title, y=0.99, fontsize=10, fontweight="bold")
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.947),
        handlelength=2.2,
    )
    fig.text(
        0.5,
        0.008,
        note,
        ha="center",
        va="bottom",
        fontsize=6.6,
        color="#555555",
    )
    fig.subplots_adjust(top=0.82, bottom=0.12, hspace=0.35, wspace=0.20)
    return fig


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    metadata = {
        "Creator": "RxnHaystack gold plotting pipeline",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, metadata=metadata)
    figure.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("paper_plots/gold/iclr2027"))
    parser.add_argument("--output", type=Path, default=Path("paper_plots/figures/gold"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.gold / "tier_efficiency_across_models.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    figures = []
    for specification in METRICS.values():
        figure = efficiency_figure(
            rows,
            column=specification["column"],
            ylabel=specification["ylabel"],
            title=specification["title"],
            n_column=specification["n_column"],
            final_column=specification["final_column"],
            note=specification["note"],
        )
        save_figure(figure, args.output, specification["filename"])
        plt.close(figure)
        figures.extend([f"{specification['filename']}.pdf", f"{specification['filename']}.png"])
    (args.output / "efficiency_by_tier_manifest.json").write_text(
        json.dumps(
            {
                "figures": figures,
                "note": "Resource metrics describe answered trajectories; failed-attempt "
                "token and cost metadata are unavailable.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {len(METRICS)} efficiency figures to {args.output}")


if __name__ == "__main__":
    main()
