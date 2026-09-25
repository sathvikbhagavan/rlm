#!/usr/bin/env python3
"""Plot trace-supported failure mechanisms against benchmark capabilities."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from plot_style import apply_paper_style

PURPLE = "#5302A3"
PINK = "#CB4679"
LIGHT_PINK = "#E9A4BF"
GRID = "#D9D9E2"

COLUMNS = (
    ("access_controlled", "Evidence\naccessible"),
    ("structured_execution", "Structured\nexecution"),
    ("chemical_abstraction", "Chemical\nabstraction"),
    ("relational_orchestration", "Relational\norchestration"),
    ("route_reasoning", "Target/route\nreasoning"),
)

ROW_LABELS = {
    "unsupported action wrapper or pseudo-execution": "Tool action not executed",
    "corpus parsed incorrectly or chemistry code crashes": "Corpus parse or code failure",
    "literal names or narrow SMARTS replace the required structural rule": (
        "Literal names or narrow SMARTS"
    ),
    "atom correspondence approximated by local signatures or invalid MCS logic": (
        "Missing atom correspondence"
    ),
    "subcall decisions are lost or not aggregated": "Subcall results not preserved",
    "common reagent creates a false graph edge": "Common reagent creates graph edge",
    "under-enumeration or overbroad chain rules under full-corpus RLM": (
        "Incomplete or overbroad chains"
    ),
    "target-specific shortcuts fail after graph construction": "Target-specific route shortcut",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["observed_mechanism"] in ROW_LABELS]


def plot(data: Path, output: Path) -> None:
    rows = read_rows(data)
    apply_paper_style()
    figure, axis = plt.subplots(figsize=(7.15, 3.15))
    axis.set_xlim(-0.5, len(COLUMNS) - 0.5)
    axis.set_ylim(len(rows) - 0.5, -0.5)

    for row_index, row in enumerate(rows):
        for column_index, (column, _) in enumerate(COLUMNS):
            value = int(row[column])
            color = "white"
            if column == "access_controlled" and value:
                color = PURPLE
            elif value == 2:
                color = PINK
            elif value == 1:
                color = LIGHT_PINK
            axis.add_patch(
                Rectangle(
                    (column_index - 0.45, row_index - 0.38),
                    0.9,
                    0.76,
                    facecolor=color,
                    edgecolor=GRID,
                    linewidth=0.65,
                )
            )
    axis.axvline(0.5, color="#888895", linewidth=0.8, linestyle="--")
    axis.set_xticks(range(len(COLUMNS)), [label for _, label in COLUMNS])
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=5)
    axis.set_yticks(range(len(rows)), [ROW_LABELS[row["observed_mechanism"]] for row in rows])
    axis.tick_params(axis="y", length=0, pad=6)
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.legend(
        handles=(
            Patch(facecolor=PURPLE, label="evidence access controlled"),
            Patch(facecolor=PINK, label="primary diagnostic locus"),
            Patch(facecolor=LIGHT_PINK, label="contributing locus"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncols=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.7,
    )
    figure.subplots_adjust(left=0.36, right=0.99, top=0.82, bottom=0.19)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    result.add_argument(
        "--data",
        type=Path,
        default=root / "paper_plots/gold/iclr2027/failure_analysis/diagnostic_objective_map.csv",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=root / "paper_plots/figures/gold/failure_analysis/failure_diagnostic_map.pdf",
    )
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    plot(arguments.data, arguments.output)
