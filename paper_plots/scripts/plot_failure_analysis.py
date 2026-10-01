#!/usr/bin/env python3
"""Plot corrected job outcomes and trace-observed set-error signatures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import apply_paper_style

METHOD_LABELS = {"llm": "LLM", "codeact": "CodeAct", "rlm": "RLM"}
PINK = "#D34A78"
PURPLE = "#5B00B5"
ORANGE = "#F99A31"
PALE = "#A8A8B3"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data = root / "paper_plots/gold/iclr2027/failure_analysis"
    out = root / "paper_plots/figures/gold/failure_analysis"
    outcomes = read(data / "job_outcomes.csv")
    signatures = read(data / "trace_error_signatures.csv")
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    methods = ("llm", "codeact", "rlm")
    x = np.arange(len(methods))

    bottom = np.zeros(len(methods))
    outcome_colors = {
        "exact": PURPLE,
        "partial": PINK,
        "zero score": ORANGE,
        "execution failure": PALE,
    }
    for category in outcome_colors:
        values = np.array(
            [
                100
                * float(
                    next(
                        r["fraction"]
                        for r in outcomes
                        if r["method"] == m and r["outcome"] == category
                    )
                )
                for m in methods
            ]
        )
        axes[0].bar(x, values, bottom=bottom, color=outcome_colors[category], label=category)
        bottom += values
    axes[0].set_title("(a) Outcomes across valid jobs", loc="left")
    axes[0].set_ylabel("Jobs (%)")
    axes[0].set_xticks(x, [METHOD_LABELS[m] for m in methods])
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, fontsize=7, ncols=2, loc="lower center")

    bottom = np.zeros(len(methods))
    signature_colors = {
        "empty answer": PALE,
        "omissions only": PURPLE,
        "extra selections only": ORANGE,
        "mixed omissions and extras": PINK,
    }
    for category in signature_colors:
        values = np.array(
            [
                100
                * float(
                    next(
                        r["fraction_of_errors"]
                        for r in signatures
                        if r["method"] == m and r["signature"] == category
                    )
                )
                for m in methods
            ]
        )
        axes[1].bar(x, values, bottom=bottom, color=signature_colors[category], label=category)
        bottom += values
    axes[1].set_title("(b) Set-error signatures in retained traces", loc="left")
    axes[1].set_ylabel("Erroneous question outputs (%)")
    axes[1].set_xticks(x, [METHOD_LABELS[m] for m in methods])
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False, fontsize=6.7, loc="upper center")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"failure_analysis.{suffix}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
