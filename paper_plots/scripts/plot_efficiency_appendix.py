#!/usr/bin/env python3
"""Plot calls, tokens, latency, tool time, wall time, and memory for the appendix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import METHOD_COLORS, apply_paper_style

CONTEXTS = ("100", "500", "full")
CONTEXT_LABELS = {"100": "100", "500": "500", "full": "Full corpus"}
COLORS = {
    "100": METHOD_COLORS["llm"],
    "500": METHOD_COLORS["codeact"],
    "full": METHOD_COLORS["rlm"],
}
MARKERS = {"100": "o", "500": "s", "full": "D"}
METRICS = (
    ("calls_per_trajectory", "Model calls / trajectory"),
    ("tokens_per_trajectory", "Tokens / trajectory"),
    ("latency_seconds_per_trajectory", "Recorded latency (s) / trajectory"),
    ("tool_time_seconds_per_trajectory", "Tool time (s) / trajectory"),
    ("wall_time_seconds_per_trajectory", "Wall time (s) / trajectory"),
    ("peak_memory_mib_per_job", "Peak memory (MiB) / job"),
)

apply_paper_style()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def make_figure(rows: list[dict[str, str]]) -> plt.Figure:
    rlm_rows = [row for row in rows if row["method"] == "rlm"]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.8), sharex=True)
    for panel, (axis, (metric, title)) in enumerate(
        zip(axes.ravel(), METRICS, strict=True), start=1
    ):
        mean_column = f"mean_{metric}"
        sem_column = f"sem_{metric}"
        for context in CONTEXTS:
            subset = sorted(
                (row for row in rlm_rows if row["context"] == context and row[mean_column]),
                key=lambda row: int(row["tier"]),
            )
            x_values = [int(row["tier"]) for row in subset]
            y_values = [float(row[mean_column]) for row in subset]
            errors = [float(row[sem_column]) if row[sem_column] else 0.0 for row in subset]
            axis.errorbar(
                x_values,
                y_values,
                yerr=errors,
                color=COLORS[context],
                marker=MARKERS[context],
                markersize=4.5,
                capsize=2.2,
                elinewidth=0.8,
                label=CONTEXT_LABELS[context],
            )
        axis.set_title(f"({chr(96 + panel)}) {title}", loc="left", pad=4)
        axis.set_yscale("log")
        axis.set_xticks((1, 2, 3, 4))
        axis.grid(axis="y", which="both", color="#D8DDE2", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=0.6)
    for axis in axes[1]:
        axis.set_xlabel("Benchmark tier")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.89, bottom=0.10, hspace=0.34, wspace=0.25)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("paper_plots/gold/iclr2027/efficiency_appendix/across_models.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("paper_plots/figures/gold"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figure = make_figure(read_csv(args.data))
    metadata = {
        "Creator": "RxnHaystack gold plotting pipeline",
        "CreationDate": None,
        "ModDate": None,
    }
    pdf = args.output / "efficiency_diagnostics.pdf"
    png = args.output / "efficiency_diagnostics.png"
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.02, metadata=metadata)
    figure.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    manifest = {
        "data": str(args.data),
        "figure_files": [pdf.name, png.name],
        "metrics": [metric for metric, _ in METRICS],
    }
    (args.output / "efficiency_diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {pdf} and {png}")


if __name__ == "__main__":
    main()
