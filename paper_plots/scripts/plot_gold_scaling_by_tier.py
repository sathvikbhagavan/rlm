#!/usr/bin/env python3
"""Plot the workshop-style four-tier scaling figure for all six models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

MODEL_ORDER = (
    "qwen3.5",
    "deepseek-v4-flash",
    "glm-5.2",
    "gemini-3.7-flash",
    "gpt-5-mini",
    "claude-haiku-4.5",
)
METHODS = ("llm", "codeact", "rlm")
METHOD_LABELS = {"llm": "LLM", "codeact": "CodeAct", "rlm": "RLM"}
COLORS = {"llm": "#0072B2", "codeact": "#D55E00", "rlm": "#009E73"}
MARKERS = {"llm": "o", "codeact": "s", "rlm": "D"}
MODEL_MARKERS = {
    "qwen3.5": "o",
    "deepseek-v4-flash": "s",
    "glm-5.2": "^",
    "gemini-3.7-flash": "D",
    "gpt-5-mini": "P",
    "claude-haiku-4.5": "X",
}
TIER_NAMES = {
    1: "Structural lookup",
    2: "Property aggregation",
    3: "Reaction-level reasoning",
    4: "Relational reasoning",
}
TIER_QUESTIONS = {1: 10, 2: 20, 3: 35, 4: 35}

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


def model_figure(
    model: str,
    rows: list[dict[str, str]],
    arms: dict[tuple[str, str], dict[str, str]],
) -> plt.Figure:
    model_rows = [row for row in rows if row["model"] == model]
    model_label = model_rows[0]["model_label"]
    has_x1000 = any(row["context"] == "1000" and row["f1"] for row in model_rows)
    contexts = ("100", "500", "1000", "full") if has_x1000 else ("100", "500", "full")
    positions = {context: index for index, context in enumerate(contexts)}
    full_position = positions["full"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.85), sharex=True, sharey=True)
    axes = axes.ravel()

    for tier, axis in enumerate(axes, start=1):
        axis.axvspan(full_position - 0.28, full_position + 0.28, color="#F1F3F5", zorder=0)
        for method in METHODS:
            subset = [
                row
                for row in model_rows
                if row["method"] == method and int(row["tier"]) == tier and row["f1"]
            ]
            subset.sort(key=lambda row: positions[row["context"]])
            if not subset:
                continue
            axis.errorbar(
                [positions[row["context"]] for row in subset],
                [float(row["f1"]) for row in subset],
                yerr=[float(row["f1_std"]) for row in subset],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4.2,
                markeredgecolor="white",
                markeredgewidth=0.45,
                capsize=2.2,
                elinewidth=0.9,
                zorder=3,
            )
        axis.set_title(
            f"({chr(96 + tier)}) Tier {tier}: {TIER_NAMES[tier]} ($n={TIER_QUESTIONS[tier]}$)",
            loc="left",
            pad=5,
        )
        axis.set_xlim(-0.18, full_position + 0.2)
        axis.set_ylim(-0.02, 1.06)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.grid(axis="y", color="#D8DDE2", linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=0.6)

    tick_labels = ["Full" if context == "full" else context for context in contexts]
    for axis in axes[2:]:
        axis.set_xticks(range(len(contexts)), tick_labels)
        axis.set_xlabel("Context size (reactions)")
    for axis in axes[::2]:
        axis.set_ylabel("Macro F1")

    incomplete = {method: not as_bool(arms[(model, method)]["is_final"]) for method in METHODS}
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=("*" if incomplete[method] else "") + METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    fig.suptitle(model_label, y=0.995, fontsize=10, fontweight="bold")
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        handlelength=2.2,
    )
    note = "Curves condition on successful runs."
    if any(incomplete.values()):
        note += "  * Arm has running, stale, pending, or unrun jobs in the gold snapshot."
    fig.text(0.5, 0.008, note, ha="center", va="bottom", fontsize=7, color="#555555")
    fig.subplots_adjust(top=0.82, bottom=0.12, hspace=0.35, wspace=0.20)
    return fig


def combined_figure(
    rows: list[dict[str, str]],
    arms: dict[tuple[str, str], dict[str, str]],
) -> plt.Figure:
    """Plot every model in the same four tier panels."""
    contexts = ("100", "500", "1000", "full")
    positions = {context: index for index, context in enumerate(contexts)}
    model_offsets = {
        model: offset
        for model, offset in zip(
            MODEL_ORDER, np.linspace(-0.075, 0.075, len(MODEL_ORDER)), strict=True
        )
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 5.35), sharex=True, sharey=True)
    axes = axes.ravel()

    for tier, axis in enumerate(axes, start=1):
        axis.axvspan(2.72, 3.28, color="#F1F3F5", zorder=0)
        for method in METHODS:
            for model in MODEL_ORDER:
                subset = [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["method"] == method
                    and int(row["tier"]) == tier
                    and row["f1"]
                ]
                subset.sort(key=lambda row: positions[row["context"]])
                if not subset:
                    continue
                provisional = not as_bool(arms[(model, method)]["is_final"])
                offset = model_offsets[model]
                axis.errorbar(
                    [positions[row["context"]] + offset for row in subset],
                    [float(row["f1"]) for row in subset],
                    yerr=[float(row["f1_std"]) for row in subset],
                    color=COLORS[method],
                    linestyle="--" if provisional else "-",
                    linewidth=1.05,
                    alpha=0.82,
                    marker=MODEL_MARKERS[model],
                    markersize=4.1,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    capsize=1.6,
                    elinewidth=0.65,
                    zorder=3,
                )
        axis.set_title(
            f"({chr(96 + tier)}) Tier {tier}: {TIER_NAMES[tier]} ($n={TIER_QUESTIONS[tier]}$)",
            loc="left",
            pad=5,
        )
        axis.set_xlim(-0.2, 3.2)
        axis.set_ylim(-0.02, 1.06)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.grid(axis="y", color="#D8DDE2", linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=0.6)

    for axis in axes[2:]:
        axis.set_xticks(range(len(contexts)), ("100", "500", "1000", "Full"))
        axis.set_xlabel("Context size (reactions)")
    for axis in axes[::2]:
        axis.set_ylabel("Macro F1")

    method_handles = [
        Line2D([0], [0], color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method])
        for method in METHODS
    ]
    model_handles = []
    for model in MODEL_ORDER:
        provisional = any(not as_bool(arms[(model, method)]["is_final"]) for method in METHODS)
        model_handles.append(
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle="none",
                marker=MODEL_MARKERS[model],
                markersize=5.2,
                label=("*" if provisional else "")
                + next(row["model_label"] for row in rows if row["model"] == model),
            )
        )
    fig.suptitle("Scaling by tier across six models", y=0.995, fontsize=10, fontweight="bold")
    method_legend = fig.legend(
        handles=method_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.962),
        handlelength=2.4,
    )
    fig.add_artist(method_legend)
    fig.legend(
        handles=model_handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.918),
        columnspacing=1.05,
        handletextpad=0.35,
    )
    fig.text(
        0.5,
        0.008,
        "Small horizontal offsets improve visibility. Dashed curves and * labels are provisional "
        "(DeepSeek and GLM agentic arms).",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    fig.subplots_adjust(top=0.79, bottom=0.11, hspace=0.34, wspace=0.20)
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
    rows = read_csv(args.gold / "tier_scaling.csv")
    arms = {
        (row["model"], row["method"]): row
        for row in read_csv(args.gold / "arm_status.csv")
        if row["scope"] == "full_benchmark"
    }
    args.output.mkdir(parents=True, exist_ok=True)
    figure_files = []
    combined = combined_figure(rows, arms)
    save_figure(combined, args.output, "scaling_by_tier_all_models")
    plt.close(combined)
    figure_files.extend(["scaling_by_tier_all_models.pdf", "scaling_by_tier_all_models.png"])
    with PdfPages(args.output / "scaling_by_tier_individual_models.pdf") as multipage:
        for model in MODEL_ORDER:
            figure = model_figure(model, rows, arms)
            name = f"scaling_by_tier_{model.replace('.', '_')}"
            save_figure(figure, args.output, name)
            multipage.savefig(figure, bbox_inches="tight", pad_inches=0.02)
            figure_files.extend([f"{name}.pdf", f"{name}.png"])
            plt.close(figure)
    figure_files.append("scaling_by_tier_individual_models.pdf")
    manifest = {
        "gold_manifest_sha256": hashlib_sha256(args.gold / "source_manifest.json"),
        "figures": figure_files,
        "note": "A leading asterisk marks an unfinished main-benchmark arm.",
    }
    (args.output / "scaling_by_tier_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote six model figures and one multipage PDF to {args.output}")


def hashlib_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
