#!/usr/bin/env python3
"""Generate the paper figures from the exported benchmark aggregates."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "plot_data"
OUT = ROOT / "figures"

METHODS = ("LLM", "CodeAct", "RLM")
COLORS = {
    "LLM": "#0072B2",
    "CodeAct": "#D55E00",
    "RLM": "#009E73",
}
MARKERS = {"LLM": "o", "CodeAct": "s", "RLM": "D"}
CONTEXT_POS = {"100": 0, "500": 1, "full": 2}
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
        "font.serif": ["Times New Roman"],
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


def normalize_context(value: object) -> str:
    text = str(value).strip()
    if text in {"-1", "-1.0", "full"}:
        return "full"
    return str(int(float(text)))


def load_overall() -> pd.DataFrame:
    frames = []
    for tier in range(1, 5):
        frame = pd.read_csv(DATA / f"tier{tier}" / "overall_agg.csv")
        frame["tier"] = tier
        frame["context"] = frame["context"].map(normalize_context)
        frame["question_count"] = TIER_QUESTIONS[tier]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pdf_metadata = {
        "Creator": "Rxn benchmark plotting pipeline",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        OUT / f"{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata=pdf_metadata,
    )
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_performance_overview(overall: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.55), sharex=True, sharey=True)
    axes = axes.ravel()

    for tier, ax in enumerate(axes, start=1):
        data = overall[overall["tier"] == tier]
        ax.axvspan(1.72, 2.28, color="#F1F3F5", zorder=0)
        for method in METHODS:
            subset = data[data["family"] == method].copy()
            subset["x"] = subset["context"].map(CONTEXT_POS)
            subset = subset.sort_values("x")
            ax.errorbar(
                subset["x"],
                subset["f1"],
                yerr=subset["f1_std"],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4.2,
                markeredgecolor="white",
                markeredgewidth=0.45,
                capsize=2.2,
                elinewidth=0.9,
                label=method,
                zorder=3,
            )
        ax.set_title(
            f"({chr(96 + tier)}) Tier {tier}: {TIER_NAMES[tier]} "
            f"($n={TIER_QUESTIONS[tier]}$)",
            loc="left",
            pad=5,
        )
        ax.set_xlim(-0.18, 2.2)
        ax.set_ylim(-0.02, 1.06)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color="#D8DDE2", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)

    for ax in axes[2:]:
        ax.set_xticks([0, 1, 2], ["100", "500", "Full"])
        ax.set_xlabel("Context size (reactions)")
    for ax in axes[::2]:
        ax.set_ylabel("Macro F1")

    handles = [
        Line2D(
            [0], [0], color=COLORS[m], marker=MARKERS[m], markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.45, label=m
        )
        for m in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.01), handlelength=2.2)
    fig.subplots_adjust(top=0.89, hspace=0.35, wspace=0.20)
    save(fig, "performance_overview")


SUBGROUPS = {
    3: [
        ("Bond-level", "subgroup_bond_level_agg.csv"),
        ("Scaffold", "subgroup_scaffold_agg.csv"),
        ("Reagent", "subgroup_reagent_agg.csv"),
        ("Stereochemistry", "subgroup_stereo_agg.csv"),
        ("Functional group", "subgroup_functional_group_agg.csv"),
        ("Mechanism", "subgroup_mechanism_agg.csv"),
    ],
    4: [
        ("Mechanical graph", "subgroup_mechanical_graph_agg.csv"),
        ("Chemical constraints", "subgroup_chemically_constrained_agg.csv"),
        ("Prospective / multi-constraint", "subgroup_prospective_multi_constraint_agg.csv"),
    ],
}

TASK_ROWS = {
    2: [
        ("2", "T2.1 Molecular-weight change (6Q)"),
        ("3", "T2.2 Ring-count change (5Q)"),
        ("4", "T2.3 Aromatic-ring formation (5Q)"),
        ("5", "T2.4 Combined weight + rings (4Q)"),
    ],
    3: [
        ("13", "T3.1 New nitrogen atom (1Q)"),
        ("14", "T3.2 C-N formed / C-O broken (1Q)"),
        ("15", "T3.3 Single C-C bond formation (1Q)"),
        ("17", "T3.4 Fused heterocycle (1Q)"),
        ("18", "T3.5 New ring system (1Q)"),
        ("20", "T3.6 Fused-ring construction (1Q)"),
        ("21", "T3.7 Transition-metal reagent (1Q)"),
        ("22", "T3.8 HATU / T3P reagent (1Q)"),
        ("23", "T3.9 New stereocenter (1Q)"),
        ("24", "T3.10 E-alkene in product (1Q)"),
        ("6", "T3.11-14 Amide couplings (4Q)"),
        ("7", "T3.15-19 Group transformations (5Q)"),
        ("8", "T3.20-21 Protecting groups (2Q)"),
        ("9", "T3.22-25 Named reactions (4Q)"),
        ("10", "T3.26-30 Mechanisms I (5Q)"),
        ("10b", "T3.31-35 Mechanisms II (5Q)"),
    ],
    4: [
        ("11", "T4.1-2 Fixed-length chains (2Q)"),
        ("12", "T4.3-4 Longest chains (2Q)"),
        ("12b", "T4.5 Hub molecules (1Q)"),
        ("13", "T4.6-9 Group-constrained chains (4Q)"),
        ("14", "T4.10-11 Protecting-group pairs (2Q)"),
        ("15", "T4.12-15 Ring-construction chains (4Q)"),
        ("16", "T4.16-25 Truncated routes (10Q)"),
        ("17", "T4.26-30 SMIRKS chains I (5Q)"),
        ("17b", "T4.31-35 SMIRKS chains II (5Q)"),
    ],
}

COLUMNS = [
    ("LLM", "100"), ("CodeAct", "100"), ("RLM", "100"),
    ("LLM", "500"), ("CodeAct", "500"), ("RLM", "500"),
    ("RLM", "full"),
]


def subgroup_matrix(tier: int) -> tuple[np.ndarray, list[str]]:
    rows = []
    labels = []
    for label, filename in SUBGROUPS[tier]:
        frame = pd.read_csv(DATA / f"tier{tier}" / filename)
        frame["context"] = frame["context"].map(normalize_context)
        lookup = frame.set_index(["family", "context"])["f1"]
        rows.append([lookup.get(column, np.nan) for column in COLUMNS])
        question_count = int(frame["question_count"].iloc[0])
        labels.append(f"{label} ({question_count}Q)")
    return np.asarray(rows), labels


def plot_capability_map() -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.15),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    image = None
    for ax, tier in zip(axes, (3, 4), strict=True):
        values, row_labels = subgroup_matrix(tier)
        image = ax.imshow(values, cmap="cividis", vmin=0, vmax=1, aspect="auto")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isnan(value):
                    continue
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.1,
                    color="white" if value < 0.52 else "#111111",
                )
        ax.set_yticks(range(len(row_labels)), row_labels)
        column_labels = [
            "LLM\n100", "CodeAct\n100", "RLM\n100",
            "LLM\n500", "CodeAct\n500", "RLM\n500", "RLM\nFull",
        ]
        ax.set_xticks(range(len(COLUMNS)), column_labels)
        ax.tick_params(axis="both", length=0)
        ax.xaxis.tick_top()
        ax.axvline(2.5, color="white", linewidth=2)
        ax.axvline(5.5, color="white", linewidth=2)
        ax.set_title(
            f"Tier {tier}: {TIER_NAMES[tier]}", loc="left", fontsize=9, pad=11
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.245, right=0.91, top=0.91, bottom=0.04, hspace=0.52)
    colorbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Macro F1", rotation=90, labelpad=6)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    save(fig, "capability_map")


def plot_task_map(tier: int) -> None:
    frame = pd.read_csv(DATA / f"tier{tier}" / "task_agg.csv")
    task_column = "task_number" if tier == 2 else "task_id"
    frame[task_column] = frame[task_column].astype(str)
    frame["context"] = frame["context"].map(normalize_context)
    lookup = frame.set_index([task_column, "family", "context"])["f1"]

    rows = TASK_ROWS[tier]
    values = np.asarray(
        [[lookup.get((task, *column), np.nan) for column in COLUMNS]
         for task, _ in rows]
    )
    height = {2: 2.15, 3: 6.25, 4: 3.45}[tier]
    fig, ax = plt.subplots(figsize=(7.0, height))
    image = ax.imshow(values, cmap="cividis", vmin=0, vmax=1, aspect="auto")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isnan(value):
                continue
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.9,
                color="white" if value < 0.52 else "#111111",
            )

    column_labels = [
        "LLM\n100", "CodeAct\n100", "RLM\n100",
        "LLM\n500", "CodeAct\n500", "RLM\n500", "RLM\nFull",
    ]
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    ax.set_xticks(range(len(COLUMNS)), column_labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0)
    ax.axvline(2.5, color="white", linewidth=2)
    ax.axvline(5.5, color="white", linewidth=2)
    ax.set_title(f"Tier {tier} task-level performance", loc="left", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.39, right=0.91, top=0.83, bottom=0.03)
    colorbar_ax = fig.add_axes([0.93, 0.14, 0.015, 0.62])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Macro F1", rotation=90, labelpad=5)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    save(fig, f"tier{tier}_task_map")


def plot_efficiency_frontier(overall: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    context_markers = {"100": "o", "500": "s", "full": "D"}
    for tier, ax in enumerate(flat_axes, start=1):
        data = overall[overall["tier"] == tier]
        for method in METHODS:
            subset = data[data["family"] == method].copy()
            subset["order"] = subset["context"].map(CONTEXT_POS)
            subset = subset.sort_values("order")
            ax.plot(subset["cost"], subset["f1"], color=COLORS[method],
                    alpha=0.45, linewidth=0.9, zorder=1)
            for _, row in subset.iterrows():
                ax.scatter(
                    row["cost"], row["f1"], s=27,
                    marker=context_markers[row["context"]],
                    color=COLORS[method], edgecolor="white", linewidth=0.5, zorder=3
                )
        ax.set_xscale("log")
        ax.set_xlim(0.0024, 0.065)
        ax.set_ylim(-0.02, 1.06)
        ax.grid(color="#D8DDE2", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"({chr(96 + tier)}) Tier {tier}", loc="left", pad=4)
    for ax in flat_axes[2:]:
        ax.set_xlabel("Mean cost per question (USD, log scale)")
    for ax in flat_axes[::2]:
        ax.set_ylabel("Macro F1")

    method_handles = [
        Line2D([0], [0], color=COLORS[m], marker="o", linestyle="-", label=m)
        for m in METHODS
    ]
    context_handles = [
        Line2D([0], [0], color="#555555", marker=context_markers[c],
               linestyle="none", label=("Full" if c == "full" else c))
        for c in ("100", "500", "full")
    ]
    fig.legend(handles=method_handles + context_handles, loc="upper center", ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 1.01), handlelength=1.5)
    fig.subplots_adjust(top=0.88, hspace=0.31, wspace=0.18)
    save(fig, "efficiency_frontier")


def plot_metric_curve(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    frame = frame.copy()
    frame["context"] = frame["context"].map(normalize_context)
    ax.axvspan(1.72, 2.28, color="#F1F3F5", zorder=0)
    for method in METHODS:
        subset = frame[frame["family"] == method].copy()
        subset["x"] = subset["context"].map(CONTEXT_POS)
        subset = subset.sort_values("x")
        ax.errorbar(
            subset["x"], subset["f1"], yerr=subset["f1_std"],
            color=COLORS[method], marker=MARKERS[method], markersize=3.8,
            markeredgecolor="white", markeredgewidth=0.4, capsize=2,
            elinewidth=0.8, zorder=3,
        )
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlim(-0.18, 2.2)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xticks([0, 1, 2], ["100", "500", "Full"])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color="#D8DDE2", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def plot_cost_curve(ax: plt.Axes, frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["context"] = frame["context"].map(normalize_context)
    ax.axvspan(1.72, 2.28, color="#F1F3F5", zorder=0)
    for method in METHODS:
        subset = frame[frame["family"] == method].copy()
        subset["x"] = subset["context"].map(CONTEXT_POS)
        subset = subset.sort_values("x")
        ax.plot(
            subset["x"], subset["cost"], color=COLORS[method],
            marker=MARKERS[method], markersize=3.8, markeredgecolor="white",
            markeredgewidth=0.4,
        )
    ax.set_title("Mean cost per question", loc="left", pad=4)
    ax.set_xlim(-0.18, 2.2)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 1, 2], ["100", "500", "Full"])
    ax.grid(axis="y", color="#D8DDE2", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    ax.set_ylabel("USD")


def detail_frames(tier: int, overall: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    panels = [("Overall", overall[overall["tier"] == tier])]
    if tier == 2:
        frame = pd.read_csv(DATA / "tier2" / "task_agg.csv")
        labels = {
            2: "Molecular weight", 3: "Ring count",
            4: "Aromatic rings", 5: "Combined criteria",
        }
        panels.extend(
            (label, frame[frame["task_number"] == task])
            for task, label in labels.items()
        )
    elif tier in (3, 4):
        for label, filename in SUBGROUPS[tier]:
            panels.append((label, pd.read_csv(DATA / f"tier{tier}" / filename)))
    return panels


def plot_tier_detail(tier: int, overall: pd.DataFrame) -> None:
    layouts = {
        1: (1, 2, (7.0, 2.7)),
        2: (2, 3, (7.0, 4.15)),
        3: (2, 4, (7.0, 4.15)),
        4: (2, 3, (7.0, 4.15)),
    }
    rows, cols, size = layouts[tier]
    fig, axes = plt.subplots(rows, cols, figsize=size)
    flat_axes = np.atleast_1d(axes).ravel()
    panels = detail_frames(tier, overall)
    for ax, (title, frame) in zip(flat_axes, panels, strict=False):
        plot_metric_curve(ax, frame, title)
    cost_ax = flat_axes[len(panels)]
    plot_cost_curve(cost_ax, overall[overall["tier"] == tier])
    for ax in flat_axes[len(panels) + 1:]:
        ax.axis("off")
    for i, ax in enumerate(flat_axes):
        if not ax.axison:
            continue
        if i % cols == 0 and i < len(panels):
            ax.set_ylabel("Macro F1")
        ax.set_xlabel("Context size")

    handles = [
        Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], label=m,
               markersize=4, markeredgecolor="white", markeredgewidth=0.4)
        for m in METHODS
    ]
    legend_y = 1.005 if tier != 1 else 0.995
    title_y = 0.94 if tier != 1 else 0.87
    top = 0.82 if tier != 1 else 0.72
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, legend_y), handlelength=1.8)
    fig.suptitle(f"Tier {tier}: {TIER_NAMES[tier]}", y=title_y, fontsize=10)
    fig.subplots_adjust(top=top, hspace=0.52, wspace=0.32)
    save(fig, f"tier{tier}_detail")


def main() -> None:
    overall = load_overall()
    plot_performance_overview(overall)
    plot_capability_map()
    plot_efficiency_frontier(overall)
    for tier in (2, 3, 4):
        plot_task_map(tier)
    for tier in range(1, 5):
        plot_tier_detail(tier, overall)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
