"""Shared visual style for the ICLR 2027 paper figures."""

from __future__ import annotations

import matplotlib as mpl

# Discrete samples from Matplotlib's perceptually uniform Plasma map.  Keep
# method identity fixed across every benchmark figure.
METHOD_COLORS = {
    "llm": "#5302A3",
    "codeact": "#CB4679",
    "rlm": "#FCA636",
}
METHOD_LABELS = {"llm": "LLM", "codeact": "CodeAct", "rlm": "RLM"}
METHOD_MARKERS = {"llm": "o", "codeact": "s", "rlm": "D"}
HEATMAP_CMAP = "plasma"


def apply_paper_style() -> None:
    """Apply the common typography and export settings."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            # The ICLR template's ``times`` package resolves to Nimbus Roman on
            # the build host.  Naming that installed face explicitly prevents
            # Matplotlib from silently falling back to Liberation/DejaVu Serif.
            "font.serif": ["Nimbus Roman"],
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
