"""Shared visual style for the ICLR 2027 paper figures."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes

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


def add_upper_headroom(axis: Axes, scale: str, fraction: float = 0.12) -> None:
    """Keep markers and error bars clear of the top frame.

    Matplotlib's default margin is visually too tight in the paper's compact
    multi-panel figures.  Expand in transformed space for logarithmic axes so
    the padding remains proportional across orders of magnitude.
    """
    lower, upper = axis.get_ylim()
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return
    if scale == "log" and lower > 0:
        log_lower, log_upper = np.log(lower), np.log(upper)
        axis.set_ylim(lower, np.exp(log_upper + fraction * (log_upper - log_lower)))
    else:
        axis.set_ylim(lower, upper + fraction * (upper - lower))


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
