# Paper plots

This directory contains the plotting pipeline and exported aggregates used for
the RxnHaystack paper figures. The checked-in PDFs are the vector files used in
the manuscript; PNGs are included for convenient previewing.

From the repository root, regenerate every figure with:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r paper_plots/requirements.txt
python paper_plots/scripts/plot_results.py
```

The pinned plotting environment requires Python 3.11 or newer, consistent with
the repository's Python requirement.

The script reads `paper_plots/data/plot_data/` and writes ten figures to
`paper_plots/figures/`. The exports cover all 100 benchmark questions. Most
configurations contain five runs; full-corpus RLM on truncated synthesis routes
contains three.
