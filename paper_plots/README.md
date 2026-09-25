# Paper plots

This directory contains the plotting pipeline and exported aggregates used for
the RxnHaystack paper figures. Final paper candidates live under
`paper_plots/figures/gold/` and are generated from the auditable tables in
`paper_plots/gold/iclr2027/`. The checked-in PDFs are vector exports; PNGs are
included for convenient previewing.

From the repository root, regenerate the gold tables and figures with the
commands in `paper_plots/gold/iclr2027/README.md`. For example:

```sh
uv run --frozen python paper_plots/scripts/build_gold_results.py
uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_main_results.py
```

`paper_plots/data/plot_data/` and `paper_plots/scripts/plot_results.py` preserve
the workshop-era input and plotting implementation for historical
reproducibility. Their superseded figure exports are intentionally not kept in
the active figure directory.
