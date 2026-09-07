# Plot data

This directory contains the aggregate and per-run exports supplied for the
paper figures. Regenerate the figures from the repository root with:

```sh
python paper_plots/scripts/plot_results.py
```

The exported results cover all 100 questions: 10 in Tier 1, 20 in Tier 2, and
35 each in Tiers 3 and 4. Most configurations contain five runs; the
full-corpus RLM export for the truncated synthesis-route task contains three.
