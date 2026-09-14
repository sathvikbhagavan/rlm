# Combining exports and analysis

```bash
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli analyze collaborator-*.zip \
  --bundle human_eval/generated/canonical-v3 --output analysis/human-v3
```

Outputs are `summary.json`, `item_metrics.csv`, and `disagreements.csv`. They report
completion by tier/category, exact match, set precision/recall/F1, abstentions,
time-limit outcomes, active/wall/offline time, tool use, confidence-ready item rows,
audit issues, prospective fractions, Cohen's kappa for two annotators, nominal
Krippendorff alpha for multiple annotators, and adjudication candidates.

Scoring uses the first submitted, non-abstained baseline answer only; post-reveal
revisions never replace it. Set-valued chain entries
are compared as complete comma-joined tuples. No missing item is treated as wrong.
Uncertain/cannot-assess prospective outcomes are separate categories. The nominal
alpha implementation uses pairwise observed disagreement over items with at least
two ratings and pooled marginal expected disagreement. Control and duplicate
analyses require administrator-side joining with `--unblinding PATH` (repeat for
multiple packs); the map is never added to annotator exports.
