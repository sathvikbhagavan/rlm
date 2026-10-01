# Frozen human-baseline results

These are derived, paper-facing results from three checksum-verified anonymous
exports, rescored against `rxnhaystack-human-1.6.0` (`canonical-v7`). Raw exports
remain immutable outside the repository. `input_manifest.json` records their hashes,
original bundle versions, the corrected protected-answer hash, and every generated
file hash.

Accuracy uses the first submitted, non-abstained answer. Later revisions made after
the reference answer was revealed cannot replace the baseline response. Set answers
are compared semantically, independent of order. Exact match is intentionally strict;
precision, recall, and F1 retain credit for near-complete large sets. Missing assigned
questions are not scored as wrong.

Active browser time, browser wall time, and self-reported offline/tool time are shown
separately. They are not summed because offline work can overlap an open browser
session. Agreement is reported only on answered overlap: 11 reviewer-pairs across
the three pairwise comparisons, covering fewer unique items. The small, deliberately
stratified overlap makes kappa descriptive rather than a population estimate.

`human_model_tier_comparison.csv` is also descriptive. Human rows are assigned
full-dataset question answers, whereas model rows are successful sampled-context
task runs. Historical model scores invalidated by corrected ground truth are excluded,
and each model row reports the resulting score coverage. This table must not be
described as a randomized head-to-head experiment.

Regenerate from the repository root with:

```bash
PYTHONPATH=. uv run --frozen --with-requirements human_eval/requirements.txt \
  python human_eval/tools/freeze_human_results.py \
  --export reviewer-1=/path/to/reviewer-1.zip \
  --export reviewer-2=/path/to/reviewer-2.zip \
  --export reviewer-3=/path/to/reviewer-3.zip \
  --bundle human_eval/generated/canonical-v7 \
  --model-gold paper_plots/gold/iclr2027 \
  --output paper_plots/gold/iclr2027/human_validation
```
