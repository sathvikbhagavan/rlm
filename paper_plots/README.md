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

## Corrected-score recovery

After a versioned ground-truth change, recover exact scores from preserved run
logs before rebuilding the tables:

```sh
WANDB_API_KEY=... uv run python \
  paper_plots/scripts/recover_corrected_scores.py \
  --snapshot-dir artifacts/control-room/shared \
  --artifact-tar /path/to/campaign-artifacts.tar \
  --phoenix-db /path/to/phoenix.db
uv run python paper_plots/scripts/build_gold_results.py
uv run python paper_plots/scripts/build_causal_controls.py
uv run python paper_plots/scripts/build_post_submission_queue.py
```

The recovery command reconstructs the original sampled context and validates the
historical score before accepting a corrected result. It recovers exact predictions
using the historical parser semantics where possible. When indices are absent, it
accepts an aggregate recovery only if every contingency table compatible with the
logged count and four-decimal precision/recall/F1 has the same corrected score.
Artifact tars are read in place without filesystem extraction; successful attempts
are selected by the W&B URL recorded in metadata, and duplicate backup copies must
be byte-identical.
When `--phoenix-db` is supplied, the database is opened in immutable read-only mode.
Execution windows from the selected artifact attempts are matched to exactly one
Phoenix task project and model before a retained answer is accepted. The recorded
prediction count and historical score must still validate. The database itself is
never copied into the repository; the recovery manifest records its checksum, size,
match diagnostics, and hashes of the extracted answers.
It writes an exact-rescore manifest and a separate prediction ledger under
`paper_plots/gold/`; neither contains credentials. Retrieved console logs stay
in the ignored local cache `artifacts/score-recovery/`. The recovery ledger is
replaced atomically and, by default, cannot be replaced by one with fewer
recovered runs. It can then be joined to the frozen experiment tables by
`build_post_submission_queue.py`, which records every still-pending run with
its experiment, arm, model, task, context, configured seed, and repetition.
Runs without sufficient retained prediction evidence remain explicitly queued
rather than being classified as exact corrected rescores.
