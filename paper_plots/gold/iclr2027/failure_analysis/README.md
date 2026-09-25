# Failure-analysis data

This directory is rebuilt by:

```bash
uv run python paper_plots/scripts/build_failure_analysis.py
python paper_plots/scripts/plot_failure_analysis.py
```

The analysis starts from `../final_arm_records.csv`, covering the five paper
models. Every run listed in `../post_submission/pending_corrected_rescores.csv`
is excluded. The current corrected-score queue leaves 4,971 jobs and 16,601
expected question trajectories. Rows marked `not_affected` are valid. Historical
scores marked `historical_score_invalidated` do not enter failure statistics or
trace-cause selection until exact rescoring completes.

`job_outcomes.csv` covers the entire valid job population. It separates exact,
partial, and zero scientific scores from execution failures.

`trace_error_records.csv` is the reconstructable set-error cohort. At the
current snapshot it is empty because all affected historical scores await exact
rescoring. Once corrected rows are restored, its labels remain observable
set-error signatures:

- empty answer;
- omissions only (precision 1, recall below 1);
- extra selections only (recall 1, precision below 1);
- mixed omissions and extras.

These signatures do not assert a chemical cause. Chemistry-specific causes are
supported separately by `reviewed_trace_causes.csv` and documented in
`deep_dive_review.md`. `diagnostic_objective_map.csv` connects those causes to
the access, execution, chemical-abstraction, relational-orchestration, and
route-reasoning capabilities isolated by the benchmark controls. Outputs
lacking sufficient retained evidence remain unclassified instead of being
imputed. Corrected scores use
`rxnhaystack-human-1.6.0`; GLM is outside the paper population.

The trace-review automation is run with:

```bash
uv run python paper_plots/scripts/build_failure_trace_review.py \
  --wandb-records paper_plots/data/iclr2027/records.csv \
  --output-dir /tmp/rxnhaystack-failure-review \
  --per-model-method 4
```

Its regular-expression signals retrieve evidence for human inspection; they are
never treated as scientific failure labels.
