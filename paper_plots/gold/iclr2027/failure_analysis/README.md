# Failure-analysis data

This directory is rebuilt by:

```bash
uv run python paper_plots/scripts/build_failure_analysis.py
python paper_plots/scripts/plot_failure_analysis.py
```

The analysis starts from `../final_arm_records.csv`, covering the five paper
models. Every run listed in `../post_submission/pending_corrected_rescores.csv`
is excluded. This leaves 6,029 jobs and 20,271 expected question trajectories.
Rows marked `corrected_exact_rescore` and `not_affected` are valid. The 51
`affected_without_historical_score` rows are retained only as execution
failures and never interpreted as chemical wrong answers.

`job_outcomes.csv` covers the entire valid job population. It separates exact,
partial, and zero scientific scores from execution failures.

`trace_error_records.csv` is narrower. It contains the 2,229 question outputs
from corrected Tier-3 Tasks 6, 7, 10, 18, and 23 for which retained predictions
support exact reconstruction of corrected precision and recall. Its labels are
observable set-error signatures:

- empty answer;
- omissions only (precision 1, recall below 1);
- extra selections only (recall 1, precision below 1);
- mixed omissions and extras.

These signatures do not assert a chemical cause. Chemistry-specific causes are
supported separately by direct inspection of the representative traces quoted
in the paper. Outputs lacking reconstructable precision and recall remain
unclassified instead of being imputed. All corrected scores use
`rxnhaystack-human-1.6.0`; GLM is outside the paper population.
