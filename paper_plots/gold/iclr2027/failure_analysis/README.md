# Failure-analysis data

This directory is rebuilt by `build_failure_analysis.py` and `plot_failure_analysis.py`.

The frozen five-model record contains 6,300 jobs. Excluding the 121 unresolved main-arm score recoveries leaves 6,179 trace-eligible jobs and 20,771 expected question trajectories. Terminal execution failures remain separate from scientific wrong answers.

`job_outcomes.csv` covers the trace-eligible job population. `trace_error_records.csv` contains the 2,229 corrected-task question outputs whose retained predictions support exact reconstruction of precision and recall. Its labels are observable set-error signatures, not inferred chemical causes. Outputs without sufficient retained detail remain unclassified rather than imputed.
