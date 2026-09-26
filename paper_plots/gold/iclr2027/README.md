# Gold ICLR 2027 plotting results

Frozen from the shared experiment dashboard at `2026-09-26T00:47:16.402100+00:00`.

Gold means that the plotting input is frozen, auditable, and provenance-recorded. It does not mean every experiment arm is finished. `arm_status.csv` and the `arm_final` columns distinguish terminal arms from provisional ones.

The CSV files contain sanitized metrics sufficient to regenerate paper plots; bulky raw trajectories remain in their original experiment artifact stores.
All plotting aggregates score terminal failed jobs as zero. Running, stale, and pending jobs are excluded from the current score and keep their arm provisional. Exact corrected rescores are used wherever recoverable. For remaining rows in the internal post-submission rescore queue, the last available historical score is carried forward under an explicit status marker so terminal trajectory denominators remain complete; resource measurements remain unchanged.

## Main benchmark arms

| Model | LLM | CodeAct | RLM |
| --- | ---: | ---: | ---: |
| Qwen 3.5 | 300/300 final | 300/300 final | 449/450 final |
| DeepSeek V4 Flash | 300/300 final | 269/300 final | 450/450 final |
| GLM 5.2 | 300/300 final | 0/300 provisional | 384/450 provisional |
| Gemini 3.7 Flash | 300/300 final | 300/300 final | 447/450 final |
| GPT-5 mini | 300/300 final | 300/300 final | 450/450 final |
| Claude Haiku 4.5 | 300/300 final | 300/300 final | 450/450 final |

## Files

- `full_benchmark_records.csv`: every one of the 6,300 expected main-benchmark jobs.
- `codeact_x1000_records.csv`: the final Qwen, DeepSeek, Gemini, and GPT-5-mini CodeAct x1000 extensions.
- `rlm_x1000_records.csv`: terminal RLM x1000 extensions available at the freeze time.
- `final_arm_records.csv`: records in the five-model paper scope belonging to terminal arms.
- `provisional_arm_records.csv`: unfinished arms and completed campaign records outside the five-model paper scope (currently GLM LLM).
- `arm_status.csv`: the finality decision used for legend asterisks.
- `tier_scaling.csv`: the faithful four-tier plotting aggregate.
- `tier_scaling_across_models.csv`: unweighted means and standard errors across the five paper models; terminal failed trajectories contribute zero.
- `tier_efficiency_by_model.csv`: recorded cost, tokens, and wall time per successfully answered trajectory for each model. Failed jobs do not enter resource averages.
- `tier_efficiency_across_models.csv`: unweighted efficiency means and standard errors across terminal model arms. Cost averages include only paid Gemini, GPT-5-mini, and Claude models; free SwissAI access is excluded.
- `capability_split/`: checked per-model and across-model full-corpus RLM summaries for the six operation-specific task groups.
- `efficiency_appendix/`: checked calls, tokens, recorded latency, tool time, process wall time, and peak-memory summaries for terminal arms.
- `source_manifest.json`: source snapshot and file checksums.

## Causal controls

The `causal_controls/` directory freezes the terminal GPT-5-mini and Qwen matched-cardinality arms, Qwen/Claude chemistry-rule controls and their ordinary RLM counterparts, and the deterministic executor ceiling. Its record tables and source manifest preserve the aggregation rules and contributing snapshots. Qwen has 673 successful and 52 terminal failed cells; failures contribute zero.
Exact corrected rescores are included and labeled `corrected_exact_rescore`. Pending corrected control scores follow the same submission-freeze carry-forward policy as the main benchmark and remain listed in the internal queue.

## Prospective-route control

The `prospective_decomposition/` directory freezes the completed Task-16 control: two models, three target-information conditions, five repetitions, and three targets per run (30 jobs and 90 trajectories). Aggregate, per-run, and per-target records are retained.

Regenerate from the repository root:

```bash
uv run --frozen python paper_plots/scripts/build_gold_results.py
uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_gold_scaling_by_tier.py
uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_main_results.py
uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_gold_efficiency_by_tier.py
uv run --frozen python paper_plots/scripts/build_capability_split.py
uv run --frozen python paper_plots/scripts/build_efficiency_appendix.py
uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_efficiency_appendix.py
uv run --frozen python paper_plots/scripts/build_causal_controls.py
uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_control_appendix.py
uv run --frozen python paper_plots/scripts/build_prospective_decomposition.py
uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_prospective_appendix.py
```
