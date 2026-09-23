# Gold ICLR 2027 plotting results

Frozen from the shared experiment dashboard at `2026-09-22T19:03:40.570724+00:00`.

Gold means that the plotting input is frozen, auditable, and provenance-recorded. It does not mean every experiment arm is finished. `arm_status.csv` and the `arm_final` columns distinguish terminal arms from provisional ones.

The CSV files contain sanitized metrics sufficient to regenerate paper plots; bulky raw trajectories remain in their original experiment artifact stores.
All plotting aggregates score terminal failed jobs as zero. Running, stale, and pending jobs are excluded from the current score and keep their arm provisional.

## Main benchmark arms

| Model | LLM | CodeAct | RLM |
| --- | ---: | ---: | ---: |
| Qwen 3.5 | 300/300 final | 300/300 final | 449/450 final |
| DeepSeek V4 Flash | 300/300 final | 269/300 provisional | 423/450 provisional |
| GLM 5.2 | 300/300 final | 0/300 provisional | 344/450 provisional |
| Gemini 3.7 Flash | 300/300 final | 300/300 final | 442/450 final |
| GPT-5 mini | 300/300 final | 300/300 final | 450/450 final |
| Claude Haiku 4.5 | 300/300 final | 300/300 final | 450/450 final |

## Files

- `full_benchmark_records.csv`: every one of the 6,300 expected main-benchmark jobs.
- `codeact_x1000_records.csv`: the final DeepSeek and Gemini CodeAct x1000 extensions.
- `final_arm_records.csv`: records belonging to terminal arms.
- `provisional_arm_records.csv`: records belonging to unfinished arms.
- `arm_status.csv`: the finality decision used for legend asterisks.
- `tier_scaling.csv`: the faithful four-tier plotting aggregate.
- `tier_scaling_across_models.csv`: unweighted means and standard errors across terminal model arms; terminal failed trajectories contribute zero.
- `tier_efficiency_by_model.csv`: recorded cost, tokens, and wall time per successfully answered trajectory for each model. Failed jobs do not enter resource averages.
- `tier_efficiency_across_models.csv`: unweighted efficiency means and standard errors across terminal model arms. Cost averages include only paid Gemini, GPT-5-mini, and Claude models; free SwissAI access is excluded.
- `source_manifest.json`: source snapshot and file checksums.

## Causal controls

The `causal_controls/` directory freezes only completed arms used to separate
corpus scale, answer cardinality, predicate induction, and execution:

- all 725 successful GPT-5-mini matched-cardinality jobs;
- all 150 successful Qwen/Claude oracle-predicate jobs;
- the corresponding 150 successful ordinary-RLM jobs from the main benchmark;
- all 15 successful deterministic-executor jobs.

The unfinished Qwen matched-cardinality arm is deliberately excluded. The
record-level input, question-weighted five-repetition aggregates, and source
snapshot checksums are stored in `records.csv`, `aggregates.csv`, and
`source_manifest.json`, respectively.

Regenerate from the repository root:

```bash
uv run --frozen python paper_plots/scripts/build_gold_results.py
uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_gold_scaling_by_tier.py
uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_gold_efficiency_by_tier.py

uv run --frozen python paper_plots/scripts/build_causal_controls.py
uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_causal_controls.py
```
