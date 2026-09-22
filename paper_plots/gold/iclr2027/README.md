# Gold ICLR 2027 plotting results

Frozen from the shared experiment dashboard at `2026-09-22T15:05:59.351833+00:00`.

Gold means that the plotting input is frozen, auditable, and provenance-recorded. It does not mean every experiment arm is finished. `arm_status.csv` and the `arm_final` columns distinguish terminal arms from provisional ones.

The CSV files contain sanitized metrics sufficient to regenerate paper plots; bulky raw trajectories remain in their original experiment artifact stores.

## Main benchmark arms

| Model | LLM | CodeAct | RLM |
| --- | ---: | ---: | ---: |
| Qwen 3.5 | 300/300 final | 300/300 final | 449/450 final |
| DeepSeek V4 Flash | 300/300 final | 268/300 provisional | 408/450 provisional |
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
- `source_manifest.json`: source snapshot and file checksums.

Regenerate from the repository root:

```bash
uv run --frozen python paper_plots/scripts/build_gold_results.py
uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_gold_scaling_by_tier.py
```
