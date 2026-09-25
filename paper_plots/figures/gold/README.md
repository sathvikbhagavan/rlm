# Gold paper figures

This directory contains the paper-plot candidates generated from the frozen,
auditable result tables in [`paper_plots/gold/iclr2027`](../../gold/iclr2027/README.md).
The current snapshot is dated **2026-09-22**. It contains final results where
available and explicitly marks provisional arms; it is not a claim that every
planned experiment has finished.

Do not edit the CSV tables or figure files manually. Update the source results,
rerun the gold-data builder, and then regenerate the plots.

## Figure inventory

| Figure | Purpose | Plot code | Main input |
| --- | --- | --- | --- |
| [`performance_overview.pdf`](performance_overview.pdf) | Main GPT-5 mini scaling figure with repetition-level variation | [`plot_main_results.py`](../../scripts/plot_main_results.py) | [`tier_scaling.csv`](../../gold/iclr2027/tier_scaling.csv) |
| [`efficiency_frontier.pdf`](efficiency_frontier.pdf) | Cross-model macro-F1 against recorded USD cost per question trajectory for the three complete paid-API model arms | [`plot_main_results.py`](../../scripts/plot_main_results.py) | [`full_benchmark_records.csv`](../../gold/iclr2027/full_benchmark_records.csv) |
| [`efficiency_frontier_by_model.pdf`](efficiency_frontier_by_model.pdf) | The same accuracy--cost frontier separated into Gemini, GPT-5 mini, and Claude rows | [`plot_main_results.py`](../../scripts/plot_main_results.py) | [`full_benchmark_records.csv`](../../gold/iclr2027/full_benchmark_records.csv) |
| [`scaling_by_tier_all_models.pdf`](scaling_by_tier_all_models.pdf) | Main performance summary: unweighted mean and SEM across terminal model arms for LLM, CodeAct, and RLM | [`plot_gold_scaling_by_tier.py`](../../scripts/plot_gold_scaling_by_tier.py) | [`tier_scaling_across_models.csv`](../../gold/iclr2027/tier_scaling_across_models.csv) |
| [`rlm_tier_scaling_gold.pdf`](rlm_tier_scaling_gold.pdf) | RLM-only scaling for all six models; closest gold replacement for the earlier `rlm_tier_scaling.pdf` | [`plot_gold_scaling_by_tier.py`](../../scripts/plot_gold_scaling_by_tier.py) | [`tier_scaling.csv`](../../gold/iclr2027/tier_scaling.csv) |
| [`scaling_by_tier_all_model_curves.pdf`](scaling_by_tier_all_model_curves.pdf) | Diagnostic overlay of every model/interface curve; useful for checking heterogeneity, not recommended as the main paper figure | [`plot_gold_scaling_by_tier.py`](../../scripts/plot_gold_scaling_by_tier.py) | [`tier_scaling.csv`](../../gold/iclr2027/tier_scaling.csv) |
| [`scaling_by_tier_individual_models.pdf`](scaling_by_tier_individual_models.pdf) | Multipage diagnostic with one four-tier figure per model | [`plot_gold_scaling_by_tier.py`](../../scripts/plot_gold_scaling_by_tier.py) | [`tier_scaling.csv`](../../gold/iclr2027/tier_scaling.csv) |
| [`cost_by_tier_across_models.pdf`](cost_by_tier_across_models.pdf) | Recorded billed cost per successful trajectory, averaged over paid models only | [`plot_gold_efficiency_by_tier.py`](../../scripts/plot_gold_efficiency_by_tier.py) | [`tier_efficiency_across_models.csv`](../../gold/iclr2027/tier_efficiency_across_models.csv) |
| [`tokens_by_tier_across_models.pdf`](tokens_by_tier_across_models.pdf) | Recorded tokens per successful trajectory | [`plot_gold_efficiency_by_tier.py`](../../scripts/plot_gold_efficiency_by_tier.py) | [`tier_efficiency_across_models.csv`](../../gold/iclr2027/tier_efficiency_across_models.csv) |
| [`wall_time_by_tier_across_models.pdf`](wall_time_by_tier_across_models.pdf) | Wall time per successful trajectory; failed jobs are excluded | [`plot_gold_efficiency_by_tier.py`](../../scripts/plot_gold_efficiency_by_tier.py) | [`tier_efficiency_across_models.csv`](../../gold/iclr2027/tier_efficiency_across_models.csv) |
| `scaling_by_tier_<model>.pdf` | Individual Qwen, DeepSeek, GLM, Gemini, GPT, and Claude diagnostics | [`plot_gold_scaling_by_tier.py`](../../scripts/plot_gold_scaling_by_tier.py) | [`tier_scaling.csv`](../../gold/iclr2027/tier_scaling.csv) |
| `model_appendix/<model>_core_metrics.pdf` | Four horizontal tier panels for F1, tokens, cost, and wall time | [`plot_model_appendix.py`](../../scripts/plot_model_appendix.py) | `model_appendix/<model>/tier_metrics.csv` |
| `model_appendix/<model>_resource_diagnostics.pdf` | Four horizontal tier panels for calls, latency, tool time, and peak memory | [`plot_model_appendix.py`](../../scripts/plot_model_appendix.py) | `model_appendix/<model>/tier_metrics.csv` |
| `model_appendix/<model>_task_heatmap.pdf` | All 30 task configurations across the seven benchmark arms | [`plot_model_appendix.py`](../../scripts/plot_model_appendix.py) | `model_appendix/<model>/task_f1.csv` |

PNG previews accompany every PDF. The PDFs are the vector versions intended
for manuscript use. The JSON files in this directory list the generated files
and link them to the gold-data manifest.

Paper plots use discrete samples from the perceptually uniform Plasma palette.
LLM, CodeAct, and RLM retain the same purple, magenta, and orange identities in
every per-model profile; task heatmaps use the continuous Plasma scale.

## Aggregation rules

The rules below are deliberate and should remain consistent across future
figures.

### Performance

- A successful job contributes its measured task score.
- A terminal failed job is a wrong answer and contributes zero.
- Running, stale, and pending jobs are not assigned zero. They are excluded
  until resolved, and the affected arm is marked provisional with `*` and/or a
  hollow marker.
- Tier scores are question-weighted within a model. Cross-model summaries then
  give every eligible model equal weight.
- Cross-model error bars are standard errors across model-level means. The
  individual-model plots use variation across resolved repetitions.
- Tier-4 Task 15 uses reaction F1; all other tasks use macro-F1.

### Cost, tokens, and time

- Cost averages include only models for which we paid: Gemini 3.7 Flash,
  GPT-5 mini, and Claude Haiku 4.5. Free SwissAI Qwen, DeepSeek, and GLM calls
  are excluded rather than entered as zero.
- The CodeAct `x=1000` cost point currently has only Gemini (`n=1`). Add GPT
  when its result pack becomes available.
- Token and wall-time summaries use successful jobs only. Failed jobs enter
  performance as zero but do not enter resource numerators or denominators.
- Provider token and cost metadata were not preserved for failed attempts.
  Resource plots therefore describe successfully answered trajectories and
  are not total campaign-spend reports.
- Wall time is end-to-end process time, including model calls and local tool
  execution, divided by the number of successfully answered trajectories.

## Current interpretation

The gold RLM averages over terminal models currently show:

| Tier | 100 reactions | Full corpus |
| --- | ---: | ---: |
| Tier 1 | 0.910 | 0.915 |
| Tier 2 | 0.915 | 0.922 |
| Tier 3 | 0.716 | 0.568 |
| Tier 4 | 0.616 | 0.318 |

This supports the claim that RLM performance is stable with corpus scale for
lookup and aggregation, while reaction-level and relational reasoning degrade
at full-corpus scale. The defensible wording is **high and stable**, not
universally **near-perfect**, because absolute performance remains
model-dependent.

DeepSeek CodeAct/RLM and GLM CodeAct/RLM remain provisional in this snapshot.
Always consult [`arm_status.csv`](../../gold/iclr2027/arm_status.csv) rather than
copying these counts into new prose.

## Regenerating everything

Run from the repository root:

```bash
uv run --frozen python paper_plots/scripts/build_gold_results.py

uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_gold_scaling_by_tier.py

uv run --with-requirements paper_plots/requirements.txt \
  python paper_plots/scripts/plot_gold_efficiency_by_tier.py
```

The builder reads the corrected shared dashboard snapshots from
`artifacts/control-room/shared/` and the checked-in Gemini CodeAct `x=1000`
result pack. [`source_manifest.json`](../../gold/iclr2027/source_manifest.json)
records the source snapshots, campaign hashes, input-pack checksum, repository
commit, and generated-table checksums.

Before freezing a new snapshot, commit changes to the builder and plot scripts
first. This ensures `source_manifest.json` records the exact code revision that
created the tables.

## Continuing with new results

When another result pack or dashboard update arrives:

1. Preserve the original result pack unchanged and record its checksum.
2. Extend [`build_gold_results.py`](../../scripts/build_gold_results.py) with a
   strict importer; validate model, method, context, expected run IDs,
   cardinality, and score availability.
3. Regenerate the gold tables and inspect `arm_status.csv`, coverage, and the
   failed/unresolved counts.
4. Regenerate every figure, not only the figure expected to change.
5. Visually inspect PDF/PNG outputs and confirm that `n`, `*`, and hollow-marker
   annotations still match the tables.
6. Run the focused and complete checks:

```bash
uv run --frozen ruff check \
  paper_plots/scripts/build_gold_results.py \
  paper_plots/scripts/plot_gold_scaling_by_tier.py \
  paper_plots/scripts/plot_gold_efficiency_by_tier.py \
  tests/test_gold_plot_data.py

uv run --frozen pytest -q tests/test_gold_plot_data.py
uv run --frozen pytest -q
```

The aggregate construction and failure/resource accounting tests live in
[`tests/test_gold_plot_data.py`](../../../tests/test_gold_plot_data.py).
