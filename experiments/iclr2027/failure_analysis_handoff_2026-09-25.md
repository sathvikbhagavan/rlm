# Failure-analysis handoff (2026-09-25)

This document defines the evidence that is safe to use for the paper's automated
failure analysis after the chemistry-aware human audit and corrected rescoring.

## Paper scope

- Paper models: Qwen3.5-397B-A17B, DeepSeek-V4-Flash-0731, Gemini 3.7 Flash,
  GPT-5 mini, and Claude Haiku 4.5.
- GLM-5.2 belongs to the six-model campaign inventory but is **outside the
  five-model paper analysis** because only its LLM arms are terminal.
- Canonical paper input: `paper_plots/gold/iclr2027/final_arm_records.csv`.
- Frozen scope: 6,150 jobs and **20,500 expected question trajectories**. This
  is the designed denominator of the included arms, not the number of successful
  trajectories. There are 19,687 successful trajectories; 174 terminal failed
  jobs remain in the denominator as zero-score outcomes.

## Mandatory exclusions

Before sampling or categorizing traces, exclude every `run_id` in:

`paper_plots/gold/iclr2027/post_submission/pending_corrected_rescores.csv`

Those 201 unique historical runs do not retain enough answer detail for exact
rescoring against the corrected oracle. Their `f1` values are carried only for
the submission freeze and must not be treated as corrected evidence in failure
analysis.

- 170 of the 201 runs use the five paper models; 31 use out-of-paper GLM-5.2.
- 121 occur in the final five-model paper arms.
- By affected task: Task 18 = 119, Task 23 = 45, Task 10 = 19, Task 7 = 18.
- By method: RLM = 142, plain LLM = 59. No CodeAct run remains unresolved.
- By model: GPT-5 mini = 61, Gemini = 32, Qwen = 32, GLM = 31,
  Claude = 30, DeepSeek = 15.
- Every row records experiment, arm, model, context, configured seed 42,
  repetition, source run URL, historical score, and recovery reason.

Rows marked `affected_without_historical_score` have no stale score to correct.
In the five-model final arms these are 51 terminal failures (48 CodeAct, 3 RLM).
They may be categorized as runtime/execution/terminal failures when the trace
supports that label, but they contain no answer suitable for chemical
wrong-answer categorization.

## Valid corrected evidence

The correction ledger covers 1,743 unique affected historical runs:

- 1,542 have an exact corrected rescore and are valid.
- 201 are the mandatory-exclusion queue above.
- Within the final five-model paper arms, 1,058 successful runs are labeled
  `corrected_exact_rescore`; use their current `f1`, not `original_f1`.
- Rows labeled `not_affected` are valid under the current oracle.

Recovered predictions and per-question corrected scores are retained in:

- `paper_plots/gold/source_packs/corrected_score_recovery_predictions.jsonl`
- `paper_plots/gold/corrected_score_recoveries.json`
- `paper_plots/gold/iclr2027/full_benchmark_records.csv`
- `paper_plots/gold/iclr2027/codeact_x1000_records.csv`
- `paper_plots/gold/iclr2027/rlm_x1000_records.csv`
- `paper_plots/gold/iclr2027/causal_controls/records.csv`

The exact extraction validated reconstructed predictions against the historical
per-question or macro score before applying the corrected oracle. Do not infer a
prediction from an aggregate score when no recovered prediction is present.

## Corrected benchmark semantics

The current canonical bundle is `rxnhaystack-human-1.6.0`. Six model-task
families required historical rescoring:

| Task | Correction |
| --- | --- |
| Tier 3 Task 6 | Acyl-chloride/primary-amine membership now requires carbon-substituted R-C(=O)Cl, excluding chloroformates and carbamoyl chlorides; 1,347 to 942 reactions. |
| Tier 3 Task 7 | Repeated transformations and connectivity-only product comparison correct multi-site and stereospecified-product omissions; all six subquestions were regenerated. |
| Tier 3 Task 10 | Wittig matching restricts the phosphorus-bound atom to carbon and excludes P=S thionations; 99 to 45 reactions. |
| Tier 3 Task 18 | Ring-system comparison ignores external substituent properties; 46,528 to 17,022 reactions. |
| Tier 3 Task 23 | Only uppercase absolute R/S centers satisfy the prompt; 1,456 to 1,410 reactions. |
| Tier 4 Task 15 | Quinoline and indole alternative-chain sets are exhaustive (299 and 241), and scoring uses `macro_reaction_f1`. |

Use current ground truth for any trace-level judgment. The versioned correction
contract is in `paper_plots/gold/ground_truth_corrections.json`; the full
scientific audit is in `human_eval/TIER3_TIER4_AUDIT.md`.

## Audited issues that do not invalidate model traces

- Tier 4 Task 12b: the old human extractor exposed eight sampled support hubs
  instead of the exhaustive 2,091-molecule full-dataset answer. Model runners
  were scored against their sampled contexts, so model scores are unaffected.
- Tier 4 Task 13: `carboxylic_acid` means neutral R-C(=O)-OH. The 550 stored
  chains regenerate exactly. The prompt was clarified, but historical model
  results remain an original-prompt condition; avoid calling carboxylate handling
  a model chemistry failure without noting that former ambiguity.
- Tier 3 Task 12: reaction 55015 is RDKit-version-sensitive. The benchmark
  explicitly pins RDKit 2025.09.6 for this bond-perception contract.
- Tier 2 Task 3: `max(product rings) - min(reactant rings)` is algebraically
  identical to the stated maximum over all pairwise deltas; this is not a defect.
- Tier 2 Task 4-5: formation of five aromatic rings is surprising but is a valid
  dataset/predicate result, not evidence of a broken oracle.
- Tier 4 Task 16: stored model chains/prefixes regenerate. The standalone human
  interface could expose withheld terminal reactions through its full-dataset
  browser, which limits the human-baseline interpretation but not model scoring.
- Theo's Task-13 discrepancy came from admitting alkyl fluorides and switching
  molecule identity between linked steps; it was a reviewer-code mismatch.

All other canonical Tier 3 and Tier 4 definitions passed the exhaustive audit
described in `human_eval/TIER3_TIER4_AUDIT.md`.

## Recommended analysis procedure

1. Start from `final_arm_records.csv`; do not reconstruct paper scope from the
   six-model campaign inventory.
2. Remove every queued `run_id` before selecting traces.
3. Separate terminal/runtime failures from successful-but-wrong answers.
4. For affected tasks, require `corrected_exact_rescore` or `not_affected` and
   use the current bundle to identify false positives and false negatives.
5. Preserve model, interface, context, tier, task, seed/repetition, and failure
   evidence for every assigned category. Permit multiple labels when supported.
6. Useful initial categories are access/retrieval, incomplete enumeration,
   answer parsing/format, incorrect predicate translation, chemistry-rule
   induction, RDKit/API misuse, reaction-chain identity/linking, heuristic
   shortcut, runtime/timeout/memory, and uncertain or insufficient evidence.
7. Report category counts with a denominator and retain uncategorized/uncertain
   traces rather than forcing a label.

## Human-audit provenance

Three reviewer exports contain 52 first submissions (51 non-abstained). Their
claims and adjudications are frozen in
`paper_plots/gold/iclr2027/human_validation/reviewer_feedback_audit.csv`.
Reviewer source archives are preserved under `human_eval/reviewer_code/`. Human
annotations motivated the corrections above, but post-reference revisions do
not replace independent first submissions in the human baseline.
