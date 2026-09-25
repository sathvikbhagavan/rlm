# Post-submission score audit

This internal checklist is deliberately separate from manuscript prose. The
submission freeze describes the benchmark as the complete 100-question suite.
It uses exact corrected rescores wherever archived predictions permit them and
carries the preserved historical score only for unresolved rows, with an
explicit internal status marker.

## Required rescoring

- Use `rxnhaystack-human-1.6.0` for Tier-3 Tasks 6, 7, 10, 18, and 23 and
  Tier-4 Task 15.
- Audit every affected run listed in
  `paper_plots/gold/iclr2027/post_submission/pending_corrected_rescores.csv`.
- Replace a carried historical score only with an exact corrected rescore.
- Regenerate all gold tables, figures, manuscript numbers, and the submission
  manifest after the audit; compare every changed claim with the frozen PDF.

The current queue contains the 201 unresolved identities. The other 1,542 of
1,743 affected runs have exact corrected rescores. The builders preserve
`original_f1` for audit and mark unresolved carried values as
`historical_score_invalidated` until exact recovery or rerunning is complete.

## Items that do not invalidate model scores

- Tier-4 Task 12b was a human-extractor defect; model scores are unchanged.
- Tier-4 Task 13 retains the same answer, although the original wording was
  ambiguous about neutral carboxylic acids.
- Tier-3 Task 12 intentionally uses RDKit 2025.09.6.
- Tier-2 Task 3's alternative formula is algebraically equivalent.
- Tier-2 Tasks 4--5 are chemically surprising but valid under the dataset
  predicate.
- Tier-4 Task 16 concerns possible leakage in the human-facing interface, not
  model scoring.

## Submission-locked terminal results

- Qwen matched cardinality: 673 successful cells and 52 terminal failures out
  of 725; terminal failures score zero where the task score is eligible.
- DeepSeek V4 Flash standard CodeAct: 269 successful cells and 31 terminal
  failures out of 300.
- DeepSeek V4 Flash RLM at x1000: 150/150 successful, including the 15 Docker
  cells from Sathvik's result pack.
- GPT-5 mini RLM at x1000: 150/150 successful. The 135 Jed and 15 Docker jobs
  are merged canonically; the one dashboard row missing `macro_f1` is restored
  from checksum-verified W&B summary `hqwpwv2t`.
