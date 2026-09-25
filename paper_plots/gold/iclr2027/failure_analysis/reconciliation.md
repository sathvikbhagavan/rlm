# Reconciliation with the detailed appendix analysis

The quantitative and qualitative analyses answer different questions.

- `job_outcomes.csv` covers every valid final-arm job and separates scientific
  scores from execution failures.
- `trace_error_records.csv` quantifies observable answer-set signatures where
  corrected precision and recall can be reconstructed exactly.
- `manual_trace_audit.csv` tests whether the chemically informative mechanisms
  described in the earlier appendix recur in the corrected five-model data.

The manual audit is purposive, not a prevalence estimate. It includes all five
paper models, CodeAct and RLM, corrected Tier-3 Tasks 6 and 18, corrected Tier-4
Task 15, and unaffected Tier-4 Task 16. Every row was checked against the
executed code or transcript and the final-arm correction status. The audit
confirms recurrence of five earlier observations:

1. invalid or unavailable RDKit operations can derail otherwise reasonable
   decomposition;
2. independent reactant and product flags do not establish that the same atoms
   undergo the requested transformation;
3. generated SMARTS can be narrower or broader than the benchmark definition;
4. models substitute coarse string or molecular-property proxies for exact
   structural identity;
5. route searches can repeatedly apply a target-agnostic terminal-feature test
   until the execution budget is exhausted.

The corrected quantitative analysis supersedes historical numerical scores in
the older appendix examples for Tasks 6, 7, 10, 18, 23, and 15. The older code
excerpts remain useful illustrations of mechanisms, but they are not used to
estimate frequency or support corrected score claims. Task 16 examples are
unaffected by the score corrections; its separate concern is possible leakage
through the human annotation interface, not model scoring.
