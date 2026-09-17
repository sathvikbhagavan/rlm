# Task 16 prospective decomposition

This control applies only to Tier-4 Task 16, the truncated-synthesis task. Tasks
17 and 17b are multi-constraint sequential-template searches: all transformation
steps are stated and they have no named target or withheld final reaction.

## Question being tested

Task 16 asks for four-reaction prefixes that can reach a target through one
withheld final reaction. Three paired prompts separate possible difficulties:

1. `name_only` supplies only the target name.
2. `structure_only` replaces the name with the exact target SMILES.
3. `structure_plus_class` adds a concise class for the withheld transformation.

The submitted legacy prompt is retained for reproducibility, but it is not the
name-only condition: it also includes a descriptive paragraph that discloses
route and final-step information.

All three new conditions use exactly the same context, ground truth, question
order and seeds. Every reaction whose product contains the exact canonical
target molecule is removed. This is stricter than the legacy task, which removed
only target-producing reactions used as terminals in accepted five-step routes.
Without the stricter rule, four targets remain visible as products elsewhere in
the full corpus and the structure condition can collapse into exact lookup.

This does not show that all original Task-16 reaction chains or scores were
wrong. It shows that the legacy prompt and corpus-exclusion protocol did not
support the strong interpretation of a clean name-only prospective test. Two
legacy descriptions also named a final transformation inconsistent with their
withheld reaction. Preserve those results as a reproducible legacy condition;
use the corrected control for prospective claims.

Code verifies exact-target removal and consistency with the frozen records. On
September 17, 2026, the project lead relayed the chemistry co-author's approval
of the three concise final-step classes. The reviewer accepted the first and
third labels and corrected the second from "spirocyclic amine" to "secondary
amine." Record the reviewer's name in the manuscript's internal validation
record before submission.

## Initial scope

The initial control contains three targets with distinct, auditable final steps:

- `pyrimidine_piperazine`: aryl carbon-nitrogen coupling;
- `lactam_dipeptide`: Boc deprotection of secondary amine;
- `benzamide_pyrazole`: aryl-halide borylation.

The approved labels and supporting cleaned-USPTO indices are frozen in
`tier4/task16_prospective.py`. The generated experiment records the approval;
do not replace these descriptions through an ad-hoc environment override in a
recorded run. None of the labels contains an exact precursor, reaction index,
or preceding route. Their intended role is specifically to disclose the broad
final-step class in that experimental condition.

The study uses Qwen3.5-397B-A17B and Claude Haiku 4.5, five repetitions, full
corpus RLM, and three questions per run:

```text
3 targets × 3 prompt conditions × 2 models × 5 repetitions = 90 trajectories
3 conditions × 2 models × 5 repetitions = 30 tracked jobs
```

The estimate is CHF 2.73, with a CHF 15 ceiling. Full-corpus Task-16 RLM requires
the tested Docker sandbox, so Jed and Kuma cannot run it under the archival
protocol.

## Prepare and inspect

```bash
uv run --frozen python experiments/iclr2027/generate_prospective_decomposition.py --check
uv run --frozen rxnhaystack validate experiments/iclr2027/prospective-decomposition.toml
uv run --frozen rxnhaystack plan experiments/iclr2027/prospective-decomposition.toml
```

The class-label gate is approved. Do not launch the remaining jobs until one
single-repetition pilot has produced valid metrics and
`task16-predictions.json`.

## Human-review candidates

Successful runs write parsed predictions, ground truth and false positives to
`task16-predictions.json` in each attempt directory. Build a deterministic pack:

```bash
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli export-task16-candidates \
  human_eval/local_state/task16-candidates-v1.json \
  artifacts/iclr2027-task16-prospective-decomposition-v1/runs/*/attempt-*/task16-predictions.json \
  --pack-id task16-prospective-v1 \
  --version 1.0.0 \
  --max-false-positives 150
```

The output is mode 0600 because it contains administrator-only model/run
metadata. `import-candidates` separates those fields into a private unblinding
map before anything is shown to annotators.
