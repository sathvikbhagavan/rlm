# Oracle-predicate control

This control asks whether an RLM can execute reliably over a growing reaction
corpus once the chemical classification rule is supplied. It does not give the
model any reaction indices, answer counts, sampled support rows, or graph
solutions.

The fixed scope is Tier-3 Tasks 6, 10, and 23 and Tier-4 Tasks 13 and 14. These
contain 16 questions. Qwen3.5-397B and Claude Haiku 4.5 are evaluated at 100,
500, and all 122,456 reactions, with five repetitions. This gives:

- 150 tracked model jobs: 5 task configurations x 3 corpus sizes x 2 models x
  5 repetitions;
- 480 model trajectories: 16 questions x 3 corpus sizes x 2 models x 5
  repetitions;
- 15 deterministic jobs and 48 deterministic question evaluations, run once
  per task configuration and corpus size.

The Tier-3 helpers classify one reaction row. For Tier 4, the supplied helpers
only identify functional groups or protecting-group events. The RLM must still
scan the context, apply filters, construct exact-SMILES links, search or join
the graph, and format the result.

## Source of the predicates

These are the benchmark's existing ground-truth computations, not new chemical
definitions written for this control. Git history attributes the source
definitions to Sathvik Bhagavan's June/July benchmark commits:

| Control | Existing benchmark source reused exactly |
| --- | --- |
| Tier-3 Task 6 | Amide-coupling reaction SMARTS and RDKit `RunReactants` matching from `tier3/generate_hardcoded_ground_truth.py` and `tier3/task6_hardcoded_ground_truth.py` |
| Tier-3 Task 10 | `reaction_line_matches_mechanism` from `tier3/task10_mechanism_evaluator.py` |
| Tier-3 Task 23 | `reaction_creates_stereocenter_from_achiral` from `tier3/task23_stereocenter_evaluator.py` |
| Tier-4 Task 13 | `FUNCTIONAL_GROUP_SMARTS` and functional-group detection from `tier4/task13_fg_chain_graph.py` |
| Tier-4 Task 14 | `PROTECTING_GROUPS`, substructure detection, and stripped-scaffold logic from `tier4/task14_protecting_group_graph.py` |

The new `oracle_predicates.py` files are answer-free adapters: they expose the
same row- or molecule-level decisions without the frozen answer indices,
counts, sampled rows, or completed graph solutions. Full-corpus parity tests
guard against accidental changes during that refactoring.

The parity gate exposed one RDKit-version difference before inference: frozen
Task-10 Mitsunobu record 96808 matches under the dataset/ground-truth RDKit
2022.09.5 environment but RDKit 2025.09.6 rejects a generated six-valent
phosphorus intermediate during property sanitization. The evaluator now has a
narrow compatibility path for phosphorus `AtomValenceException` only; every
other sanitization failure remains rejected. Complete full-corpus parity must
still pass after this correction before an oracle model cell is released.

The project lead confirms that the original benchmark predicates received
chemist validation before this control was created. Before submission, retain a
short validation record naming the reviewer(s), date, reviewed predicate/file
version, and scope so the manuscript can state this with auditable provenance.

This origin is important to interpretation. The control intentionally asks,
"Can the RLM execute the benchmark reliably when given the benchmark's own
label function?" It does not independently establish that the label function
is universally complete beyond the benchmark definition; the prior chemist
review supports the benchmark's intended chemical validity.

Generate and check the two experiment definitions:

```bash
uv run --frozen python experiments/iclr2027/generate_oracle_predicate_campaign.py
uv run --frozen python experiments/iclr2027/generate_oracle_predicate_campaign.py --check
uv run --frozen rxnhaystack validate experiments/iclr2027/oracle-predicate-campaign.toml
uv run --frozen rxnhaystack validate experiments/iclr2027/oracle-executor-campaign.toml
```

Before any model request, run the complete dataset parity audit. It applies the
answer-free helpers to every relevant row or molecule and compares the results
with the frozen benchmark definitions:

```bash
RXNHAYSTACK_RUN_FULL_ORACLE_PARITY=1 \
uv run --frozen pytest -q tests/test_iclr_oracle_predicates.py \
  -k full_dataset_oracle_predicate_parity
```

Run the deterministic ceiling before model work. It costs no API money:

```bash
uv run --frozen rxnhaystack run experiments/iclr2027/oracle-executor-campaign.toml
```

Then release one inexpensive oracle cell per provider. Only after inspecting
its prompt, metrics, predicate hash, context hash, and artifacts should the
remaining model cells be run. Qwen shares the SwissAI account-wide request
quota with other active work, so its host cap must be assigned centrally.

The deterministic ceiling is not independent chemical validation: benchmark
ground truth is defined using the same predicates. Its purpose is to confirm
that exact execution reaches the benchmark ceiling. The RLM comparison is the
causal test of whether corpus-scale orchestration remains reliable after the
chemical abstraction is fixed.
