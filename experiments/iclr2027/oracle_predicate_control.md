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
