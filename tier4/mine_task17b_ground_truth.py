"""Mine 2- or 3-step SMIRKS sequential chains for tier4 task17b."""

from __future__ import annotations

import json
from pathlib import Path

from rlm.codeact_helpers import load_lines
from task17b_smirks_sequential_graph import (
    BUILTIN_QUESTIONS,
    build_line_mol_cache,
    classify_question_steps,
    enumerate_chains,
    parse_records_from_lines,
    verify_chain,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
TIER4_DIR = Path(__file__).resolve().parent
CHAINS_JSON_PATH = TIER4_DIR / "task17b_hardcoded_chains.json"


def audit_chains(
    records,
    lines_by_index,
    chains,
    *,
    question_id: str,
    step_hits,
    chain_length: int,
    n: int = 5,
) -> None:
    print(f"\n--- Hand-check {min(n, len(chains))} chains for {question_id} ---")
    ranked = sorted(chains, key=lambda c: c.reaction_indices)
    step_hit_sets = [set(h.keys()) for h in step_hits]
    for chain in ranked[:n]:
        ok, reason = verify_chain(
            chain.reaction_indices,
            question_id,
            records,
            lines_by_index,
            step_hits=step_hit_sets,
            chain_length=chain_length,
        )
        indices = ",".join(str(i) for i in chain.reaction_indices)
        print(f"  {indices} verify={ok} ({reason})")
        for label, idx in zip(
            [f"r{i}" for i in range(chain_length)],
            chain.reaction_indices,
        ):
            print(f"    {label}: {records[idx].raw[:120]}...")


def mine_question(spec, lines, records, lines_by_index, mol_cache) -> dict[str, object]:
    print(f"\n=== {spec.question_id} (L={spec.chain_length}) ===", flush=True)
    step_hits, templates = classify_question_steps(lines, spec, mol_cache=mol_cache)
    for i, template in enumerate(templates):
        print(
            f"Step{i + 1} [{template.name}] reactions={len(step_hits[i])}",
            flush=True,
        )

    step_sets = [set(h.keys()) for h in step_hits]
    chains = enumerate_chains(
        records,
        step_sets,
        question_id=spec.question_id,
        chain_length=spec.chain_length,
        require_persistence=True,
    )
    chains_no_persist = enumerate_chains(
        records,
        step_sets,
        question_id=spec.question_id,
        chain_length=spec.chain_length,
        require_persistence=False,
    )
    print(
        f"Chains with persistence: {len(chains)} | without: {len(chains_no_persist)}",
        flush=True,
    )

    if chains:
        audit_chains(
            records,
            lines_by_index,
            chains,
            question_id=spec.question_id,
            step_hits=step_hits,
            chain_length=spec.chain_length,
        )

    support = sorted({idx for c in chains for idx in c.reaction_indices})
    return {
        "question_id": spec.question_id,
        "label": spec.label,
        "path_length": spec.chain_length,
        "chain_count": len(chains),
        "chain_count_no_persistence": len(chains_no_persist),
        "support_index_count": len(support),
        "example_chain": list(chains[0].reaction_indices) if chains else [],
        "chains": [list(c.reaction_indices) for c in chains],
        "chain_details": [
            {
                "reaction_indices": list(c.reaction_indices),
                "spine_smiles": list(c.spine_smiles),
            }
            for c in chains
        ],
        "step_reaction_counts": [len(h) for h in step_hits],
        "smirks": __import__(
            "task17b_smirks_sequential_graph", fromlist=["smirks_documentation"]
        ).smirks_documentation(spec),
    }


def main() -> None:
    lines = load_lines(DATASET_PATH)
    records = parse_records_from_lines(lines)
    lines_by_index = {int(line.split(" ", 1)[0]): line for line in lines}
    print(f"Loaded {len(records)} reactions", flush=True)
    mol_cache = build_line_mol_cache(lines)
    print("Built mol cache", flush=True)

    mined: dict[str, dict[str, object]] = {}
    for spec in BUILTIN_QUESTIONS:
        mined[spec.question_id] = mine_question(
            spec, lines, records, lines_by_index, mol_cache
        )

    with CHAINS_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(mined, handle, indent=2)
    print(f"\nWrote {CHAINS_JSON_PATH}")


if __name__ == "__main__":
    main()
