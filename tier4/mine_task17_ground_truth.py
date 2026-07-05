"""Mine 2-step SMIRKS sequential chains for tier4 task17."""

from __future__ import annotations

import json
from pathlib import Path

from rlm.codeact_helpers import load_lines
from task17_smirks_sequential_graph import (
    BUILTIN_QUESTIONS,
    CHAIN_LENGTH,
    build_line_mol_cache,
    classify_question_steps,
    enumerate_chains,
    parse_records_from_lines,
    smirks_documentation,
    verify_chain,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
TIER4_DIR = Path(__file__).resolve().parent
CHAINS_JSON_PATH = TIER4_DIR / "task17_hardcoded_chains.json"


def audit_chains(
    records,
    lines_by_index,
    chains,
    *,
    question_id: str,
    s1_hits,
    s2_hits,
    n: int = 5,
) -> None:
    print(f"\n--- Hand-check {min(n, len(chains))} chains for {question_id} ---")
    ranked = sorted(chains, key=lambda c: c.reaction_indices)
    for chain in ranked[:n]:
        ok, reason = verify_chain(
            chain.reaction_indices,
            question_id,
            records,
            lines_by_index,
            s1_hits=s1_hits,
            s2_hits=s2_hits,
        )
        r1, r2 = chain.reaction_indices
        print(f"  {r1},{r2} verify={ok} ({reason})")
        for label, idx in [("r1", r1), ("r2", r2)]:
            print(f"    {label}: {records[idx].raw[:120]}...")


def mine_question(spec, lines, records, lines_by_index, mol_cache) -> dict[str, object]:
    print(f"\n=== {spec.question_id} ===", flush=True)
    s1_hits, s2_hits, s1_template, s2_template = classify_question_steps(
        lines, spec, mol_cache=mol_cache
    )
    print(
        f"Step1 [{s1_template.name}] reactions={len(s1_hits)} | "
        f"Step2 [{s2_template.name}] reactions={len(s2_hits)}",
        flush=True,
    )

    chains = enumerate_chains(
        records,
        set(s1_hits),
        set(s2_hits),
        question_id=spec.question_id,
        require_persistence=True,
    )
    chains_no_persist = enumerate_chains(
        records,
        set(s1_hits),
        set(s2_hits),
        question_id=spec.question_id,
        require_persistence=False,
    )
    print(f"Chains with persistence: {len(chains)} | without: {len(chains_no_persist)}")

    if chains:
        audit_chains(
            records,
            lines_by_index,
            chains,
            question_id=spec.question_id,
            s1_hits=s1_hits,
            s2_hits=s2_hits,
        )

    support = sorted({idx for c in chains for idx in c.reaction_indices})
    return {
        "question_id": spec.question_id,
        "label": spec.label,
        "path_length": CHAIN_LENGTH,
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
        "step1_reaction_count": len(s1_hits),
        "step2_reaction_count": len(s2_hits),
        "smirks": smirks_documentation(spec),
        "step1_template_key": s1_template.key,
        "step2_template_key": s2_template.key,
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
        mined[spec.question_id] = mine_question(spec, lines, records, lines_by_index, mol_cache)

    with CHAINS_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(mined, handle, indent=2)
    print(f"\nWrote {CHAINS_JSON_PATH}")


if __name__ == "__main__":
    main()
