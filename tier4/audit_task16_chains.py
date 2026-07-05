"""Audit task16 mined chains for trivial vs interesting synthesis steps."""

from __future__ import annotations

from collections import Counter

from rdkit import Chem
from rlm.codeact_helpers import load_lines, parse_reaction_sides
from task16_truncated_synthesis_graph import (
    FULL_CHAIN_LENGTH,
    is_clean_synthesis_chain,
    parse_records_from_lines,
)
from task16_truncated_synthesis_ground_truth import (
    FIXED_QUESTIONS,
    hardcoded_full_chains_for_question,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MIN_ORGANIC_HEAVY = 3


def heavy(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol else 0


def organic_components(components: tuple[str, ...]) -> set[str]:
    return {smi for smi in components if heavy(smi) >= MIN_ORGANIC_HEAVY}


def reaction_delta(rec) -> int:
    r_heavy = sum(heavy(smi) for smi in rec.reactants if heavy(smi) >= MIN_ORGANIC_HEAVY)
    p_heavy = sum(heavy(smi) for smi in rec.products if heavy(smi) >= MIN_ORGANIC_HEAVY)
    return p_heavy - r_heavy


def is_identity_step(rec) -> bool:
    return organic_components(rec.reactants) == organic_components(rec.products) and bool(
        organic_components(rec.reactants)
    )


def is_salt_only_step(rec) -> bool:
    r_org = organic_components(rec.reactants)
    p_org = organic_components(rec.products)
    if not r_org or not p_org:
        return True
    return r_org == p_org


def analyze_target(question, records, line_by_id) -> dict[str, object]:
    full = hardcoded_full_chains_for_question(question.question_id)
    target = Chem.CanonSmiles(question.target_smiles)

    identity_by_pos = Counter()
    salt_by_pos = Counter()
    zero_delta_by_pos = Counter()
    rxn_freq = Counter()
    hub_rxns = Counter()
    meaningful_steps_per_chain: list[int] = []

    for chain in full:
        meaningful = 0
        for pos, reaction_idx in enumerate(chain):
            rec = records[reaction_idx]
            if is_identity_step(rec):
                identity_by_pos[pos] += 1
            if is_salt_only_step(rec):
                salt_by_pos[pos] += 1
            if reaction_delta(rec) == 0:
                zero_delta_by_pos[pos] += 1
            if not is_salt_only_step(rec) and reaction_delta(rec) != 0:
                meaningful += 1
            rxn_freq[reaction_idx] += 1
            if reaction_idx in {1300, 1301, 66636}:
                hub_rxns[reaction_idx] += 1
        meaningful_steps_per_chain.append(meaningful)

    example = full[0]
    example_steps = []
    for pos, reaction_idx in enumerate(example):
        rec = records[reaction_idx]
        example_steps.append(
            {
                "pos": pos + 1,
                "rid": reaction_idx,
                "identity": is_identity_step(rec),
                "salt_only": is_salt_only_step(rec),
                "delta": reaction_delta(rec),
                "line": line_by_id[reaction_idx].split(" ", 1)[1][:140],
            }
        )

    return {
        "question_id": question.question_id,
        "chain_count": len(full),
        "fails_clean": sum(
            1 for chain in full if not is_clean_synthesis_chain(chain, question.target_smiles, records)
        ),
        "identity_by_pos": [identity_by_pos[i] for i in range(FULL_CHAIN_LENGTH)],
        "salt_by_pos": [salt_by_pos[i] for i in range(FULL_CHAIN_LENGTH)],
        "zero_delta_by_pos": [zero_delta_by_pos[i] for i in range(FULL_CHAIN_LENGTH)],
        "mean_meaningful_steps": sum(meaningful_steps_per_chain) / len(meaningful_steps_per_chain),
        "min_meaningful_steps": min(meaningful_steps_per_chain),
        "max_meaningful_steps": max(meaningful_steps_per_chain),
        "top_rxns": rxn_freq.most_common(8),
        "hub_rxns": dict(hub_rxns),
        "example_chain": example,
        "example_steps": example_steps,
    }


def main() -> None:
    lines = load_lines(DATASET_PATH)
    records = parse_records_from_lines(lines)
    line_by_id = {int(line.split(" ", 1)[0]): line for line in lines}

    for question in FIXED_QUESTIONS:
        stats = analyze_target(question, records, line_by_id)
        print(f"\n{'=' * 80}")
        print(f"{stats['question_id']}: {stats['chain_count']} chains")
        print(f"  fails clean filter: {stats['fails_clean']}")
        print(f"  identity steps by pos 1-5: {stats['identity_by_pos']}")
        print(f"  salt-only steps by pos:    {stats['salt_by_pos']}")
        print(f"  zero-delta steps by pos:   {stats['zero_delta_by_pos']}")
        print(
            f"  meaningful steps/chain: min={stats['min_meaningful_steps']} "
            f"mean={stats['mean_meaningful_steps']:.1f} max={stats['max_meaningful_steps']}"
        )
        print(f"  hub rxns in chains: {stats['hub_rxns']}")
        print(f"  top reactions: {stats['top_rxns']}")
        print(f"  example chain: {stats['example_chain']}")
        for step in stats["example_steps"]:
            print(
                f"    [{step['pos']}] {step['rid']}: "
                f"identity={step['identity']} salt={step['salt_only']} dH={step['delta']}"
            )
            print(f"         {step['line']}")


if __name__ == "__main__":
    main()
