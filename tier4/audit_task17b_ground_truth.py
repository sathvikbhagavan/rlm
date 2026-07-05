"""Audit task17b hardcoded ground truth."""

from __future__ import annotations

from rlm.codeact_helpers import load_lines
from task17b_ground_truth import (
    CHAINS_JSON_PATH,
    FIXED_QUESTIONS,
    HARDCODED_GT_CHAIN_COUNTS,
    HARDCODED_GT_EXAMPLE,
    hardcoded_chains_for_question,
    load_mined_payload,
)
from task17b_smirks_sequential_graph import (
    build_line_mol_cache,
    classify_question_steps,
    parse_records_from_lines,
    verify_chain,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"


def main() -> None:
    payload = load_mined_payload()
    lines = load_lines(DATASET_PATH)
    records = parse_records_from_lines(lines)
    lines_by_index = {int(line.split(" ", 1)[0]): line for line in lines}
    mol_cache = build_line_mol_cache(lines)

    ok = True
    for spec in FIXED_QUESTIONS:
        qid = spec.question_id
        mined = payload[qid]
        chains = hardcoded_chains_for_question(qid)
        expected = HARDCODED_GT_CHAIN_COUNTS[qid]
        if len(chains) != expected:
            print(f"FAIL {qid}: chain count {len(chains)} != hardcoded {expected}")
            ok = False
        if mined["chain_count"] != len(chains):
            print(f"FAIL {qid}: JSON chain_count mismatch")
            ok = False
        example = HARDCODED_GT_EXAMPLE[qid]
        if example not in chains:
            print(f"FAIL {qid}: example chain {example} not in mined chains")
            ok = False

        step_hits, _ = classify_question_steps(lines, spec, mol_cache=mol_cache)
        step_hit_sets = [set(h.keys()) for h in step_hits]
        for chain in chains[:10]:
            good, reason = verify_chain(
                chain,
                qid,
                records,
                lines_by_index,
                step_hits=step_hit_sets,
                chain_length=spec.chain_length,
            )
            if not good:
                print(f"FAIL {qid}: chain {chain} verify={reason}")
                ok = False

        print(
            f"OK {qid}: L={spec.chain_length} chains={len(chains)} "
            f"support={mined['support_index_count']} example={list(example)}"
        )

    print(f"\nJSON: {CHAINS_JSON_PATH}")
    print("AUDIT PASSED" if ok else "AUDIT FAILED")


if __name__ == "__main__":
    main()
