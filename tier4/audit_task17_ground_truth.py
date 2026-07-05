"""Verify task17 hardcoded ground truth against mined JSON and classifiers."""

from __future__ import annotations

from rlm.codeact_helpers import load_lines
from task17_ground_truth import (
    CHAINS_JSON_PATH,
    HARDCODED_GT_CHAIN_COUNTS,
    HARDCODED_GT_EXAMPLE,
    FIXED_QUESTIONS,
    full_dataset_chain_count,
    hardcoded_chains_for_question,
    load_mined_payload,
)
from task17_smirks_sequential_graph import (
    CHAIN_LENGTH,
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

    print(f"JSON: {CHAINS_JSON_PATH}")
    print(f"Questions in JSON: {len(payload)}")
    assert set(payload) == {q.question_id for q in FIXED_QUESTIONS}, "JSON keys != FIXED_QUESTIONS"

    failures = 0
    for spec in FIXED_QUESTIONS:
        qid = spec.question_id
        mined = payload[qid]
        chains = hardcoded_chains_for_question(qid)
        json_count = int(mined["chain_count"])
        hard_count = HARDCODED_GT_CHAIN_COUNTS[qid]
        example = tuple(mined["example_chain"])

        ok = True
        if json_count != len(chains):
            print(f"FAIL {qid}: json chain_count={json_count} != len(chains)={len(chains)}")
            ok = False
        if hard_count != json_count:
            print(f"FAIL {qid}: hardcoded count={hard_count} != json={json_count}")
            ok = False
        if full_dataset_chain_count(spec) != json_count:
            print(f"FAIL {qid}: full_dataset_chain_count mismatch")
            ok = False
        if example and example != HARDCODED_GT_EXAMPLE.get(qid):
            print(f"FAIL {qid}: example chain mismatch hardcoded vs json")
            ok = False
        if example and example not in chains:
            print(f"FAIL {qid}: example chain not in chains list")
            ok = False

        s1, s2, _, _ = classify_question_steps(lines, spec)
        verify_fail = 0
        for chain in chains[: min(10, len(chains))]:
            ok_v, reason = verify_chain(
                chain,
                qid,
                records,
                lines_by_index,
                s1_hits=s1,
                s2_hits=s2,
            )
            if not ok_v:
                verify_fail += 1
                print(f"  verify fail {chain}: {reason}")
        if verify_fail:
            print(f"FAIL {qid}: {verify_fail} chains failed verify (sampled up to 10)")
            ok = False

        status = "OK" if ok else "BAD"
        print(
            f"{status} {qid}: chains={json_count} "
            f"s1={len(s1)} s2={len(s2)} example={list(example)}"
        )
        if not ok:
            failures += 1

    if failures:
        raise SystemExit(f"{failures} question(s) failed audit")
    print(f"\nAll {len(FIXED_QUESTIONS)} questions passed audit.")


if __name__ == "__main__":
    main()
