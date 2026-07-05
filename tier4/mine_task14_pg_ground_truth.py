"""Mine full-dataset protecting-group pairs for tier4 task14."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rlm.codeact_helpers import load_lines
from task14_protecting_group_graph import (
    DATASET_TOTAL_REACTIONS,
    PROTECTING_GROUPS,
    build_ground_truth_pairs,
    count_by_label,
    mine_protection_events,
    parse_records_from_lines,
    summarize_gt_pair,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
TIER4_DIR = Path(__file__).resolve().parent
PAIRS_JSON_PATH = TIER4_DIR / "task14_pg_hardcoded_pairs.json"
GROUND_TRUTH_PY_PATH = TIER4_DIR / "task14_protecting_group_ground_truth.py"


def mine_full_dataset() -> dict[str, dict[str, object]]:
    lines = load_lines(DATASET_PATH)
    records = parse_records_from_lines(lines)
    events = mine_protection_events(records)
    pairs = build_ground_truth_pairs(events, max_pairs_per_group=0)

    print(f"Loaded {len(records)} reactions from {DATASET_TOTAL_REACTIONS}-reaction cleaned dataset")
    print(f"Mined {len(events)} protection events")
    print(f"Event counts: {json.dumps(count_by_label(events), sort_keys=True)}")
    print(f"Pair counts: {json.dumps(count_by_label(pairs), sort_keys=True)}")

    mined: dict[str, dict[str, object]] = {}
    for spec in PROTECTING_GROUPS:
        group_pairs = [pair for pair in pairs if pair.pg_label == spec.label]
        support = sorted({idx for pair in group_pairs for idx in (pair.install_index, pair.remove_index)})
        mined[spec.label] = {
            "pg_label": spec.label,
            "functional_group": spec.functional_group,
            "pair_count": len(group_pairs),
            "support_index_count": len(support),
            "example_pair": (
                [group_pairs[0].install_index, group_pairs[0].remove_index]
                if group_pairs
                else None
            ),
            "pairs": [summarize_gt_pair(pair) for pair in group_pairs],
        }
        print(
            f"{spec.label}: pairs={len(group_pairs)} "
            f"support={len(support)} "
            f"example={mined[spec.label]['example_pair']}"
        )
    return mined


def write_pairs_json(mined: dict[str, dict[str, object]]) -> None:
    with PAIRS_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(mined, handle, separators=(",", ":"))


def update_pair_counts(mined: dict[str, dict[str, object]]) -> None:
    text = GROUND_TRUTH_PY_PATH.read_text(encoding="utf-8")
    count_lines = [
        f'    {label!r}: {payload["pair_count"]},'
        for label, payload in mined.items()
    ]
    replacement = "HARDCODED_GT_PAIR_COUNTS: dict[str, int] = {\n" + "\n".join(count_lines) + "\n}"
    text, count = re.subn(
        r"HARDCODED_GT_PAIR_COUNTS: dict\[str, int\] = \{[^}]+\}",
        replacement,
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Failed to update HARDCODED_GT_PAIR_COUNTS in ground_truth module")
    GROUND_TRUTH_PY_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    mined = mine_full_dataset()
    write_pairs_json(mined)
    update_pair_counts(mined)
    print(f"Wrote {PAIRS_JSON_PATH}")
    print(f"Updated {GROUND_TRUTH_PY_PATH}")


if __name__ == "__main__":
    main()
