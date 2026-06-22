"""Generate hardcoded ground truth for tier3 task22.

Usage:
  /home/bhagavan/rlms/.rlm/bin/python generate_task22_hardcoded_ground_truth.py
"""

from __future__ import annotations

from collections import Counter
from pprint import pformat

from rlm.codeact_helpers import load_lines
from task22_coupling_reagent_evaluator import (
    HATU_PF6_CANONICAL_FRAGMENTS,
    TASK22_GROUND_TRUTH_DEFINITION,
    T3P_CANONICAL_SMILES,
    classify_reagent_match,
    compute_ground_truth_indices,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
OUTPUT_PATH = "task22_hardcoded_ground_truth.py"


def main() -> None:
    lines = load_lines(DATASET_PATH)
    indices, skipped = compute_ground_truth_indices(lines)
    line_by_idx = {}
    for line in lines:
        try:
            idx_str, reaction = line.split(" ", 1)
            line_by_idx[int(idx_str)] = reaction
        except ValueError:
            continue

    reagent_counts: Counter[str] = Counter()
    for idx in indices:
        reagent_counts.update(classify_reagent_match(line_by_idx[idx]))

    contents = f'''"""Hardcoded ground-truth indices for tier3 task22.
Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit reagent-slot canonical matching.
"""

TASK22_TOTAL_REACTIONS = {len(lines)}
TASK22_VALID_REACTIONS = {len(lines) - skipped}
TASK22_SKIPPED_REACTIONS = {skipped}
TASK22_POSITIVE_REACTIONS = {len(indices)}
TASK22_GROUND_TRUTH_DEFINITION = {TASK22_GROUND_TRUTH_DEFINITION!r}
TASK22_HATU_PF6_CANONICAL_FRAGMENTS = {tuple(sorted(HATU_PF6_CANONICAL_FRAGMENTS))!r}
TASK22_T3P_CANONICAL_SMILES = {T3P_CANONICAL_SMILES!r}
TASK22_REAGENT_COUNTS = {pformat(dict(reagent_counts.most_common()), width=100, sort_dicts=False)}

TASK22_HARDCODED_GROUND_TRUTH_INDICES = {pformat(indices, width=100)}
'''
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(contents)
    print(f"Wrote {OUTPUT_PATH}: valid={len(lines) - skipped} skipped={skipped}")
    print(f"positives={len(indices)}")
    print(f"reagent_counts={dict(reagent_counts.most_common())}")


if __name__ == "__main__":
    main()
