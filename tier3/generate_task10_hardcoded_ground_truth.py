"""Generate hardcoded ground truth for tier3 task10.

Usage:
  /home/bhagavan/rlms/.rlm/bin/python generate_task10_hardcoded_ground_truth.py
"""

from __future__ import annotations

from pprint import pformat

from rlm.codeact_helpers import load_lines
from task10_mechanism_evaluator import (
    REACTION_KEYS,
    TASK10_GROUND_TRUTH_DEFINITION,
    compute_ground_truth_indices_by_reaction,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
OUTPUT_PATH = "task10_hardcoded_ground_truth.py"


def main() -> None:
    lines = load_lines(DATASET_PATH)
    indices_by_reaction, skipped = compute_ground_truth_indices_by_reaction(lines)
    positive_counts = {
        reaction_key: len(indices_by_reaction[reaction_key])
        for reaction_key in REACTION_KEYS
    }
    contents = f'''"""Hardcoded ground-truth indices for tier3 task10.
Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with the task10 RDKit mechanism cascade evaluator.
"""

TASK10_TOTAL_REACTIONS = {len(lines)}
TASK10_VALID_REACTIONS = {len(lines) - skipped}
TASK10_SKIPPED_REACTIONS = {skipped}
TASK10_GROUND_TRUTH_DEFINITION = {TASK10_GROUND_TRUTH_DEFINITION!r}
TASK10_POSITIVE_REACTIONS_BY_KEY = {pformat(positive_counts, width=100)}

TASK10_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION = {pformat(indices_by_reaction, width=100)}
'''
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(contents)
    print(f"Wrote {OUTPUT_PATH}: valid={len(lines) - skipped} skipped={skipped}")
    for reaction_key in REACTION_KEYS:
        print(f"{reaction_key}: positives={positive_counts[reaction_key]}")


if __name__ == "__main__":
    main()
