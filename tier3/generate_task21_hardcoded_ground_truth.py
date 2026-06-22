"""Generate hardcoded ground truth for tier3 task21.

Usage:
  /home/bhagavan/rlms/.rlm/bin/python generate_task21_hardcoded_ground_truth.py
"""

from __future__ import annotations

from pprint import pformat

from rlm.codeact_helpers import load_lines
from task21_transition_metal_evaluator import (
    TASK21_GROUND_TRUTH_DEFINITION,
    TRANSITION_METAL_SYMBOLS,
    compute_ground_truth_indices,
    compute_metal_frequency,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
OUTPUT_PATH = "task21_hardcoded_ground_truth.py"


def main() -> None:
    lines = load_lines(DATASET_PATH)
    indices, skipped = compute_ground_truth_indices(lines)
    metal_frequency = dict(compute_metal_frequency(lines, indices).most_common())
    contents = f'''"""Hardcoded ground-truth indices for tier3 task21.
Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit reagent-slot atom-number detection.
"""

TASK21_TOTAL_REACTIONS = {len(lines)}
TASK21_VALID_REACTIONS = {len(lines) - skipped}
TASK21_SKIPPED_REACTIONS = {skipped}
TASK21_POSITIVE_REACTIONS = {len(indices)}
TASK21_GROUND_TRUTH_DEFINITION = {TASK21_GROUND_TRUTH_DEFINITION!r}
TASK21_TRANSITION_METAL_SYMBOLS = {TRANSITION_METAL_SYMBOLS!r}
TASK21_METAL_FREQUENCY = {pformat(metal_frequency, width=100, sort_dicts=False)}

TASK21_HARDCODED_GROUND_TRUTH_INDICES = {pformat(indices, width=100)}
'''
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(contents)
    print(f"Wrote {OUTPUT_PATH}: valid={len(lines) - skipped} skipped={skipped}")
    print(f"positives={len(indices)}")
    print(f"metal_frequency={metal_frequency}")


if __name__ == "__main__":
    main()
