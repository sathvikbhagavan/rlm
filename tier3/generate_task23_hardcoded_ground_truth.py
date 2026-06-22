"""Generate hardcoded ground truth for tier3 task23.

Usage:
  /home/bhagavan/rlms/.rlm/bin/python generate_task23_hardcoded_ground_truth.py
"""

from __future__ import annotations

import random
from pprint import pformat

from rdkit import rdBase

from rlm.codeact_helpers import load_lines
from task23_stereocenter_evaluator import (
    TASK23_GROUND_TRUTH_DEFINITION,
    compute_ground_truth_indices,
    count_assigned_stereocenters,
    parse_reaction,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
OUTPUT_PATH = "task23_hardcoded_ground_truth.py"


def main() -> None:
    lines = load_lines(DATASET_PATH)
    indices, skipped = compute_ground_truth_indices(lines)
    rdkit_version = rdBase.rdkitVersion

    print(f"Total reactions: {len(lines)}")
    print(f"Matching: {len(indices)} ({100 * len(indices) / len(lines):.1f}%)")
    print(f"Skipped: {skipped}")
    print(f"RDKit version: {rdkit_version}")

    line_by_idx = {}
    for line in lines:
        try:
            idx_str, reaction = line.split(" ", 1)
            line_by_idx[int(idx_str)] = reaction
        except ValueError:
            continue

    sample = random.sample(sorted(indices), min(5, len(indices)))
    print("Sample matching reactions:")
    for idx in sample:
        print(f"  {idx}: {line_by_idx[idx][:200]}")

    contents = f'''"""Hardcoded ground-truth indices for tier3 task23.
Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit {rdkit_version}.
Ground truth computed using Chem.FindMolChiralCenters(mol, includeUnassigned=False,
useLegacyImplementation=False). A reaction is included if and only if every reactant SMILES
has zero assigned stereocenters and at least one product SMILES has one or more assigned
stereocenters. Reagents are not considered. Unparseable reactions are excluded.
"""

TASK23_TOTAL_REACTIONS = {len(lines)}
TASK23_VALID_REACTIONS = {len(lines) - skipped}
TASK23_SKIPPED_REACTIONS = {skipped}
TASK23_POSITIVE_REACTIONS = {len(indices)}
TASK23_GROUND_TRUTH_DEFINITION = {TASK23_GROUND_TRUTH_DEFINITION!r}
TASK23_RDKIT_VERSION = {rdkit_version!r}

TASK23_HARDCODED_GROUND_TRUTH_INDICES = {pformat(indices, width=100)}
'''
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(contents)
    print(f"Wrote {OUTPUT_PATH}: valid={len(lines) - skipped} skipped={skipped}")
    print(f"positives={len(indices)}")


if __name__ == "__main__":
    main()
