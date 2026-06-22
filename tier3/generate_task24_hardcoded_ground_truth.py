"""Generate hardcoded ground truth for tier3 task24.

Usage:
  /home/bhagavan/rlms/.rlm/bin/python generate_task24_hardcoded_ground_truth.py
"""

from __future__ import annotations

import random
from pprint import pformat

from rdkit import Chem, rdBase

from rlm.codeact_helpers import load_lines
from task24_e_double_bond_evaluator import (
    TASK24_GROUND_TRUTH_DEFINITION,
    compute_ground_truth_indices,
    count_e_double_bonds,
    parse_reaction,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
OUTPUT_PATH = "task24_hardcoded_ground_truth.py"


def main() -> None:
    lines = load_lines(DATASET_PATH)
    indices, skipped = compute_ground_truth_indices(lines)
    rdkit_version = rdBase.rdkitVersion

    print(f"Total reactions: {len(lines)}")
    print(f"Matching: {len(indices)} ({100 * len(indices) / len(lines):.1f}%)")
    print(f"Skipped: {skipped}")
    print(f"RDKit version: {rdkit_version}")

    # Unit checks from the task spec
    assert count_e_double_bonds(Chem.MolFromSmiles("C/C=C/C")) == 1
    assert count_e_double_bonds(Chem.MolFromSmiles("C/C=C\\C")) == 0
    assert count_e_double_bonds(Chem.MolFromSmiles("CC=CC")) == 0
    assert count_e_double_bonds(Chem.MolFromSmiles("c1ccccc1")) == 0
    assert count_e_double_bonds(Chem.MolFromSmiles("C/C=C/C=C/C")) == 2
    print("Unit checks passed")

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
        _, _, products = parse_reaction(line_by_idx[idx])
        print(f"  {idx}: {products[0][:200] if products else ''}")

    contents = f'''"""Hardcoded ground-truth indices for tier3 task24.
Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit {rdkit_version}.
Ground truth computed using Chem.AssignStereochemistry(mol, cleanIt=True, force=True).
Bonds are counted as E-configured if they are non-aromatic C=C double bonds and have
bond.GetStereo() == Chem.BondStereo.STEREOE (or STEREOTRANS as fallback). Reactions are
included if at least one product molecule contains at least one such bond. Unannotated
double bonds are excluded.
"""

TASK24_TOTAL_REACTIONS = {len(lines)}
TASK24_VALID_REACTIONS = {len(lines) - skipped}
TASK24_SKIPPED_REACTIONS = {skipped}
TASK24_POSITIVE_REACTIONS = {len(indices)}
TASK24_GROUND_TRUTH_DEFINITION = {TASK24_GROUND_TRUTH_DEFINITION!r}
TASK24_RDKIT_VERSION = {rdkit_version!r}

TASK24_HARDCODED_GROUND_TRUTH_INDICES = {pformat(indices, width=100)}
'''
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(contents)
    print(f"Wrote {OUTPUT_PATH}: valid={len(lines) - skipped} skipped={skipped}")
    print(f"positives={len(indices)}")


if __name__ == "__main__":
    main()
