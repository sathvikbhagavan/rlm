from __future__ import annotations

from pathlib import Path

from tier3.generate_hardcoded_ground_truth import (
    compute_task18_gt,
    load_indexed_lines,
    reaction_constructs_new_ring_system,
)
from tier3.task18_hardcoded_ground_truth import (
    TASK18_HARDCODED_GROUND_TRUTH_INDICES,
    TASK18_POSITIVE_REACTIONS,
)
from tier3.task23_hardcoded_ground_truth import (
    TASK23_HARDCODED_GROUND_TRUTH_INDICES,
    TASK23_POSITIVE_REACTIONS,
)
from tier3.task23_stereocenter_evaluator import compute_ground_truth_indices as compute_task23


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "human_eval/data/reactionSmilesFigShareUSPTO2023_cleaned.txt"


def test_task18_ring_equivalence_ignores_substitution_but_retains_ring_chemistry():
    assert not reaction_constructs_new_ring_system(["c1ccccc1Br"], ["c1ccccc1C"])
    assert reaction_constructs_new_ring_system(["CCCCCC"], ["C1CCCCC1"])
    assert reaction_constructs_new_ring_system(["C1CCCCC1"], ["c1ccccc1"])
    assert reaction_constructs_new_ring_system(["c1ccccc1"], ["c1ccncc1"])


def test_task18_frozen_answer_recomputes_from_the_clean_dataset():
    lines = load_indexed_lines(str(DATASET))
    computed, valid, skipped = compute_task18_gt(lines)
    assert len(lines) == valid == 122456
    assert skipped == 0
    assert TASK18_POSITIVE_REACTIONS == 17022
    assert computed == TASK18_HARDCODED_GROUND_TRUTH_INDICES
    # Dataset reaction 3 is a coupling/substitution on existing aromatic rings,
    # not construction of a new ring system.
    assert 3 not in computed


def test_task23_frozen_answer_uses_uppercase_absolute_rs_only():
    lines = load_indexed_lines(str(DATASET))
    computed, skipped = compute_task23(lines)
    assert skipped == 0
    assert TASK23_POSITIVE_REACTIONS == 1410
    assert computed == TASK23_HARDCODED_GROUND_TRUTH_INDICES
    # 5375 was one of the reported pseudoasymmetric false positives; 55161 is
    # retained because lowercase r/s on a reactant is outside the stated R/S predicate.
    assert 5375 not in computed
    assert 55161 in computed
