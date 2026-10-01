from __future__ import annotations

from pathlib import Path

from tier3.generate_hardcoded_ground_truth import (
    TASK6_AMIDE_COUPLING_SMIRKS,
    TASK7_TO_FG_SMIRKS,
    compute_task7_gt,
    compute_task18_gt,
    load_indexed_lines,
    reaction_constructs_new_ring_system,
    task6_build_reaction_query,
    task6_reaction_matches,
)
from tier3.task7_hardcoded_ground_truth import (
    TASK7_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION,
    TASK7_POSITIVE_REACTIONS_BY_KEY,
)
from tier3.task10_mechanism_evaluator import reaction_line_matches_mechanism
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


def test_task7_frozen_answer_includes_repeated_multi_site_transformations():
    lines = load_indexed_lines(str(DATASET))
    computed, valid, skipped = compute_task7_gt(lines)
    assert len(lines) == valid == 122456
    assert skipped == 0
    assert computed == TASK7_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION
    assert TASK7_POSITIVE_REACTIONS_BY_KEY == {
        "grignard_ketone_to_tertiary_alcohol": 43,
        "grignard_aldehyde_to_secondary_alcohol": 78,
        "nitrile_to_amine": 275,
        "nitro_groups_to_amines": 2067,
        "alcohol_to_azide": 133,
        "alcohol_to_carboxylic_acid": 64,
    }
    # Reaction 3907 oxidizes both primary alcohols in one reactant to the
    # corresponding dicarboxylic acid in the recorded product.
    assert 3907 in computed["alcohol_to_carboxylic_acid"]


def test_task6_acyl_chloride_excludes_chloroformates():
    lines = load_indexed_lines(str(DATASET))
    query = task6_build_reaction_query(
        TASK6_AMIDE_COUPLING_SMIRKS["acyl_chloride_with_primary_amine"]
    )
    # 68 is carbamate formation from benzyl chloroformate; 72 is ordinary
    # carbon-substituted acyl-chloride amide formation.
    assert not task6_reaction_matches(lines[68], query)
    assert task6_reaction_matches(lines[72], query)


def test_task7_connectivity_contract_accepts_stereospecified_grignard_product():
    lines = load_indexed_lines(str(DATASET))
    query = task6_build_reaction_query(
        TASK7_TO_FG_SMIRKS["grignard_ketone_to_tertiary_alcohol"]
    )
    assert task6_reaction_matches(
        lines[8243],
        query,
        allow_repeated_single_reactant_transform=True,
        isomeric_smiles=False,
    )


def test_task10_wittig_requires_a_carbon_ylide_not_phosphorus_sulfide():
    lines = load_indexed_lines(str(DATASET))
    assert reaction_line_matches_mechanism(lines[2533], "wittig_olefination")
    # Lawesson-reagent thionation was the first reported false-positive family.
    assert not reaction_line_matches_mechanism(lines[11206], "wittig_olefination")


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
