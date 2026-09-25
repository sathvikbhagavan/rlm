from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from human_eval.canonical import (
    EXPECTED,
    SUGGESTED_MINUTES_BY_TIER,
    BundleBuilder,
    build_bundle,
)
from human_eval.schema import Question
from tier4.task12b_full_dataset_ground_truth import (
    TASK12B_FULL_DATASET_HUB_COUNT,
    TASK12B_FULL_DATASET_HUB_MOLECULES,
    TASK12B_FULL_DATASET_RDKIT_VERSION,
)
from tier4.task12b_hub_molecule_graph import hub_molecules_in_context
from tier4.task13_fg_chain_graph import build_question as build_task13_question
from tier4.task13_fg_chain_graph import detect_functional_groups
from tier4.task15_ring_chain_graph import (
    MoleculeAnnotation,
    MoleculeStep,
    shortest_ring_construction_path,
)

ROOT = Path(__file__).resolve().parents[2]


def test_real_extraction_is_exact_and_deterministic(tmp_path: Path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    manifest_a = build_bundle(ROOT, first, "dataset-checksum")
    manifest_b = build_bundle(ROOT, second, "dataset-checksum")
    assert manifest_a["question_count"] == 100
    assert manifest_a["schema_version"] == "1.3.0"
    assert manifest_a["bundle_version"] == "rxnhaystack-human-1.6.0"
    assert manifest_a["taxonomy"] == EXPECTED
    assert manifest_a["questions_sha256"] == manifest_b["questions_sha256"]
    assert (first / "questions.jsonl").read_bytes() == (second / "questions.jsonl").read_bytes()
    public = [json.loads(line) for line in (first / "questions.jsonl").read_text().splitlines()]
    assert all("representation" not in item for item in public)
    assert all("relevant_reaction_indices" not in item for item in public)
    assert all(
        item["suggested_time_minutes"] == SUGGESTED_MINUTES_BY_TIER[item["tier"]] for item in public
    )
    task12b = next(item for item in public if item["question_id"] == "rxh-t4-task12b")
    assert task12b["metadata"]["rdkit_version"] == "2026.03.6"
    assert "complete 122456-reaction" in task12b["canonical_prompt"]
    assert "not only selected support hubs" in task12b["canonical_prompt"]
    protected = [
        json.loads(line) for line in (first / "admin/ground_truth.jsonl").read_text().splitlines()
    ]
    task12b_truth = next(item for item in protected if item["question_id"] == "rxh-t4-task12b")
    assert len(task12b_truth["representation"]) == 2091
    assert "BrCc1cccc(Br)c1" in task12b_truth["representation"]

    task7 = next(
        item
        for item in protected
        if item["question_id"] == "rxh-t3-task7-alcohol-to-carboxylic-acid"
    )
    assert 3907 in task7["representation"]

    task13 = next(
        item
        for item in public
        if item["question_id"] == "rxh-t4-task13-primary-alcohol-carboxylic-acid"
    )
    assert "neutral, protonated R-C(=O)-OH" in task13["canonical_prompt"]

    task15_truth = {
        item["question_id"]: len(item["representation"])
        for item in protected
        if item["question_id"].startswith("rxh-t4-task15-")
    }
    assert task15_truth == {
        "rxh-t4-task15-benzimidazole": 142,
        "rxh-t4-task15-benzothiazole": 44,
        "rxh-t4-task15-indole": 241,
        "rxh-t4-task15-quinoline": 299,
    }


def test_schema_round_trip_and_stable_ids():
    questions, _ = BundleBuilder(ROOT, "checksum").build()
    assert len({q.question_id for q in questions}) == 100
    assert [q.question_id for q in questions] == [
        Question.from_dict(q.to_dict()).question_id for q in questions
    ]
    assert Counter(q.tier for q in questions) == Counter(EXPECTED["tiers"])
    warnings = [q for q in questions if q.metadata.get("extraction_warning")]
    assert len(warnings) == 4
    task16 = [q for q in questions if "-task16-" in q.question_id and q.tier == 4]
    sequential = [
        q
        for q in questions
        if q.tier == 4 and ("-task17-" in q.question_id or "-task17b-" in q.question_id)
    ]
    assert len(task16) == 10
    assert len(sequential) == 10
    assert {q.metadata["conceptual_family"] for q in task16} == {"prospective-truncated-synthesis"}
    assert {q.metadata["conceptual_family"] for q in sequential} == {
        "multi-constraint-sequential-template"
    }
    task15 = [q for q in questions if q.tier == 4 and q.subcategory == "ring-chain"]
    assert len(task15) == 4
    assert {q.answer_type for q in task15} == {"one_of_reaction_chains"}
    assert all(q.scoring["method"].startswith("one submitted chain") for q in task15)


def test_task12b_frozen_answer_is_exhaustive_for_clean_dataset():
    dataset = ROOT / "human_eval/data/reactionSmilesFigShareUSPTO2023_cleaned.txt"
    reactions = [line.strip() for line in dataset.read_text().splitlines() if line]
    lines = [f"{index} {reaction}" for index, reaction in enumerate(reactions)]
    computed = hub_molecules_in_context(lines, min_downstream=3, dag_mode="index_asc")
    assert len(reactions) == 122456
    assert TASK12B_FULL_DATASET_RDKIT_VERSION == "2026.03.6"
    assert TASK12B_FULL_DATASET_HUB_COUNT == 2091
    assert tuple(computed) == TASK12B_FULL_DATASET_HUB_MOLECULES
    assert computed[2] == "BrCc1cccc(Br)c1"


def test_task13_carboxylic_acid_contract_is_explicit_and_neutral_only():
    assert "carboxylic_acid" in detect_functional_groups("CC(=O)O")
    assert "carboxylic_acid" not in detect_functional_groups("CC(=O)[O-]")
    prompt = build_task13_question(
        "primary_alcohol",
        "carboxylic_acid",
        context_reaction_count=122456,
    )
    assert "[CX3](=O)[OX2H1]" in prompt
    assert "Carboxylate anions and their salts do not" in prompt


def test_task15_frozen_alternatives_are_exhaustive_not_capped_at_200():
    payload = json.loads((ROOT / "tier4/task15_ring_hardcoded_chains.json").read_text())
    assert {key: value["chain_count"] for key, value in payload.items()} == {
        "quinoline": 299,
        "indole": 241,
        "benzothiazole": 44,
        "benzimidazole": 142,
    }


def test_task15_canonical_mining_can_return_more_than_200_alternatives():
    reverse_graph: dict[str, list[MoleculeStep]] = {}
    annotations: dict[str, MoleculeAnnotation] = {}
    for number in range(201):
        precursor, first, second, target = (
            f"precursor-{number}",
            f"first-{number}",
            f"second-{number}",
            f"target-{number}",
        )
        annotations[precursor] = MoleculeAnnotation(acyclic=True, ring_systems=())
        annotations[first] = MoleculeAnnotation(acyclic=False, ring_systems=())
        annotations[second] = MoleculeAnnotation(acyclic=False, ring_systems=())
        annotations[target] = MoleculeAnnotation(acyclic=False, ring_systems=("indole",))
        reverse_graph[target] = [MoleculeStep(number * 3 + 2, second, target)]
        reverse_graph[second] = [MoleculeStep(number * 3 + 1, first, second)]
        reverse_graph[first] = [MoleculeStep(number * 3, precursor, first)]

    capped = shortest_ring_construction_path(reverse_graph, annotations, "indole", 3, 3)
    exhaustive = shortest_ring_construction_path(
        reverse_graph, annotations, "indole", 3, 3, max_accepted_chains=None
    )
    assert capped is not None and len(capped.accepted_reaction_indices) == 200
    assert exhaustive is not None and len(exhaustive.accepted_reaction_indices) == 201
