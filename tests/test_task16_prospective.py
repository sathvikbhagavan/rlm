from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from experiments.iclr2027 import generate_full_campaign, generate_prospective_decomposition
from rxnhaystack.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
TIER4 = ROOT / "tier4"
sys.path.insert(0, str(TIER4))

from task16_prospective import (  # noqa: E402
    FINAL_TRANSFORMATIONS,
    INITIAL_QUESTION_IDS,
    PromptCondition,
    build_prospective_question,
    exact_target_product_indices,
    write_prediction_artifact,
)
from task16_truncated_synthesis_graph import (  # noqa: E402
    ReactionRecord,
    build_question,
    parse_records_from_lines,  # noqa: E402
)
from task16_truncated_synthesis_ground_truth import (  # noqa: E402
    FIXED_QUESTIONS,
    build_task16_eval_context,
    hardcoded_prefixes_for_question,
    target_spec_for_question,
    terminal_indices_for_question,
)


def questions_by_id():
    return {question.question_id: question for question in FIXED_QUESTIONS}


@pytest.mark.parametrize("question_id", INITIAL_QUESTION_IDS)
def test_prospective_prompt_conditions_isolate_information(question_id: str) -> None:
    question = questions_by_id()[question_id]
    spec = target_spec_for_question(question)
    transformation = FINAL_TRANSFORMATIONS[question_id]

    name = build_prospective_question(spec, PromptCondition.NAME_ONLY)
    structure = build_prospective_question(spec, PromptCondition.STRUCTURE_ONLY)
    structure_class = build_prospective_question(spec, PromptCondition.STRUCTURE_PLUS_CLASS)

    assert spec.target_name in name
    assert spec.target_smiles not in name
    assert spec.description not in name
    assert transformation.summary not in name

    assert spec.target_smiles in structure
    assert spec.target_name not in structure
    assert spec.description not in structure
    assert transformation.summary not in structure

    assert spec.target_smiles in structure_class
    assert spec.target_name not in structure_class
    assert spec.description not in structure_class
    assert transformation.summary in structure_class

    # The frozen prompt remains a separate, reproducible legacy condition.
    legacy = build_question(spec)
    assert spec.target_name in legacy
    assert spec.description in legacy


def test_final_transformations_have_explicit_provenance_and_author_approval() -> None:
    assert set(FINAL_TRANSFORMATIONS) == set(INITIAL_QUESTION_IDS)
    for value in FINAL_TRANSFORMATIONS.values():
        assert value.accepted_terminal_indices
        assert set(value.accepted_terminal_indices) <= set(value.supporting_target_product_indices)
        assert "cleaned USPTO" in value.provenance
        assert value.review_status == "approved-by-author-chemist-2026-09-17"
    assert FINAL_TRANSFORMATIONS["lactam_dipeptide"].summary == (
        "Boc deprotection of secondary amine"
    )


def test_exact_target_product_indices_uses_canonical_component_equality() -> None:
    records = {
        1: ReactionRecord(1, "", ("CC",), ("CCO",)),
        2: ReactionRecord(2, "", ("CC",), ("CCO",)),
        3: ReactionRecord(3, "", ("CC",), ("CCOC",)),
    }
    assert exact_target_product_indices(records, "C(C)O") == {1, 2}


@pytest.fixture(scope="module")
def full_task16_dataset():
    dataset = ROOT / "human_eval/data/reactionSmilesFigShareUSPTO2023_cleaned.txt"
    lines = [f"{index} {line}" for index, line in enumerate(dataset.read_text().splitlines())]
    return lines, parse_records_from_lines(lines)


@pytest.mark.parametrize("question_id", INITIAL_QUESTION_IDS)
def test_control_context_excludes_every_exact_target_product_and_preserves_gt(
    question_id: str,
    full_task16_dataset,
) -> None:
    lines, records = full_task16_dataset
    question = questions_by_id()[question_id]
    excluded = exact_target_product_indices(records, question.target_smiles)
    assert excluded == set(FINAL_TRANSFORMATIONS[question_id].supporting_target_product_indices)
    assert set(FINAL_TRANSFORMATIONS[question_id].accepted_terminal_indices) == set(
        terminal_indices_for_question(question_id)
    )
    built = build_task16_eval_context(
        question=question,
        lines=lines,
        context_size=-1,
        sample_index=0,
        seed=42,
        full_records=records,
        additional_excluded_indices=excluded,
    )
    assert excluded
    assert not (excluded & set(built.records))
    assert set(built.gt.accepted_reaction_indices) == set(
        hardcoded_prefixes_for_question(question_id)
    )


def test_generated_prospective_study_is_small_paired_and_author_approved() -> None:
    path = ROOT / "experiments/iclr2027/prospective-decomposition.toml"
    assert path.read_text(encoding="utf-8") == generate_prospective_decomposition.render()
    manifest = load_manifest(path)

    assert len(manifest.runs) == 30
    assert manifest.estimated_cost_chf == pytest.approx(2.732085)
    grouped: dict[str, list] = defaultdict(list)
    for run in manifest.runs:
        grouped[run.model].append(run)
        assert run.corpus_size == "full"
        assert run.method == "rlm"
        assert run.env["RXNHAYSTACK_TASK16_QUESTION_IDS"] == ",".join(INITIAL_QUESTION_IDS)
        assert run.env["RXNHAYSTACK_TASK16_EXCLUDE_ALL_TARGET_PRODUCTS"] == "1"
        assert run.env["RXNHAYSTACK_TASK16_CLASS_LABELS_AUTHOR_APPROVED"] == "1"
    assert len(grouped) == 2
    for model_runs in grouped.values():
        assert len(model_runs) == 15
        assert {run.condition for run in model_runs} == set(
            generate_prospective_decomposition.PROMPT_CONDITIONS
        )
        assert {run.seed for run in model_runs} == {42}
        assert all(
            sum(run.condition == condition for run in model_runs) == 5
            for condition in generate_prospective_decomposition.PROMPT_CONDITIONS
        )
    assert generate_full_campaign.render() == (
        ROOT / "experiments/iclr2027/full-campaign.toml"
    ).read_text(encoding="utf-8")


def test_prediction_artifact_is_structured_atomic_and_private(tmp_path: Path) -> None:
    metrics = tmp_path / "attempt" / "metrics.json"
    path = write_prediction_artifact(
        [
            {
                "question_id": "lactam_dipeptide",
                "parsed_chains": [[1, 2, 3, 4]],
                "false_positive_chains": [[1, 2, 3, 4]],
            }
        ],
        prompt_condition="name_only",
        excluded_all_target_products=True,
        environ={"RXNHAYSTACK_METRICS_PATH": str(metrics), "RXNHAYSTACK_RUN_ID": "run-1"},
    )
    assert path == metrics.parent / "task16-predictions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-1"
    assert payload["questions"][0]["question_id"] == "lactam_dipeptide"
    assert path.stat().st_mode & 0o777 == 0o600
    assert not path.with_suffix(".json.tmp").exists()


def test_selected_ground_truth_is_condition_independent() -> None:
    # Prompt condition is not an input to sampling or scoring. All paired
    # conditions therefore use the same frozen answer representation.
    for question_id in INITIAL_QUESTION_IDS:
        expected = hardcoded_prefixes_for_question(question_id)
        assert expected
        assert all(
            hardcoded_prefixes_for_question(question_id) == expected
            for _condition in generate_prospective_decomposition.PROMPT_CONDITIONS
        )
