from __future__ import annotations

import json
import random

import pytest

from paper_plots.scripts.recover_corrected_scores import load_targets
from rlm.codeact_helpers import RandomContextPipeline
from rxnhaystack.score_recovery import (
    SampleMetric,
    apply_corrected_score_recoveries,
    context_indices,
    extract_tier3_predictions,
    historical_min_selected_ground_truth,
    infer_prediction,
    parse_sample_metrics,
    precision_recall_f1,
)


@pytest.mark.parametrize("context_size", [5, 10, -1])
@pytest.mark.parametrize("positive_cardinality", [None, 2])
def test_context_indices_matches_benchmark_pipeline(
    context_size: int, positive_cardinality: int | None
) -> None:
    lines = [f"{index} reaction-{index}" for index in range(30)]
    correct = {1, 4, 7, 12, 18, 22}
    expected_text = RandomContextPipeline(
        lines=lines,
        rng=random.Random(42),
        positive_cardinality=positive_cardinality,
    ).build_context(context_size=context_size, correct_indices=correct)
    expected = {int(line.split(" ", 1)[0]) for line in expected_text.splitlines()}
    actual = context_indices(
        dataset_size=len(lines),
        context_size=context_size,
        correct_indices=correct,
        rng=random.Random(42),
        positive_cardinality=positive_cardinality,
    )
    assert actual == expected


def test_extract_codeact_predictions_and_metrics() -> None:
    log = """
Question 1/1 task=new_ring_construction
===== ITERATION 1 =====
<answer>7, 2, 7</answer>
Predicted [new_ring_construction] count: 2
Ground truth [new_ring_construction] count: 2
Metrics [new_ring_construction] -> precision=0.5000 recall=0.5000 f1=0.5000 exact_match=False count_error=0 count_exact=1
"""
    predictions = extract_tier3_predictions(log, task="tier3/task18", method="codeact")
    assert predictions["rxh-t3-task18"].indices == (7, 2)
    metrics = parse_sample_metrics(log)
    assert metrics["new_ring_construction"].predicted_count == 2
    assert metrics["new_ring_construction"].f1 == 0.5


def test_extract_rlm_rich_final_answer() -> None:
    log = """
╭─ ★ Final Answer ─────────╮
│                          │
│  19, 7, 12              │
│                          │
╰──────────────────────────╯
Predicted [wittig_olefination] count: 3
Metrics [wittig_olefination] -> precision=1.0000 recall=0.6000 f1=0.7500 exact_match=False count_error=2 count_exact=0
"""
    predictions = extract_tier3_predictions(log, task="tier3/task10", method="rlm")
    assert predictions["rxh-t3-task10-wittig-olefination"].indices == (19, 7, 12)


def test_inference_requires_mathematically_determined_prediction() -> None:
    old = frozenset({1, 2})
    new = frozenset({1})
    empty = SampleMetric("x", 0, 0.0, 0.0, 0.0, False)
    prediction, reason = infer_prediction(
        metric=empty,
        historical_ground_truth_in_context=old,
        corrected_ground_truth_in_context=new,
    )
    assert prediction == frozenset()
    assert reason == "empty-prediction-from-count"

    ambiguous = SampleMetric("x", 1, 1.0, 0.5, 2 / 3, False)
    prediction, reason = infer_prediction(
        metric=ambiguous,
        historical_ground_truth_in_context=old,
        corrected_ground_truth_in_context=new,
    )
    assert prediction is None
    assert reason == "prediction-underdetermined"


def test_precision_recall_f1_empty_is_zero() -> None:
    assert precision_recall_f1(set(), {1}) == (0.0, 0.0, 0.0)


def test_apply_corrected_score_recoveries_validates_and_restores(tmp_path) -> None:
    rows = [
        {
            "run_id": "run-1",
            "task": "tier3/task23",
            "score_name": "macro_f1",
            "f1": None,
            "original_f1": 0.5,
            "score_available": False,
            "score_correction_status": "historical_score_invalidated",
            "sources": "ledger",
        }
    ]
    path = tmp_path / "recoveries.json"
    path.write_text(
        json.dumps(
            {
                "recovery_id": "recovery-v1",
                "recoveries": [
                    {
                        "run_id": "run-1",
                        "task": "tier3/task23",
                        "score_name": "macro_f1",
                        "historical_score": 0.5,
                        "score_value": 0.75,
                    },
                    {
                        "run_id": "another-table",
                        "task": "tier3/task23",
                        "score_name": "macro_f1",
                        "historical_score": 1.0,
                        "score_value": 1.0,
                    },
                ],
            }
        )
    )

    _payload, applied = apply_corrected_score_recoveries(rows, path)

    assert applied == 1
    assert rows[0]["f1"] == 0.75
    assert rows[0]["score_available"] is True
    assert rows[0]["score_correction_status"] == "corrected_exact_rescore"
    assert rows[0]["sources"].endswith("corrected-score-recovery:recovery-v1")


def test_recovery_targets_survive_a_gold_table_rebuild(tmp_path) -> None:
    path = tmp_path / "records.csv"
    path.write_text(
        "run_id,task,method,score_correction_status,original_f1\n"
        "still-pending,tier3/task6,llm,historical_score_invalidated,0.2\n"
        "already-recovered,tier3/task6,rlm,corrected_exact_rescore,0.3\n"
        "never-scored,tier3/task6,rlm,affected_without_historical_score,\n"
        "unaffected,tier2/task2,llm,not_affected,0.4\n"
    )

    targets = load_targets((path,))

    assert set(targets) == {"still-pending", "already-recovered"}


@pytest.mark.parametrize(
    ("task", "method", "expected"),
    [
        ("tier3/task6", "llm", 1),
        ("tier3/task6", "rlm", 5),
        ("tier3/task6", "codeact", 5),
        ("tier3/task18", "llm", 1),
        ("tier3/task18", "rlm", 1),
        ("tier3/task18", "codeact", 5),
        ("tier3/task7", "llm", 5),
        ("tier3/task10", "rlm", 5),
        ("tier3/task23", "codeact", 5),
    ],
)
def test_historical_min_selected_ground_truth(task: str, method: str, expected: int) -> None:
    assert historical_min_selected_ground_truth(task=task, method=method) == expected
