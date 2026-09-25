from __future__ import annotations

import json
import random
import tarfile
from io import BytesIO

import pytest

from paper_plots.scripts.recover_corrected_scores import (
    load_artifact_tar_logs,
    load_targets,
)
from rlm.codeact_helpers import RandomContextPipeline
from rxnhaystack.score_recovery import (
    ExtractedPrediction,
    SampleMetric,
    apply_corrected_score_recoveries,
    context_indices,
    extract_tier3_predictions,
    historical_min_selected_ground_truth,
    infer_metric_from_aggregate,
    infer_prediction,
    parse_sample_metrics,
    precision_recall_f1,
    score_tier3_run,
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


def test_extract_codeact_reproduces_historical_first_answer_substring() -> None:
    log = """
Question 1/1 task=new_ring_construction
===== ITERATION 1 =====
print(f"Final answer: {result_str}")
for item in matches[:5]:
    print(item[:60])
    print(item[0])
ANSWER: 6338,35263,50413
---- FINISH REASON: stop ----
Predicted [new_ring_construction] count: 6
Metrics [new_ring_construction] -> precision=0.5000 recall=0.0811 f1=0.1395 exact_match=False count_error=31 count_exact=0
"""

    predictions = extract_tier3_predictions(log, task="tier3/task18", method="codeact")

    prediction = predictions["rxh-t3-task18"]
    assert prediction.extraction_method == "codeact-historical-transcript"
    assert prediction.indices == (5, 60, 0, 6338, 35263, 50413)


def test_extract_codeact_recovers_final_code_block_without_answer_marker() -> None:
    log = """
Question 1/1 task=new_ring_construction
===== ITERATION 1 =====
No usable answer yet.
===== ITERATION 10 =====
```python
result = [0, 1, 2]
```
---- FINISH REASON: length ----
Predicted [new_ring_construction] count: 3
Metrics [new_ring_construction] -> precision=0.0000 recall=0.0000 f1=0.0000 exact_match=False count_error=34 count_exact=0
"""

    predictions = extract_tier3_predictions(log, task="tier3/task18", method="codeact")

    prediction = predictions["rxh-t3-task18"]
    assert prediction.extraction_method == "codeact-final-code-block"
    assert prediction.indices == (0, 1, 2)


def test_extract_rlm_rejoins_digits_split_by_rich_line_wrapping() -> None:
    log = """
╭─ ★ Final Answer ─────────╮
│  11                     │
│  6855,42                │
╰──────────────────────────╯
Predicted [new_ring_construction] count: 2
Metrics [new_ring_construction] -> precision=1.0000 recall=0.5000 f1=0.6667 exact_match=False count_error=2 count_exact=0
"""

    predictions = extract_tier3_predictions(log, task="tier3/task18", method="rlm")

    prediction = predictions["rxh-t3-task18"]
    assert prediction.extraction_method == "rlm-final-answer-box-wrapped"
    assert prediction.indices == (116855, 42)


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


def test_extract_rlm_reproduces_historical_first_answer_substring() -> None:
    log = """
╭─ ★ Final Answer ─────────╮
│ There should be 2334.   │
│ The final answer: all   │
│ results satisfy:        │
│ 1. nitro removed       │
│ 2. amine formed        │
│ 3. product valid       │
╰──────────────────────────╯
Predicted [nitro_groups_to_amines] count: 3
Metrics [nitro_groups_to_amines] -> precision=0.0000 recall=0.0000 f1=0.0000 exact_match=False count_error=2050 count_exact=0
"""

    predictions = extract_tier3_predictions(log, task="tier3/task7", method="rlm")

    prediction = predictions["rxh-t3-task7-nitro-groups-to-amines"]
    assert prediction.extraction_method == "rlm-final-answer-box-wrapped-historical-answer"
    assert prediction.indices == (1, 2, 3)


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


def test_aggregate_recovery_handles_removed_ground_truth_when_old_tp_is_zero() -> None:
    recovered = infer_metric_from_aggregate(
        metric=SampleMetric("x", 3, 0.0, 0.0, 0.0, False),
        historical_ground_truth_in_context=frozenset({1, 2}),
        corrected_ground_truth_in_context=frozenset({1}),
    )

    assert recovered is not None
    assert recovered.historical_true_positives == (0,)
    assert recovered.corrected_true_positives == 0
    assert recovered.corrected_f1 == 0.0


def test_aggregate_recovery_handles_added_ground_truth_at_old_precision_one() -> None:
    recovered = infer_metric_from_aggregate(
        metric=SampleMetric("x", 1, 1.0, 0.5, 2 / 3, False),
        historical_ground_truth_in_context=frozenset({1, 2}),
        corrected_ground_truth_in_context=frozenset({1, 2, 3}),
    )

    assert recovered is not None
    assert recovered.corrected_true_positives == 1
    assert recovered.corrected_f1 == 0.5


def test_aggregate_recovery_rejects_ambiguous_removed_positive_membership() -> None:
    recovered = infer_metric_from_aggregate(
        metric=SampleMetric("x", 1, 1.0, 0.5, 2 / 3, False),
        historical_ground_truth_in_context=frozenset({1, 2}),
        corrected_ground_truth_in_context=frozenset({1}),
    )

    assert recovered is None


def test_invalid_extracted_candidate_falls_back_to_determined_prediction() -> None:
    score, question_scores = score_tier3_run(
        task="tier3/task18",
        context_by_question={"rxh-t3-task18": frozenset({1, 2, 3})},
        historical_ground_truth={"rxh-t3-task18": frozenset({1, 2})},
        corrected_ground_truth={"rxh-t3-task18": frozenset({1})},
        metrics={
            "new_ring_construction": SampleMetric("new_ring_construction", 2, 1.0, 1.0, 1.0, True)
        },
        extracted={
            "rxh-t3-task18": ExtractedPrediction("rxh-t3-task18", (8, 9), "spurious-code-block")
        },
    )

    assert score == 2 / 3
    assert question_scores[0]["recovery"] == "prediction-equals-historical-ground-truth"


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


def test_artifact_tar_selects_dashboard_attempt_and_deduplicates_backups(tmp_path) -> None:
    path = tmp_path / "artifacts.tar"
    run_id = "full-model-tier3-task23-rlm-x100-r01"
    selected_url = "https://wandb.ai/entity/project/runs/selected"

    def add(archive, name, content):
        encoded = content.encode()
        info = tarfile.TarInfo(name)
        info.size = len(encoded)
        archive.addfile(info, BytesIO(encoded))

    with tarfile.open(path, "w") as archive:
        for root in ("artifacts/current", "artifacts/backup"):
            base = f"{root}/runs/{run_id}/attempt-001"
            metadata = {
                "run": {"run_id": run_id},
                "result": {
                    "status": "succeeded",
                    "metrics": {"wandb_url": selected_url},
                },
            }
            add(archive, f"{base}/metadata.json", json.dumps(metadata))
            add(archive, f"{base}/stdout.log", "preserved prediction\n")
        base = f"artifacts/current/runs/{run_id}/attempt-002"
        metadata = {
            "run": {"run_id": run_id},
            "result": {
                "status": "succeeded",
                "metrics": {"wandb_url": "https://wandb.ai/entity/project/runs/other"},
            },
        }
        add(archive, f"{base}/metadata.json", json.dumps(metadata))
        add(archive, f"{base}/stdout.log", "wrong attempt\n")

    logs, status = load_artifact_tar_logs(
        path=path,
        target_run_ids={run_id},
        wandb_urls={run_id: selected_url},
    )

    assert logs == {run_id: b"preserved prediction\n"}
    assert status == {run_id: "artifact-tar"}


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
