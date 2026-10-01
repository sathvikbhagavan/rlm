from __future__ import annotations

import json
import random
import sqlite3
import tarfile
from io import BytesIO

import pytest

from paper_plots.scripts.recover_corrected_scores import (
    load_artifact_run_windows,
    load_artifact_tar_logs,
    load_targets,
)
from rlm.codeact_helpers import RandomContextPipeline
from rxnhaystack.phoenix_recovery import (
    PhoenixRunWindow,
    matching_session,
    root_chat_spans,
    select_prediction_candidate,
)
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
                "execution": {
                    "started_at": "2026-09-15T12:00:00+00:00",
                    "finished_at": "2026-09-15T12:01:00+00:00",
                },
                "run": {
                    "run_id": run_id,
                    "task": "tier3/task23",
                    "method": "rlm",
                    "model": "model-a",
                },
                "result": {
                    "status": "succeeded",
                    "metrics": {"wandb_url": selected_url},
                },
            }
            add(archive, f"{base}/metadata.json", json.dumps(metadata))
            add(archive, f"{base}/stdout.log", "preserved prediction\n")
        base = f"artifacts/current/runs/{run_id}/attempt-002"
        metadata = {
            "execution": {
                "started_at": "2026-09-15T13:00:00+00:00",
                "finished_at": "2026-09-15T13:01:00+00:00",
            },
            "run": {
                "run_id": run_id,
                "task": "tier3/task23",
                "method": "rlm",
                "model": "model-a",
            },
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

    windows = load_artifact_run_windows(
        path=path,
        target_run_ids={run_id},
        wandb_urls={run_id: selected_url},
    )
    assert windows[run_id] == PhoenixRunWindow(
        run_id=run_id,
        task="tier3/task23",
        method="rlm",
        model="model-a",
        started_at="2026-09-15T12:00:00+00:00",
        finished_at="2026-09-15T12:01:00+00:00",
    )


def test_phoenix_prediction_selection_uses_latest_count_validated_answer() -> None:
    direct = {"output": {"value": json.dumps({"choices": [{"message": {"content": "9, 4, 2"}}]})}}
    assert select_prediction_candidate(
        method="llm", root_chat_attributes=[direct], expected_count=3
    )[:2] == ((9, 4, 2), "phoenix-direct-output")

    rlm = {
        "llm": {
            "input_messages": [
                {"message": {"role": "user", "content": "Answer: 1,2"}},
                {
                    "message": {
                        "role": "user",
                        "content": "REPL output:\nAll indices: 7, 3, 5",
                    }
                },
            ]
        }
    }
    selected = select_prediction_candidate(
        method="rlm", root_chat_attributes=[rlm], expected_count=3
    )
    assert selected is not None
    assert selected[:2] == ((7, 3, 5), "phoenix-rlm-message-1-answer-line")


def test_phoenix_session_matching_is_strict_and_root_depth_only(tmp_path) -> None:
    database = tmp_path / "phoenix.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE project_sessions (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            project_id INTEGER NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL
        );
        CREATE TABLE traces (
            id INTEGER PRIMARY KEY,
            project_session_rowid INTEGER NOT NULL
        );
        CREATE TABLE spans (
            id INTEGER PRIMARY KEY,
            trace_rowid INTEGER NOT NULL,
            span_id TEXT NOT NULL,
            name TEXT NOT NULL,
            span_kind TEXT NOT NULL,
            start_time TEXT NOT NULL,
            attributes TEXT NOT NULL
        );
        INSERT INTO projects VALUES (1, 'RLMs-Task18_tier3');
        INSERT INTO project_sessions VALUES (
            1, 'session-a', 1,
            '2026-09-15 12:00:05', '2026-09-15 12:00:55'
        );
        INSERT INTO traces VALUES (1, 1);
        """
    )
    root_attributes = {
        "metadata": {"depth": 0},
        "llm": {"model_name": "model-a", "input_messages": []},
    }
    child_attributes = {
        "metadata": {"depth": 1},
        "llm": {"model_name": "model-a", "input_messages": []},
    }
    connection.executemany(
        "INSERT INTO spans VALUES (?, 1, ?, 'ChatCompletion', 'LLM', ?, ?)",
        [
            (1, "root", "2026-09-15 12:00:10", json.dumps(root_attributes)),
            (2, "child", "2026-09-15 12:00:20", json.dumps(child_attributes)),
        ],
    )
    connection.commit()

    window = PhoenixRunWindow(
        run_id="run-a",
        task="tier3/task18",
        method="rlm",
        model="model-a",
        started_at="2026-09-15T12:00:00+00:00",
        finished_at="2026-09-15T12:01:00+00:00",
    )
    assert matching_session(connection, window=window) == (
        "session-a",
        "RLMs-Task18_tier3",
    )
    assert [
        span_id for span_id, _attributes in root_chat_spans(connection, session_id="session-a")
    ] == ["root"]

    wrong_model = PhoenixRunWindow(**{**window.__dict__, "model": "model-b"})
    assert matching_session(connection, window=wrong_model) is None


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
