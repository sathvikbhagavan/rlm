import pytest

from paper_plots.scripts.build_model_appendix import TASK_LABELS, summarize_model


def record(*, repetition: int, status: str, f1: str) -> dict[str, str]:
    successful = status == "succeeded"
    return {
        "model": "test-model",
        "method": "llm",
        "context": "100",
        "tier": "1",
        "task": "tier1/task1",
        "repetition": str(repetition),
        "question_count": "2",
        "status": status,
        "f1": f1,
        "calls": "2" if successful else "",
        "cost_usd": "1" if successful else "",
        "total_tokens": "20" if successful else "",
        "latency_seconds": "4" if successful else "",
        "tool_time_seconds": "0" if successful else "",
        "process_wall_time_seconds": "6" if successful else "",
        "peak_combined_memory_mib": "100" if successful else "",
    }


def complete_tier_one(*, repetition: int, status: str, f1: str) -> list[dict[str, str]]:
    row = record(repetition=repetition, status=status, f1=f1)
    row["question_count"] = "10"
    return [row]


def test_summarize_model_scores_failures_zero_and_excludes_their_resources() -> None:
    rows = [
        *complete_tier_one(repetition=1, status="succeeded", f1="0.5"),
        *complete_tier_one(repetition=2, status="failed", f1=""),
    ]

    tier_rows, task_rows = summarize_model(rows, "test-model")

    assert len(tier_rows) == 1
    tier = tier_rows[0]
    assert tier["f1_mean"] == 0.25
    assert tier["f1_std"] == 0.25
    assert tier["calls_mean"] == 0.2
    assert tier["total_tokens_mean"] == 2.0
    assert tier["cost_usd_mean"] == 0.1
    assert tier["peak_combined_memory_mib_mean"] == 100.0
    assert tier["successful_jobs"] == 1
    assert tier["failed_jobs"] == 1

    assert len(task_rows) == 1
    assert task_rows[0]["f1_mean"] == 0.25
    assert task_rows[0]["successful_jobs"] == 1
    assert task_rows[0]["failed_jobs"] == 1


def test_summarize_model_rejects_unresolved_records() -> None:
    rows = complete_tier_one(repetition=1, status="running", f1="")

    try:
        summarize_model(rows, "test-model")
    except ValueError as error:
        assert "unresolved" in str(error)
    else:
        raise AssertionError("Expected unresolved model records to fail")


def test_summarize_model_rejects_incomplete_task_coverage() -> None:
    rows = complete_tier_one(repetition=1, status="succeeded", f1="1")
    rows[0]["task"] = "tier1/missing"

    with pytest.raises(ValueError, match="Incomplete task coverage"):
        summarize_model(rows, "test-model")


def test_all_tiers_define_the_expected_hundred_questions() -> None:
    counts = {tier: 0 for tier in range(1, 5)}
    expected_per_task = {
        "tier1/task1": 10,
        "tier2/task2": 6,
        "tier2/task3": 5,
        "tier2/task4": 5,
        "tier2/task5": 4,
        "tier3/task13": 1,
        "tier3/task14": 1,
        "tier3/task15": 1,
        "tier3/task17": 1,
        "tier3/task18": 1,
        "tier3/task20": 1,
        "tier3/task21": 1,
        "tier3/task22": 1,
        "tier3/task23": 1,
        "tier3/task24": 1,
        "tier3/task6": 4,
        "tier3/task7": 5,
        "tier3/task8": 2,
        "tier3/task9": 4,
        "tier3/task10": 5,
        "tier3/task10b": 5,
        "tier4/task11": 2,
        "tier4/task12": 2,
        "tier4/task12b": 1,
        "tier4/task13": 4,
        "tier4/task14": 2,
        "tier4/task15": 4,
        "tier4/task16": 10,
        "tier4/task17": 5,
        "tier4/task17b": 5,
    }
    assert set(expected_per_task) == set(TASK_LABELS)
    for task, count in expected_per_task.items():
        counts[int(task[4])] += count
    assert counts == {1: 10, 2: 20, 3: 35, 4: 35}
