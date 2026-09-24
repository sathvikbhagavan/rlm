from paper_plots.scripts.build_model_appendix import summarize_model


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


def test_summarize_model_scores_failures_zero_and_excludes_their_resources() -> None:
    rows = [
        record(repetition=1, status="succeeded", f1="0.5"),
        record(repetition=2, status="failed", f1=""),
    ]

    tier_rows, task_rows = summarize_model(rows, "test-model")

    assert len(tier_rows) == 1
    tier = tier_rows[0]
    assert tier["f1_mean"] == 0.25
    assert tier["f1_std"] == 0.25
    assert tier["calls_mean"] == 1.0
    assert tier["total_tokens_mean"] == 10.0
    assert tier["cost_usd_mean"] == 0.5
    assert tier["peak_combined_memory_mib_mean"] == 100.0
    assert tier["successful_jobs"] == 1
    assert tier["failed_jobs"] == 1

    assert len(task_rows) == 1
    assert task_rows[0]["f1_mean"] == 0.25
    assert task_rows[0]["successful_jobs"] == 1
    assert task_rows[0]["failed_jobs"] == 1


def test_summarize_model_rejects_unresolved_records() -> None:
    rows = [record(repetition=1, status="running", f1="")]

    try:
        summarize_model(rows, "test-model")
    except ValueError as error:
        assert "unresolved" in str(error)
    else:
        raise AssertionError("Expected unresolved model records to fail")
