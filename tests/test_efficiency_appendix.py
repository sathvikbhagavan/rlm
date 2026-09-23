from __future__ import annotations

import pytest

from paper_plots.scripts.build_efficiency_appendix import (
    across_model_summaries,
    per_model_summaries,
)


def test_efficiency_normalizes_additive_metrics_but_not_peak_memory() -> None:
    rows = [
        {
            "model": "gpt-5-mini",
            "method": "rlm",
            "context": "full",
            "tier": 4,
            "status": "succeeded",
            "score_available": True,
            "question_count": 2,
            "calls": 9,
            "total_tokens": 300,
            "latency_seconds": 12,
            "tool_time_seconds": 3,
            "process_wall_time_seconds": 30,
            "peak_combined_memory_mib": 100,
            "arm_final": True,
        },
        {
            "model": "gpt-5-mini",
            "method": "rlm",
            "context": "full",
            "tier": 4,
            "status": "succeeded",
            "score_available": True,
            "question_count": 1,
            "calls": 6,
            "total_tokens": 600,
            "latency_seconds": 18,
            "tool_time_seconds": 6,
            "process_wall_time_seconds": 60,
            "peak_combined_memory_mib": 200,
            "arm_final": True,
        },
        {
            "model": "gpt-5-mini",
            "method": "rlm",
            "context": "full",
            "tier": 4,
            "status": "failed",
            "score_available": False,
            "question_count": 5,
            "arm_final": True,
        },
    ]

    summary = per_model_summaries(rows)[0]

    assert summary["calls_per_trajectory"] == pytest.approx(5.0)
    assert summary["tokens_per_trajectory"] == pytest.approx(300.0)
    assert summary["latency_seconds_per_trajectory"] == pytest.approx(10.0)
    assert summary["tool_time_seconds_per_trajectory"] == pytest.approx(3.0)
    assert summary["wall_time_seconds_per_trajectory"] == pytest.approx(30.0)
    assert summary["peak_memory_mib_per_job"] == pytest.approx(150.0)
    assert summary["failed_jobs"] == 1
    assert summary["calls_per_trajectory_coverage"] == 1.0


def test_cross_model_summary_is_unweighted_and_preserves_failure_count() -> None:
    model_rows = []
    for model, calls, failures in (
        ("qwen3.5", 2.0, 0),
        ("gpt-5-mini", 6.0, 1),
    ):
        row = {
            "model": model,
            "method": "rlm",
            "context": "full",
            "tier": 3,
            "successful_jobs": 10,
            "successful_trajectories": 10,
            "failed_jobs": failures,
        }
        for metric in (
            "calls_per_trajectory",
            "tokens_per_trajectory",
            "latency_seconds_per_trajectory",
            "tool_time_seconds_per_trajectory",
            "wall_time_seconds_per_trajectory",
            "peak_memory_mib_per_job",
        ):
            row[metric] = calls
            row[f"{metric}_coverage"] = 1.0
        model_rows.append(row)

    summary = across_model_summaries(model_rows)[0]

    assert summary["mean_calls_per_trajectory"] == pytest.approx(4.0)
    assert summary["n_models"] == 2
    assert summary["failed_jobs"] == 1
    assert summary["minimum_calls_per_trajectory_coverage"] == 1.0
