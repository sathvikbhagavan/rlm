from __future__ import annotations

import pytest

from paper_plots.scripts.build_gold_results import (
    add_arm_finality,
    arm_summaries,
    result_score,
    scaling_summaries,
)


def test_task15_uses_reaction_f1() -> None:
    metrics = {"results": {"macro_f1": 0.1, "macro_reaction_f1": 0.8}}

    assert result_score("tier4/task15", metrics) == ("macro_reaction_f1", 0.8)
    assert result_score("tier4/task14", metrics) == ("macro_f1", 0.1)


def test_terminal_failures_do_not_make_an_arm_provisional() -> None:
    rows = [
        {"scope": "full_benchmark", "model": "qwen3.5", "method": "rlm", "status": status}
        for status in ("succeeded", "failed")
    ]

    summary = arm_summaries(rows)[0]

    assert summary["is_final"] is True
    assert summary["failed_jobs"] == 1


def test_pending_or_running_jobs_mark_an_arm_provisional() -> None:
    rows = [
        {
            "scope": "full_benchmark",
            "model": "deepseek-v4-flash",
            "method": "codeact",
            "status": status,
        }
        for status in ("succeeded", "running", "pending")
    ]

    summary = arm_summaries(rows)[0]

    assert summary["is_final"] is False


def test_scaling_summary_matches_weighted_mean_and_repetition_std() -> None:
    rows = [
        {
            "scope": "full_benchmark",
            "model": "gpt-5-mini",
            "method": "llm",
            "context": "100",
            "tier": 2,
            "question_count": question_count,
            "repetition": repetition,
            "status": "succeeded",
            "f1": f1,
        }
        for repetition, f1 in ((1, 1.0), (2, 0.0))
        for question_count in (6, 4)
    ]
    arms = arm_summaries(rows)
    add_arm_finality(rows, arms)

    summary = scaling_summaries(rows)[0]

    assert summary["f1"] == pytest.approx(0.5)
    assert summary["f1_std"] == pytest.approx(0.5)
    assert summary["coverage"] == 1.0
    assert summary["arm_final"] is True
