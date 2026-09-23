from __future__ import annotations

import pytest

from paper_plots.scripts.build_capability_split import capability_summaries


def test_capability_split_weights_questions_and_scores_terminal_failure_zero() -> None:
    rows = [
        {
            "model": "gpt-5-mini",
            "method": "rlm",
            "context": "full",
            "task": "tier3/task13",
            "question_count": 1,
            "status": "succeeded",
            "f1": 1.0,
            "arm_final": True,
        },
        {
            "model": "gpt-5-mini",
            "method": "rlm",
            "context": "full",
            "task": "tier3/task14",
            "question_count": 3,
            "status": "failed",
            "f1": None,
            "arm_final": True,
        },
        {
            "model": "qwen3.5",
            "method": "rlm",
            "context": "full",
            "task": "tier3/task13",
            "question_count": 1,
            "status": "succeeded",
            "f1": 0.5,
            "arm_final": False,
        },
    ]

    per_model, across_models = capability_summaries(rows)

    assert per_model == [
        {
            "model": "gpt-5-mini",
            "capability_group": "bond_changes",
            "mean_f1": 0.25,
            "weighted_question_runs": 4,
            "terminal_failed_jobs": 1,
        }
    ]
    bond = across_models[0]
    assert bond["across_model_mean_f1"] == pytest.approx(0.25)
    assert bond["n_models"] == 1
