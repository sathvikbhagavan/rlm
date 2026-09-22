from __future__ import annotations

import io
import json
import tarfile

import pytest

from paper_plots.scripts.build_gold_results import (
    add_arm_finality,
    arm_summaries,
    cross_model_scaling_summaries,
    load_result_pack,
    result_score,
    scaling_summaries,
)


def test_task15_uses_reaction_f1() -> None:
    metrics = {"results": {"macro_f1": 0.1, "macro_reaction_f1": 0.8}}

    assert result_score("tier4/task15", metrics) == ("macro_reaction_f1", 0.8)
    assert result_score("tier4/task14", metrics) == ("macro_f1", 0.1)


def test_terminal_failures_do_not_make_an_arm_provisional() -> None:
    rows = [
        {
            "scope": "full_benchmark",
            "model": "qwen3.5",
            "method": "rlm",
            "status": status,
            "score_available": status == "succeeded",
        }
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
            "score_available": status == "succeeded",
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
            "score_available": True,
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


def test_scaling_summary_scores_failures_zero_and_excludes_pending() -> None:
    rows = [
        {
            "scope": "full_benchmark",
            "model": "gpt-5-mini",
            "method": "codeact",
            "context": "100",
            "tier": 3,
            "question_count": 1,
            "repetition": repetition,
            "status": status,
            "score_available": status == "succeeded",
            "f1": 1.0 if status == "succeeded" else None,
        }
        for repetition, status in enumerate(("succeeded", "failed", "pending"), start=1)
    ]
    arms = arm_summaries(rows)
    add_arm_finality(rows, arms)

    summary = scaling_summaries(rows)[0]

    assert summary["f1_success_only"] == 1.0
    assert summary["f1"] == 0.5
    assert summary["f1_std"] == 0.5
    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["successful_coverage"] == pytest.approx(1 / 3)
    assert summary["f1_all_unresolved_zero"] == pytest.approx(1 / 3)
    assert summary["f1_best_case"] == pytest.approx(2 / 3)
    assert summary["arm_final"] is False


def test_success_without_a_scientific_score_is_provisional() -> None:
    rows = [
        {
            "scope": "full_benchmark",
            "model": "deepseek-v4-flash",
            "method": "codeact",
            "status": "succeeded",
            "score_available": False,
        }
    ]

    summary = arm_summaries(rows)[0]

    assert summary["is_final"] is False
    assert summary["unscored_success_jobs"] == 1


def test_external_x1000_result_pack_is_validated_and_flattened(tmp_path) -> None:
    run_id = "full-gemini-3.7-flash-tier4-task15-codeact-x1000-r01"
    manifest = {
        "packed_at": "2026-09-22T13:42:48+00:00",
        "models": ["gemini-3.7-flash"],
        "n_runs": 1,
        "rule": "test pack",
    }
    runs = [
        {
            "run_id": run_id,
            "model": "gemini-3.7-flash",
            "tier": "4",
            "task": "15",
            "method": "codeact",
            "context": "1000",
            "repetition": "01",
            "status": "succeeded",
            "attempt": 2,
            "metrics.results.macro_f1": 0.1,
            "metrics.results.macro_reaction_f1": 0.75,
            "metrics.total_tokens": 123,
        }
    ]
    archive_path = tmp_path / "results.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in (("manifest.json", manifest), ("runs.json", runs)):
            data = json.dumps(payload).encode()
            info = tarfile.TarInfo(f"pack/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    rows, loaded_manifest = load_result_pack(
        archive_path,
        expected_model="gemini-3.7-flash",
        expected_run_ids={run_id},
    )

    assert loaded_manifest == manifest
    assert rows[0]["task"] == "tier4/task15"
    assert rows[0]["score_name"] == "macro_reaction_f1"
    assert rows[0]["f1"] == 0.75
    assert rows[0]["total_tokens"] == 123


def test_cross_model_summary_uses_model_mean_and_standard_error() -> None:
    rows = [
        {
            "model": model,
            "method": "llm",
            "context": "100",
            "tier": 1,
            "f1": f1,
            "f1_zero_imputed": f1,
            "arm_final": True,
        }
        for model, f1 in zip(
            (
                "qwen3.5",
                "deepseek-v4-flash",
                "glm-5.2",
                "gemini-3.7-flash",
                "gpt-5-mini",
                "claude-haiku-4.5",
            ),
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            strict=True,
        )
    ]

    summary = cross_model_scaling_summaries(rows)[0]

    assert summary["mean_f1"] == pytest.approx(0.5)
    assert summary["model_std"] == pytest.approx(0.5477225575)
    assert summary["model_sem"] == pytest.approx(0.2236067977)
    assert summary["n_models"] == 6
    assert summary["is_final"] is True


def test_cross_model_summary_records_missing_and_provisional_models() -> None:
    rows = [
        {
            "model": "deepseek-v4-flash",
            "method": "codeact",
            "context": "1000",
            "tier": 3,
            "f1": 0.5,
            "f1_zero_imputed": 0.25,
            "arm_final": True,
        },
        {
            "model": "gemini-3.7-flash",
            "method": "codeact",
            "context": "1000",
            "tier": 3,
            "f1": 0.7,
            "f1_zero_imputed": 0.7,
            "arm_final": False,
        },
    ]

    summary = cross_model_scaling_summaries(rows)[0]

    assert summary["mean_f1"] == pytest.approx(0.25)
    assert summary["model_sem"] is None
    assert summary["n_models"] == 1
    assert summary["target_n_models"] == 3
    assert summary["missing_models"] == "gemini-3.7-flash;gpt-5-mini"
    assert summary["excluded_provisional_models"] == "gemini-3.7-flash"
    assert summary["is_final"] is False
