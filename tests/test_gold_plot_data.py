from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from paper_plots.scripts.build_causal_controls import (
    apply_ground_truth_corrections as apply_control_ground_truth_corrections,
)
from paper_plots.scripts.build_gold_results import (
    add_arm_finality,
    apply_ground_truth_corrections,
    apply_score_recoveries,
    arm_summaries,
    cross_model_efficiency_summaries,
    cross_model_scaling_summaries,
    load_result_pack,
    result_score,
    scaling_summaries,
    tier_efficiency_summaries,
)
from paper_plots.scripts.build_post_submission_queue import build_queue
from rxnhaystack.score_recovery import carry_forward_pending_corrected_scores


def test_task15_uses_reaction_f1() -> None:
    metrics = {"results": {"macro_f1": 0.1, "macro_reaction_f1": 0.8}}

    assert result_score("tier4/task15", metrics) == ("macro_reaction_f1", 0.8)
    assert result_score("tier4/task14", metrics) == ("macro_f1", 0.1)


def test_main_capability_matrix_weights_questions_and_scores_failures_zero() -> None:
    pytest.importorskip("matplotlib")
    from paper_plots.scripts.plot_main_results import capability_matrix

    rows = [
        {
            "model": "gpt-5-mini",
            "tier": "3",
            "task": task,
            "method": "rlm",
            "context": "full",
            "repetition": str(repetition),
            "question_count": question_count,
            "status": status,
            "f1": f1,
        }
        for repetition in (1, 2)
        for task, question_count, status, f1 in (
            ("tier3/task13", "1", "succeeded", "1.0"),
            ("tier3/task14", "3", "failed", ""),
        )
    ]

    matrix, labels = capability_matrix(rows, 3)

    assert labels[0] == "Bond-level"
    assert matrix[0, 6] == pytest.approx(0.25)


def test_human_tier_reference_requires_one_row_per_tier() -> None:
    pytest.importorskip("matplotlib")
    from paper_plots.scripts.plot_main_results import human_tier_f1

    rows = [
        {
            "population": "human_assigned_items",
            "tier": str(tier),
            "mean_f1": str(tier / 4),
        }
        for tier in range(1, 5)
    ]

    assert human_tier_f1(rows) == {1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
    with pytest.raises(ValueError, match="one human-assigned result"):
        human_tier_f1(rows[:-1])


def test_human_tier_summary_keeps_active_and_offline_time_separate() -> None:
    pytest.importorskip("matplotlib")
    from paper_plots.scripts.plot_main_results import human_tier_summary

    rows = [
        {
            "tier": str(tier),
            "submitted": "True",
            "abstention": "",
            "f1": "0.5",
            "exact_match": "0.0",
            "active": "120",
            "offline_minutes": "5",
        }
        for tier in range(1, 5)
    ]

    summary = human_tier_summary(rows)

    assert summary[1]["f1"] == 0.5
    assert summary[1]["median_active_minutes"] == 2.0
    assert summary[1]["median_offline_minutes"] == 5.0


def test_efficiency_frontier_uses_usd_per_trajectory_and_scores_failures_zero() -> None:
    pytest.importorskip("matplotlib")
    from paper_plots.scripts.plot_main_results import efficiency_points

    rows = [
        {
            "model": "gpt-5-mini",
            "tier": "4",
            "method": "rlm",
            "context": "full",
            "repetition": "1",
            "question_count": "3",
            "status": "succeeded",
            "f1": "0.75",
            "cost_usd": "0.30",
        },
        {
            "model": "gpt-5-mini",
            "tier": "4",
            "method": "rlm",
            "context": "full",
            "repetition": "1",
            "question_count": "1",
            "status": "failed",
            "f1": "",
            "cost_usd": "0.10",
        },
    ]

    cost, f1, f1_std = efficiency_points(rows)[(4, "rlm", "full")]

    assert cost == pytest.approx(0.10)
    assert f1 == pytest.approx(0.5625)
    assert f1_std == 0.0


def test_aggregate_efficiency_frontier_uses_only_paid_models() -> None:
    pytest.importorskip("matplotlib")
    from paper_plots.scripts.plot_main_results import aggregate_efficiency_points

    rows = []
    for model, score, cost in (
        ("gemini-3.7-flash", 0.3, 0.03),
        ("gpt-5-mini", 0.6, 0.06),
        ("claude-haiku-4.5", 0.9, 0.09),
        ("qwen3.5", 1.0, 0.0),
    ):
        rows.append(
            {
                "model": model,
                "tier": "3",
                "method": "rlm",
                "context": "full",
                "repetition": "1",
                "question_count": "1",
                "status": "succeeded",
                "f1": str(score),
                "cost_usd": str(cost),
            }
        )

    cost, _, f1, _, model_count = aggregate_efficiency_points(rows)[(3, "rlm", "full")]

    assert cost == pytest.approx(0.06)
    assert f1 == pytest.approx(0.6)
    assert model_count == 3


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


def test_score_recovery_only_fills_an_unscored_success(tmp_path) -> None:
    rows = [
        {
            "run_id": "legacy-run",
            "status": "succeeded",
            "score_available": False,
            "score_name": "macro_f1",
            "f1": None,
            "sources": "legacy-ledger",
        }
    ]
    path = tmp_path / "recoveries.json"
    path.write_text(
        json.dumps(
            {
                "recoveries": [
                    {
                        "run_id": "legacy-run",
                        "score_name": "macro_f1",
                        "score_value": 0.625,
                    }
                ]
            }
        )
    )

    apply_score_recoveries(rows, path)

    assert rows[0]["score_available"] is True
    assert rows[0]["f1"] == 0.625
    assert rows[0]["sources"].endswith("score-recovery:recoveries.json")


def test_ground_truth_correction_invalidates_score_but_preserves_provenance(tmp_path) -> None:
    rows = [
        {
            "scope": "full_benchmark",
            "run_id": "old-run",
            "model": "gpt-5-mini",
            "method": "llm",
            "task": "tier3/task10",
            "status": "succeeded",
            "score_available": True,
            "f1": 0.75,
            "sources": "immutable-ledger",
        }
    ]
    path = tmp_path / "corrections.json"
    path.write_text(
        json.dumps(
            {
                "correction_id": "test-correction",
                "corrected_bundle": "test-bundle",
                "affected_tasks": {"tier3/task10": "test reason"},
            }
        )
    )

    apply_ground_truth_corrections(rows, path)

    assert rows[0]["original_f1"] == 0.75
    assert rows[0]["f1"] is None
    assert rows[0]["score_available"] is False
    assert rows[0]["score_correction_status"] == "historical_score_invalidated"
    assert carry_forward_pending_corrected_scores(rows) == 1
    assert rows[0]["f1"] == 0.75
    assert rows[0]["score_available"] is True
    assert rows[0]["score_correction_status"] == "historical_score_invalidated"
    summary = arm_summaries(rows)[0]
    assert summary["is_final"] is True
    assert summary["scored_success_jobs"] == 1
    assert summary["ground_truth_invalidated_jobs"] == 1


def test_control_ground_truth_correction_invalidates_only_affected_scores(tmp_path) -> None:
    rows = [
        {
            "task": task,
            "f1": 0.75,
            "score_available": True,
            "sources": "immutable-control-ledger",
        }
        for task in ("tier3/task10", "tier2/task2")
    ]
    path = tmp_path / "corrections.json"
    path.write_text(
        json.dumps(
            {
                "correction_id": "test-correction",
                "corrected_bundle": "test-bundle",
                "affected_tasks": {"tier3/task10": "test reason"},
            }
        )
    )

    apply_control_ground_truth_corrections(rows, path)

    assert rows[0]["original_f1"] == 0.75
    assert rows[0]["f1"] == ""
    assert rows[0]["score_available"] is False
    assert rows[0]["score_correction_status"] == "historical_score_invalidated"
    assert rows[1]["f1"] == 0.75
    assert rows[1]["score_available"] is True
    assert rows[1]["score_correction_status"] == "not_affected"
    assert carry_forward_pending_corrected_scores(rows) == 1
    assert rows[0]["f1"] == 0.75
    assert rows[0]["score_available"] is True


def test_post_submission_queue_freezes_experiment_arm_model_seed_and_repetition(
    tmp_path: Path,
) -> None:
    gold = tmp_path / "gold"
    (gold / "causal_controls").mkdir(parents=True)
    run_id = "full-gpt-5-mini-tier3-task18-rlm-xfull-r03"
    main_header = (
        "scope,run_id,model,model_label,method,tier,task,context,repetition,"
        "question_count,score_name,original_f1,score_correction_status\n"
    )
    (gold / "full_benchmark_records.csv").write_text(
        main_header + f"full_benchmark,{run_id},gpt-5-mini,GPT-5 mini,rlm,3,tier3/task18,"
        "full,3,1,macro_f1,0.5,historical_score_invalidated\n"
    )
    for name in ("codeact_x1000_records.csv", "rlm_x1000_records.csv"):
        (gold / name).write_text(main_header)
    control_header = (
        "study,arm,model,model_label,context,condition,tier,task,repetition,"
        "question_count,original_f1,score_correction_status,run_id\n"
    )
    (gold / "causal_controls/records.csv").write_text(
        control_header + "oracle_predicate,ordinary,openai/gpt-5-mini,GPT-5 mini,full,ordinary,3,"
        f"tier3/task18,3,1,0.5,historical_score_invalidated,{run_id}\n"
    )
    (gold / "causal_controls/matched_qwen_terminal.csv").write_text(control_header)
    recovery = {
        "recovery_id": "recovery-v2",
        "recoveries": [],
        "unresolved": [
            {
                "run_id": run_id,
                "reason": "prediction-underdetermined-or-validation-failed",
                "source_sha256": "abc",
                "wandb_url": "https://wandb.ai/liac/project/runs/1",
            }
        ],
    }

    rows = build_queue(recovery=recovery, gold_dir=gold, configured_seed=42)

    assert len(rows) == 1
    assert rows[0]["model"] == "gpt-5-mini"
    assert rows[0]["configured_seed"] == 42
    assert rows[0]["repetition"] == 3
    assert rows[0]["experiments"] == "causal_control;main_benchmark"
    assert rows[0]["arms"] == ("full_benchmark/rlm/xfull;oracle_predicate/ordinary/ordinary/xfull")


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

    assert summary["mean_f1"] == pytest.approx(0.6)
    assert summary["model_std"] == pytest.approx(0.5477225575)
    assert summary["model_sem"] == pytest.approx(0.2449489743)
    assert summary["n_models"] == 5
    assert "glm-5.2" not in summary["included_models"]
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
    assert summary["target_n_models"] == 4
    assert summary["missing_models"] == "qwen3.5;gemini-3.7-flash;gpt-5-mini"
    assert summary["excluded_provisional_models"] == "gemini-3.7-flash"
    assert summary["is_final"] is False


def test_tier_efficiency_is_normalized_per_successful_trajectory() -> None:
    rows = [
        {
            "model": "gpt-5-mini",
            "model_label": "GPT-5 mini",
            "method": "rlm",
            "context": "full",
            "tier": 4,
            "status": "succeeded",
            "score_available": True,
            "question_count": 2,
            "cost_chf": 3.0,
            "total_tokens": 300,
            "process_wall_time_seconds": 30.0,
            "arm_final": True,
        },
        {
            "model": "gpt-5-mini",
            "model_label": "GPT-5 mini",
            "method": "rlm",
            "context": "full",
            "tier": 4,
            "status": "succeeded",
            "score_available": True,
            "question_count": 1,
            "cost_chf": 3.0,
            "total_tokens": 600,
            "process_wall_time_seconds": 60.0,
            "arm_final": True,
        },
        {
            "model": "gpt-5-mini",
            "model_label": "GPT-5 mini",
            "method": "rlm",
            "context": "full",
            "tier": 4,
            "status": "failed",
            "score_available": False,
            "question_count": 5,
            "cost_chf": None,
            "total_tokens": None,
            "process_wall_time_seconds": None,
            "arm_final": True,
        },
    ]

    summary = tier_efficiency_summaries(rows)[0]

    assert summary["cost_chf_per_trajectory"] == 2.0
    assert summary["tokens_per_trajectory"] == 300.0
    assert summary["wall_time_seconds_per_trajectory"] == 30.0
    assert summary["failed_jobs"] == 1
    assert summary["cost_chf_per_trajectory_coverage"] == 1.0


def test_cross_model_cost_average_excludes_free_models() -> None:
    rows = []
    for model, cost in (
        ("qwen3.5", 0.0),
        ("gemini-3.7-flash", 2.0),
        ("gpt-5-mini", 4.0),
        ("claude-haiku-4.5", 6.0),
    ):
        rows.append(
            {
                "model": model,
                "method": "llm",
                "context": "100",
                "tier": 1,
                "arm_final": True,
                "cost_chf_per_trajectory": cost,
                "cost_chf_per_trajectory_coverage": 1.0,
                "tokens_per_trajectory": 100.0,
                "tokens_per_trajectory_coverage": 1.0,
                "wall_time_seconds_per_trajectory": 10.0,
                "wall_time_seconds_per_trajectory_coverage": 1.0,
            }
        )

    summary = cross_model_efficiency_summaries(rows)[0]

    assert summary["mean_cost_chf_per_trajectory"] == 4.0
    assert summary["n_paid_models"] == 3
    assert summary["included_paid_models"] == ("gemini-3.7-flash;gpt-5-mini;claude-haiku-4.5")
    assert summary["n_models"] == 4
