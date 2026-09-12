from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rxnhaystack.ledger import LedgerError, RunLedger, validate_metrics
from rxnhaystack.manifest import ManifestError, PlannedRun


@pytest.fixture
def planned_run() -> PlannedRun:
    return PlannedRun(
        run_id="test-run",
        base_id="test-run",
        campaign="test-campaign",
        task="tier1/task1",
        condition="baseline",
        method="rlm",
        model="test-model",
        corpus_size=100,
        positive_cardinality=1,
        seed=42,
        repetition=1,
        command=("python", "task.py"),
        env={},
        estimated_cost_chf=1.5,
    )


def test_ledger_runs_successful_state_machine(tmp_path: Path, planned_run: PlannedRun) -> None:
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    ledger.sync_runs([planned_run], manifest_sha256="a" * 64)

    assert ledger.get(planned_run.run_id).status == "pending"
    assert ledger.claim(planned_run.run_id) == 1
    assert ledger.claim(planned_run.run_id) is None

    artifact_dir = tmp_path / "artifacts"
    ledger.finish(
        planned_run.run_id,
        return_code=0,
        artifact_dir=artifact_dir,
        metrics={"cost_chf": 0.5, "calls": 2},
    )
    record = ledger.get(planned_run.run_id)
    assert record.status == "succeeded"
    assert record.attempts == 1
    assert record.return_code == 0
    assert record.metrics == {"calls": 2, "cost_chf": 0.5}
    assert ledger.claim(planned_run.run_id, retry_failed=True) is None
    attempts = ledger.list_attempts(campaign="test-campaign")
    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
    assert attempts[0].metrics == {"calls": 2, "cost_chf": 0.5}


def test_ledger_retries_failed_run_without_overwriting_attempt(
    tmp_path: Path, planned_run: PlannedRun
) -> None:
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    ledger.sync_runs([planned_run], manifest_sha256="a" * 64)
    assert ledger.claim(planned_run.run_id) == 1
    ledger.finish(
        planned_run.run_id,
        return_code=7,
        artifact_dir=tmp_path / "attempt-001",
        error="command failed",
    )

    assert ledger.claim(planned_run.run_id) is None
    assert ledger.claim(planned_run.run_id, retry_failed=True) == 2
    ledger.finish(
        planned_run.run_id,
        return_code=0,
        artifact_dir=tmp_path / "attempt-002",
    )

    record = ledger.get(planned_run.run_id)
    assert record.status == "succeeded"
    assert record.attempts == 2
    attempts = ledger.list_attempts()
    assert [attempt.status for attempt in attempts] == ["failed", "succeeded"]
    assert [attempt.attempt for attempt in attempts] == [1, 2]


def test_ledger_requires_explicit_recovery_for_interrupted_run(
    tmp_path: Path, planned_run: PlannedRun
) -> None:
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    ledger.sync_runs([planned_run], manifest_sha256="a" * 64)
    assert ledger.claim(planned_run.run_id) == 1

    assert ledger.claim(planned_run.run_id) is None
    assert ledger.claim(planned_run.run_id, recover_running=True) == 2
    attempts = ledger.list_attempts()
    assert [attempt.status for attempt in attempts] == ["failed", "running"]
    assert "explicit recovery" in (attempts[0].error or "")


def test_ledger_rejects_changed_spec_for_existing_id(
    tmp_path: Path, planned_run: PlannedRun
) -> None:
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    ledger.sync_runs([planned_run], manifest_sha256="a" * 64)
    changed = replace(planned_run, model="different-model")

    with pytest.raises(LedgerError, match="different specification"):
        ledger.sync_runs([changed], manifest_sha256="b" * 64)


def test_ledger_rejects_finish_without_claim(tmp_path: Path, planned_run: PlannedRun) -> None:
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    ledger.sync_runs([planned_run], manifest_sha256="a" * 64)

    with pytest.raises(LedgerError, match="expected 'running'"):
        ledger.finish(planned_run.run_id, return_code=0, artifact_dir=tmp_path)


@pytest.mark.parametrize(
    "metrics",
    [
        [],
        {"calls": -1},
        {"calls": 1.5},
        {"cost_chf": True},
        {"latency_seconds": "slow"},
        {"wandb_url": ""},
    ],
)
def test_validate_metrics_rejects_invalid_contract(metrics: object) -> None:
    with pytest.raises(ManifestError):
        validate_metrics(metrics)


def test_validate_metrics_allows_required_efficiency_fields() -> None:
    metrics = {
        "calls": 3,
        "input_tokens": 100,
        "output_tokens": 25,
        "total_tokens": 125,
        "latency_seconds": 2.5,
        "tool_time_seconds": 0.5,
        "cost_usd": 0.01,
        "cost_chf": 0.008,
        "wandb_url": "https://wandb.ai/entity/project/runs/id",
        "macro_f1": 0.75,
    }

    assert validate_metrics(metrics, require_complete=True) == metrics


def test_validate_metrics_can_require_complete_efficiency_record() -> None:
    with pytest.raises(ManifestError, match="missing required fields"):
        validate_metrics({"calls": 1}, require_complete=True)
