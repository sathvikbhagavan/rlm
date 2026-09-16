from __future__ import annotations

import json

from rxnhaystack.ledger import RunLedger
from rxnhaystack.manifest import PlannedRun
from rxnhaystack.recovery import audit_missing_usage, render_dry_run


def failed_run(tmp_path, run_id: str) -> tuple[RunLedger, PlannedRun, object]:
    run = PlannedRun(
        run_id=run_id,
        base_id=run_id,
        campaign="recovery-test",
        task="tier/test",
        condition="test",
        method="rlm",
        model="openai/gpt-5-mini",
        corpus_size=100,
        positive_cardinality=None,
        seed=42,
        repetition=1,
        command=("python", "test.py"),
        env={},
        estimated_cost_chf=1.0,
    )
    root = tmp_path / "artifacts"
    ledger = RunLedger(root / "ledger.sqlite3")
    ledger.sync_runs([run], manifest_sha256="a" * 64)
    ledger.claim(run_id)
    attempt_dir = root / "runs" / run_id / "attempt-001"
    attempt_dir.mkdir(parents=True)
    ledger.finish(run_id, return_code=1, artifact_dir=attempt_dir, error="missing usage")
    return ledger, run, attempt_dir


def test_dry_run_reports_partial_trajectory_job_without_writing_ledger(tmp_path) -> None:
    ledger, _run, attempt_dir = failed_run(tmp_path, "partial-run")
    wandb_root = tmp_path / "wandb"
    config_dir = wandb_root / "run-20260101_000000-abcd" / "files"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text("num_questions:\n    value: 3\n")
    (attempt_dir / "stderr.log").write_text(
        "saved in /work/wandb/run-20260101_000000-abcd\n"
        "No usage data received. Tracking tokens not possible.\n"
    )
    (attempt_dir / "stdout.log").write_text(
        "Question 1/3\nPredicted [x] count: 1\nMetrics [x] -> f1=1.0\n"
    )
    (attempt_dir / "resource-trace.jsonl").write_text(
        json.dumps({"event": "rlm_completion_metrics", "cost_usd": None}) + "\n"
    )
    before = ledger.path.read_bytes()

    audits = audit_missing_usage(
        ledger.path,
        artifact_root=tmp_path / "artifacts",
        wandb_root=wandb_root,
    )

    assert len(audits) == 1
    assert audits[0].expected_trajectories == 3
    assert audits[0].computed_scores == 1
    assert audits[0].recoverability == "partially_recoverable"
    assert audits[0].recovered_cost_usd is None
    assert ledger.path.read_bytes() == before
    report = json.loads(render_dry_run(audits))
    assert report["jobs"]["partially_recoverable"] == 1
    assert report["accounting"]["recovered_cost_usd"] is None


def test_dry_run_ignores_non_metadata_failure(tmp_path) -> None:
    ledger, run, attempt_dir = failed_run(tmp_path, "memory-run")
    (attempt_dir / "stderr.log").write_text("cannot allocate memory for thread-local data: ABORT\n")
    (attempt_dir / "stdout.log").write_text("")

    assert audit_missing_usage(
        ledger.path, artifact_root=tmp_path / "artifacts"
    ) == []
