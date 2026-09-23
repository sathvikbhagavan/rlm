from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rxnhaystack.launcher import (
    cleanup_labeled_docker_containers,
    enforce_remaining_budget,
    run_selected,
)
from rxnhaystack.ledger import RunLedger
from rxnhaystack.manifest import ManifestError, load_manifest
from rxnhaystack.runtime import perform_preflight


def initialize_git_repository(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "tracked.txt").write_text("test\n")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "initial"], check=True)


def write_worker(
    path: Path,
    *,
    write_metrics: bool = True,
    exit_code: int = 0,
    cost_chf: float = 0.08,
    hold_seconds: float = 0.0,
) -> None:
    metrics_statement = (
        f"""
from rxnhaystack.metrics import RunMetrics, write_run_metrics
write_run_metrics(RunMetrics(
    calls=2,
    input_tokens=10,
    output_tokens=5,
    total_tokens=15,
    latency_seconds=0.1,
    tool_time_seconds=0.02,
    cost_usd=0.1,
    cost_chf={cost_chf},
    results={{"macro_f1": 0.75}},
))
"""
        if write_metrics
        else ""
    )
    path.write_text(
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        f"{metrics_statement}\n"
        "print('run=' + os.environ['RXNHAYSTACK_RUN_ID'])\n"
        "print('secret-present=' + str(bool(os.environ.get('TEST_API_KEY'))))\n"
        f"time.sleep({hold_seconds!r})\n"
        f"raise SystemExit({exit_code})\n"
    )


def write_campaign(
    path: Path,
    project_root: Path,
    worker: Path,
    *,
    repetitions: int = 2,
    budget_chf: float = 10,
    estimated_cost_chf: float = 1,
) -> Path:
    manifest = path / "campaign.toml"
    manifest.write_text(
        f"""
schema_version = 1

[campaign]
name = "integration"
project_root = {json.dumps(str(project_root))}
artifact_dir = {json.dumps(str(path / "artifacts"))}
budget_chf = {budget_chf}
usd_to_chf = 0.8
require_dataset = false
require_clean_git = true
require_metrics = true

[[runs]]
id = "integration-cell"
task = "tier1/task1"
condition = "smoke"
method = "deterministic"
model = "none"
corpus_size = 100
positive_cardinality = 1
seed = 42
repetitions = {repetitions}
estimated_cost_chf = {estimated_cost_chf}
command = [{json.dumps(sys.executable)}, {json.dumps(str(worker))}]
""".strip()
        + "\n"
    )
    return manifest


def test_launcher_executes_parallel_runs_records_artifacts_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    initialize_git_repository(project_root)
    worker = project_root / "worker.py"
    # Keep this synthetic worker alive long enough for the polling watchdog to
    # observe it even on a heavily loaded shared login node. Real benchmark
    # workers live for minutes or hours; a process that exits between spawn and
    # the first /proc sample cannot have its RSS reconstructed after the fact.
    write_worker(worker, hold_seconds=0.5)
    manifest = load_manifest(write_campaign(tmp_path, project_root, worker))
    preflight = perform_preflight(manifest)
    monkeypatch.setenv("RXNHAYSTACK_SWISSAI_HOST_REQUESTS_PER_MINUTE_CAP", "6")

    results = run_selected(
        manifest,
        preflight=preflight,
        selected=list(manifest.runs),
        secrets={"TEST_API_KEY": "never-record-this"},
        max_parallel=2,
        retry_failed=False,
        recover_running=False,
    )

    assert [result.status for result in results] == ["succeeded", "succeeded"]
    for result in results:
        assert result.artifact_dir is not None
        metadata = json.loads((result.artifact_dir / "metadata.json").read_text())
        metrics = json.loads((result.artifact_dir / "metrics.json").read_text())
        resource_events = [
            json.loads(line)
            for line in (result.artifact_dir / "resource-trace.jsonl").read_text().splitlines()
        ]
        stdout = (result.artifact_dir / "stdout.log").read_text()
        assert metadata["result"]["metrics"]["results"]["macro_f1"] == 0.75
        assert metadata["result"]["metrics"]["resources"]["peak_process_tree_rss_mib"] > 0
        assert metadata["result"]["metrics"]["resources"]["peak_combined_memory_mib"] > 0
        assert metadata["result"]["metrics"]["resources"]["peak_docker_memory_mib"] == 0
        assert metadata["execution"]["resources"]["process_wall_time_seconds"] > 0
        assert metadata["execution"]["resources"]["memory_limit_exceeded"] is False
        assert metrics["resources"] == metadata["execution"]["resources"]
        assert resource_events[0]["event"] == "process_started"
        assert resource_events[-1]["event"] == "process_finished"
        assert any(event["event"] == "resource_sample" for event in resource_events)
        assert metadata["execution"]["secret_names"] == ["TEST_API_KEY"]
        assert (
            metadata["execution"]["environment"]["RXNHAYSTACK_SWISSAI_HOST_REQUESTS_PER_MINUTE_CAP"]
            == "6"
        )
        assert "never-record-this" not in json.dumps(metadata)
        assert "secret-present=True" in stdout

    resumed = run_selected(
        manifest,
        preflight=preflight,
        selected=list(manifest.runs),
        secrets={},
        max_parallel=2,
        retry_failed=False,
        recover_running=False,
    )
    assert [result.status for result in resumed] == ["skipped", "skipped"]
    ledger = RunLedger(manifest.campaign.artifact_dir / "ledger.sqlite3")
    assert len(ledger.list_attempts()) == 2


def test_docker_cleanup_removes_only_exact_attempt_label(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[:3] == ["docker", "ps", "-aq"]:
            return subprocess.CompletedProcess(command, 0, "abc\ndef\n", "")
        return subprocess.CompletedProcess(command, 0, "abc\ndef\n", "")

    monkeypatch.setattr("rxnhaystack.launcher.subprocess.run", fake_run)

    assert cleanup_labeled_docker_containers("attempt-token") is None
    assert commands == [
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=rxnhaystack.run_token=attempt-token",
        ],
        ["docker", "container", "rm", "--force", "abc", "def"],
    ]


def test_launcher_fails_successful_process_that_omits_required_metrics(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    initialize_git_repository(project_root)
    worker = project_root / "worker.py"
    write_worker(worker, write_metrics=False)
    manifest = load_manifest(write_campaign(tmp_path, project_root, worker, repetitions=1))

    [result] = run_selected(
        manifest,
        preflight=perform_preflight(manifest),
        selected=list(manifest.runs),
        secrets={},
        max_parallel=1,
        retry_failed=False,
        recover_running=False,
    )

    assert result.status == "failed"
    assert "required metrics" in (result.error or "")
    assert (
        RunLedger(manifest.campaign.artifact_dir / "ledger.sqlite3").get(result.run_id).status
        == "failed"
    )


def test_launcher_records_wall_time_failure_and_continues(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    initialize_git_repository(project_root)
    worker = project_root / "worker.py"
    write_worker(worker, hold_seconds=60)
    manifest = load_manifest(write_campaign(tmp_path, project_root, worker, repetitions=1))

    [result] = run_selected(
        manifest,
        preflight=perform_preflight(manifest),
        selected=list(manifest.runs),
        secrets={},
        max_parallel=1,
        retry_failed=False,
        recover_running=False,
        max_run_seconds=0.1,
    )

    assert result.status == "failed"
    assert "wall time exceeded" in (result.error or "")
    assert result.artifact_dir is not None
    metadata = json.loads((result.artifact_dir / "metadata.json").read_text())
    resources = metadata["execution"]["resources"]
    assert resources["wall_time_limit_seconds"] == 0.1
    assert resources["wall_time_limit_exceeded"] is True
    record = RunLedger(manifest.campaign.artifact_dir / "ledger.sqlite3").get(result.run_id)
    assert record.status == "failed"


def test_budget_includes_failed_attempt_before_retry(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    initialize_git_repository(project_root)
    worker = project_root / "worker.py"
    write_worker(worker, exit_code=1)
    manifest = load_manifest(
        write_campaign(
            tmp_path,
            project_root,
            worker,
            repetitions=1,
            budget_chf=1.05,
            estimated_cost_chf=1,
        )
    )
    preflight = perform_preflight(manifest)
    [result] = run_selected(
        manifest,
        preflight=preflight,
        selected=list(manifest.runs),
        secrets={},
        max_parallel=1,
        retry_failed=False,
        recover_running=False,
    )
    assert result.status == "failed"

    ledger = RunLedger(manifest.campaign.artifact_dir / "ledger.sqlite3")
    with pytest.raises(ManifestError, match="exceeds campaign budget"):
        enforce_remaining_budget(manifest, ledger)
    assert enforce_remaining_budget(manifest, ledger, retry_failed=False) == 0.08


def test_selected_budget_does_not_reserve_unselected_pending_runs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    initialize_git_repository(project_root)
    worker = project_root / "worker.py"
    write_worker(worker, cost_chf=1.5)
    manifest = load_manifest(
        write_campaign(
            tmp_path,
            project_root,
            worker,
            repetitions=3,
            budget_chf=3.0,
            estimated_cost_chf=1.0,
        )
    )
    first, second, _third = manifest.runs
    [result] = run_selected(
        manifest,
        preflight=perform_preflight(manifest),
        selected=[first],
        secrets={},
        max_parallel=1,
        retry_failed=False,
        recover_running=False,
    )
    assert result.status == "succeeded"

    ledger = RunLedger(manifest.campaign.artifact_dir / "ledger.sqlite3")
    with pytest.raises(ManifestError, match="exceeds campaign budget"):
        enforce_remaining_budget(manifest, ledger)
    assert enforce_remaining_budget(manifest, ledger, selected=[second]) == 2.5


def test_launcher_stops_queued_jobs_when_actual_cost_exhausts_budget(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    initialize_git_repository(project_root)
    worker = project_root / "worker.py"
    write_worker(worker, cost_chf=0.20)
    manifest = load_manifest(
        write_campaign(
            tmp_path,
            project_root,
            worker,
            repetitions=3,
            budget_chf=0.35,
            estimated_cost_chf=0.10,
        )
    )

    results = run_selected(
        manifest,
        preflight=perform_preflight(manifest),
        selected=list(manifest.runs),
        secrets={},
        max_parallel=1,
        retry_failed=False,
        recover_running=False,
    )

    assert [result.status for result in results] == [
        "succeeded",
        "budget-stopped",
        "budget-stopped",
    ]
    assert results[1].attempt is None
    assert "exceeds campaign budget" in (results[1].error or "")
    ledger = RunLedger(manifest.campaign.artifact_dir / "ledger.sqlite3")
    assert [record.status for record in ledger.list_runs()] == [
        "succeeded",
        "pending",
        "pending",
    ]


def test_launcher_failure_after_claim_never_leaves_running_record(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    initialize_git_repository(project_root)
    worker = project_root / "worker.py"
    write_worker(worker)
    manifest = load_manifest(write_campaign(tmp_path, project_root, worker, repetitions=1))
    conflicting_attempt_dir = (
        manifest.campaign.artifact_dir / "runs" / "integration-cell" / "attempt-001"
    )
    conflicting_attempt_dir.mkdir(parents=True)

    [result] = run_selected(
        manifest,
        preflight=perform_preflight(manifest),
        selected=list(manifest.runs),
        secrets={},
        max_parallel=1,
        retry_failed=False,
        recover_running=False,
    )

    assert result.status == "failed"
    assert result.return_code == 125
    assert "FileExistsError" in (result.error or "")
    ledger = RunLedger(manifest.campaign.artifact_dir / "ledger.sqlite3")
    assert ledger.get(result.run_id).status == "failed"
    assert (conflicting_attempt_dir / "launcher-error.json").is_file()


def test_launcher_enforces_process_tree_memory_limit(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    initialize_git_repository(project_root)
    worker = project_root / "memory_worker.py"
    worker.write_text(
        "import time\n"
        "allocation = bytearray(96 * 1024 * 1024)\n"
        "print(len(allocation), flush=True)\n"
        "time.sleep(30)\n"
    )
    manifest_path = tmp_path / "memory.toml"
    manifest_path.write_text(
        f"""
schema_version = 1

[campaign]
name = "memory-integration"
project_root = {json.dumps(str(project_root))}
artifact_dir = {json.dumps(str(tmp_path / "memory-artifacts"))}
budget_chf = 0
usd_to_chf = 0.8
require_dataset = false
require_clean_git = true
require_metrics = false
max_parallel_memory_mib = 128

[[runs]]
id = "memory-cell"
task = "test/memory"
condition = "watchdog"
method = "deterministic"
model = "none"
corpus_size = 1
seed = 42
repetitions = 1
estimated_cost_chf = 0
memory_reservation_mib = 64
memory_limit_mib = 64
command = [{json.dumps(sys.executable)}, {json.dumps(str(worker))}]
""".strip()
        + "\n"
    )
    manifest = load_manifest(manifest_path)

    [result] = run_selected(
        manifest,
        preflight=perform_preflight(manifest),
        selected=list(manifest.runs),
        secrets={},
        max_parallel=1,
        retry_failed=False,
        recover_running=False,
    )

    assert result.status == "failed"
    assert "memory limit" in (result.error or "")
    assert result.artifact_dir is not None
    metadata = json.loads((result.artifact_dir / "metadata.json").read_text())
    resources = metadata["execution"]["resources"]
    assert resources["memory_limit_exceeded"] is True
    assert resources["peak_process_tree_rss_mib"] > 64
    record = RunLedger(manifest.campaign.artifact_dir / "ledger.sqlite3").get(result.run_id)
    assert record.status == "failed"
    assert "memory limit" in (record.error or "")


def test_keyboard_interrupt_terminates_worker_and_records_failure(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    initialize_git_repository(project_root)
    worker = project_root / "sleep_worker.py"
    pid_path = project_root / "worker.pid"
    worker.write_text(
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        f"Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    manifest = load_manifest(write_campaign(tmp_path, project_root, worker, repetitions=1))

    def interrupt_after_worker_starts(_futures):
        deadline = time.monotonic() + 5
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_path.exists()
        raise KeyboardInterrupt

    monkeypatch.setattr("rxnhaystack.launcher.as_completed", interrupt_after_worker_starts)

    with pytest.raises(KeyboardInterrupt):
        run_selected(
            manifest,
            preflight=perform_preflight(manifest),
            selected=list(manifest.runs),
            secrets={},
            max_parallel=1,
            retry_failed=False,
            recover_running=False,
        )

    worker_pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)
    ledger = RunLedger(manifest.campaign.artifact_dir / "ledger.sqlite3")
    record = ledger.get("integration-cell")
    assert record.status == "failed"
    assert record.error == "Run interrupted by launcher shutdown"
    metadata = json.loads(
        (
            manifest.campaign.artifact_dir / "runs/integration-cell/attempt-001/metadata.json"
        ).read_text()
    )
    assert metadata["result"]["error"] == "Run interrupted by launcher shutdown"
