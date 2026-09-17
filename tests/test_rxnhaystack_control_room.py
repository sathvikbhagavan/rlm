from __future__ import annotations

import json
import shutil
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rxnhaystack import cli, control_room
from rxnhaystack.control_room import (
    ControlRoomError,
    build_snapshot,
    classify_failure,
    load_snapshot,
    merge_snapshots,
    publish_snapshot,
    snapshot_digest,
    sync_snapshots,
    write_dashboard,
    write_markdown,
    write_snapshot,
)
from rxnhaystack.ledger import RunLedger
from rxnhaystack.manifest import load_manifest
from rxnhaystack.runtime import GitProvenance


def write_experiment(tmp_path: Path) -> Path:
    path = tmp_path / "experiment.toml"
    path.write_text(
        """
schema_version = 1

[campaign]
name = "control-room-test"
project_root = "."
artifact_dir = "artifacts/control-room-test"
budget_chf = 10.0
usd_to_chf = 0.8
require_dataset = false
require_clean_git = false

[[runs]]
id = "control-room-test-llm"
task = "tier1/task1"
condition = "full-x100"
method = "llm"
model = "openai/gpt-5-mini"
corpus_size = 100
seed = 42
repetitions = 2
estimated_cost_chf = 1.0
command = ["python", "task.py"]
""".lstrip(),
        encoding="utf-8",
    )
    return path


def populated_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    manifest = load_manifest(write_experiment(tmp_path))
    ledger = RunLedger(manifest.campaign.artifact_dir / "ledger.sqlite3")
    ledger.sync_runs(manifest.runs, manifest_sha256=manifest.sha256)
    first, second = manifest.runs
    assert ledger.claim(first.run_id) == 1
    ledger.finish(
        first.run_id,
        return_code=0,
        artifact_dir=tmp_path / "private" / "first",
        metrics={
            "calls": 2,
            "total_tokens": 120,
            "cost_chf": 0.25,
            "resources": {
                "peak_process_tree_rss_mib": 512.0,
                "trace_path": "/private/resource-trace.jsonl",
            },
            "results": {
                "macro_f1": 0.75,
                "question_latencies_seconds": [1.0, 2.0],
            },
            "wandb_url": "https://wandb.ai/liac/task/runs/abc",
        },
    )
    assert ledger.claim(second.run_id) == 1
    ledger.finish(
        second.run_id,
        return_code=1,
        artifact_dir=tmp_path / "private" / "second",
        error="HTTP 403 Policy Violation using sk-secret at /private/prompt.txt",
    )
    monkeypatch.setattr(
        control_room,
        "inspect_git",
        lambda _path: GitProvenance(commit="a" * 40, tracked_dirty=False),
    )
    return build_snapshot(
        manifest,
        source_id="liacpc14-test",
        machine="liacpc14",
        owner="Amin",
        session_name="test-session",
        ledger_path=manifest.campaign.artifact_dir / "ledger.sqlite3",
        generated_at="2026-09-17T12:00:00+00:00",
    )


def resign(snapshot: dict[str, Any]) -> dict[str, Any]:
    snapshot["snapshot_id"] = snapshot_digest(snapshot)
    return snapshot


def test_snapshot_is_sanitized_and_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = populated_snapshot(tmp_path, monkeypatch)
    serialized = json.dumps(snapshot)

    assert len(snapshot["experiment"]["expected_runs"]) == 2
    assert [item["status"] for item in snapshot["observations"]] == ["succeeded", "failed"]
    assert snapshot["observations"][1]["attempts"][0]["failure_category"] == "policy_refusal"
    assert snapshot["observations"][0]["attempts"][0]["metrics"]["results"] == {"macro_f1": 0.75}
    assert "sk-secret" not in serialized
    assert "/private" not in serialized
    assert "question_latencies_seconds" not in serialized
    assert "trace_path" not in serialized


def test_snapshot_hash_and_private_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = populated_snapshot(tmp_path, monkeypatch)
    path = tmp_path / "snapshot.json"
    write_snapshot(path, snapshot)

    assert path.stat().st_mode & 0o777 == 0o600
    assert load_snapshot(path) == snapshot
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["source"]["owner"] = "someone else"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ControlRoomError, match="hash"):
        load_snapshot(path)


def test_merge_deduplicates_copied_attempts_and_flags_independent_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = populated_snapshot(tmp_path, monkeypatch)
    copied = deepcopy(first)
    copied["source"].update({"id": "jed-copy", "machine": "jed"})
    copied["generated_at"] = "2026-09-17T12:05:00+00:00"
    resign(copied)

    merged_copy = merge_snapshots(
        [first, copied],
        now=datetime(2026, 9, 17, 12, 10, tzinfo=UTC),
    )
    campaign = merged_copy["campaigns"][0]
    assert campaign["metrics"]["attempts"] == 2
    assert campaign["metrics"]["cost_chf"] == 0.25
    assert campaign["duplicate_runs"] == 0

    independent = deepcopy(copied)
    failed = independent["observations"][1]
    failed["status"] = "succeeded"
    attempt = failed["attempts"][0]
    attempt.update(
        {
            "attempt_key": "b" * 64,
            "status": "succeeded",
            "finished_at": "2026-09-17T12:06:00+00:00",
            "return_code": 0,
            "failure_category": None,
            "metrics": {"calls": 1, "cost_chf": 0.1},
        }
    )
    resign(independent)
    merged_independent = merge_snapshots(
        [first, independent],
        now=datetime(2026, 9, 17, 12, 10, tzinfo=UTC),
    )["campaigns"][0]

    assert merged_independent["counts"]["succeeded"] == 2
    assert merged_independent["duplicate_runs"] == 1
    assert merged_independent["metrics"]["attempts"] == 3


def test_stale_is_based_on_machine_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = populated_snapshot(tmp_path, monkeypatch)
    observation = snapshot["observations"][1]
    observation["status"] = "running"
    observation["attempts"][0]["status"] = "running"
    observation["attempts"][0]["finished_at"] = None
    observation["attempts"][0]["failure_category"] = None
    resign(snapshot)

    merged = merge_snapshots(
        [snapshot],
        stale_after_seconds=3600,
        now=datetime(2026, 9, 17, 14, 0, tzinfo=UTC),
    )["campaigns"][0]
    assert merged["counts"]["stale"] == 1
    assert merged["sources"][0]["stale"] is True


@pytest.mark.parametrize(
    ("message", "category"),
    [
        ("HTTP 429 rate limit", "rate_limit"),
        ("maximum context window exceeded", "context_overflow"),
        ("cannot allocate memory", "memory_limit"),
        ("API request timed out", "api_timeout"),
        ("HTTP 503", "http_5xx"),
        ("return code 143", "interrupted"),
        ("empty choices", "malformed_response"),
    ],
)
def test_failure_classification_does_not_publish_message(message: str, category: str) -> None:
    assert classify_failure(message) == category


class FakeLoggedArtifact:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class FakeArtifact:
    def __init__(self, name: str, type: str, metadata: dict[str, Any]) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((path, name))


class FakeRun:
    def __init__(self) -> None:
        self.url = "https://wandb.ai/liac/rxnhaystack-control-room/runs/test"
        self.summary: dict[str, Any] = {}
        self.logged: list[tuple[FakeArtifact, list[str], FakeLoggedArtifact]] = []
        self.finished = False

    def log_artifact(self, artifact: FakeArtifact, *, aliases: list[str]) -> FakeLoggedArtifact:
        logged = FakeLoggedArtifact()
        self.logged.append((artifact, aliases, logged))
        return logged

    def finish(self) -> None:
        self.finished = True


class FakeWandbPublisher:
    Artifact = FakeArtifact

    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_arguments: dict[str, Any] = {}

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_arguments = kwargs
        return self.run


def test_publish_uses_latest_alias_and_restores_secret_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = populated_snapshot(tmp_path, monkeypatch)
    path = tmp_path / "snapshot.json"
    write_snapshot(path, snapshot)
    fake = FakeWandbPublisher()
    monkeypatch.setenv("WANDB_API_KEY", "original")

    url = publish_snapshot(
        path,
        api_key="temporary",
        entity="liac",
        project="rxnhaystack-control-room",
        wandb_module=fake,
    )

    artifact, aliases, logged = fake.run.logged[0]
    assert url == fake.run.url
    assert aliases == ["latest"]
    assert artifact.files[0][1] == "snapshot.json.gz"
    assert logged.waited is True
    assert fake.run.finished is True
    assert fake.init_arguments["save_code"] is False
    assert control_room.os.environ["WANDB_API_KEY"] == "original"


class FakeDownloadArtifact:
    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path

    def download(self, *, root: str) -> str:
        destination = Path(root)
        shutil.copy2(self.snapshot_path, destination / "snapshot.json")
        return str(destination)


class FakePublicApi:
    def __init__(self, snapshot_path: Path, snapshot: dict[str, Any]) -> None:
        self.snapshot_path = snapshot_path
        self.snapshot = snapshot

    def runs(self, _path: str, **_kwargs: Any) -> list[SimpleNamespace]:
        config = {
            "source_id": self.snapshot["source"]["id"],
            "artifact_name": control_room.artifact_name(self.snapshot),
            "snapshot_id": self.snapshot["snapshot_id"],
        }
        return [SimpleNamespace(config=config)]

    def artifact(self, _name: str) -> FakeDownloadArtifact:
        return FakeDownloadArtifact(self.snapshot_path)


class FakeWandbDownloader:
    def __init__(self, api: FakePublicApi) -> None:
        self.public_api = api

    def Api(self, *, api_key: str) -> FakePublicApi:
        assert api_key == "secret"
        return self.public_api


def test_sync_downloads_and_validates_latest_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = populated_snapshot(tmp_path, monkeypatch)
    source = tmp_path / "published.json"
    write_snapshot(source, snapshot)
    fake_api = FakePublicApi(source, snapshot)
    monkeypatch.setenv("WANDB_API_KEY", "original")

    paths = sync_snapshots(
        api_key="secret",
        entity="liac",
        project="rxnhaystack-control-room",
        output_dir=tmp_path / "shared",
        wandb_module=FakeWandbDownloader(fake_api),
    )

    assert paths == [tmp_path / "shared" / "liacpc14-test.json"]
    assert load_snapshot(paths[0]) == snapshot
    assert control_room.os.environ["WANDB_API_KEY"] == "original"


def test_dashboard_and_markdown_are_generated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = populated_snapshot(tmp_path, monkeypatch)
    merged = merge_snapshots([snapshot], now=datetime(2026, 9, 17, 12, 10, tzinfo=UTC))
    html_path = tmp_path / "index.html"
    markdown_path = tmp_path / "status.md"

    write_dashboard(html_path, merged)
    write_markdown(markdown_path, merged)

    dashboard = html_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "RxnHaystack Dashboard" in dashboard
    assert "Keep reporting sources up to date" in dashboard
    assert "refreshSafely" in dashboard
    assert 'http-equiv="refresh"' not in dashboard
    assert "Run explorer" in dashboard
    assert '<details class="panel run-explorer">' in dashboard
    assert '<details class="panel run-explorer" open>' not in dashboard
    assert "GPT-5 mini" in dashboard
    assert "Model and method matrix" in dashboard
    assert "| GPT-5 mini | llm | 2 | 1 |" in markdown


def test_dashboard_command_and_legacy_alias_are_both_available() -> None:
    parser = cli.build_parser()
    current = parser.parse_args(["dashboard", "view", "--no-sync", "--no-serve"])
    legacy = parser.parse_args(["control-room", "view", "--no-sync", "--no-serve"])

    assert current.command == "dashboard"
    assert legacy.command == "control-room"


def test_fresh_running_source_remains_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = populated_snapshot(tmp_path, monkeypatch)
    observation = snapshot["observations"][1]
    observation["status"] = "running"
    observation["attempts"][0]["status"] = "running"
    observation["attempts"][0]["finished_at"] = None
    observation["attempts"][0]["failure_category"] = None
    resign(snapshot)

    merged = merge_snapshots(
        [snapshot],
        stale_after_seconds=3600,
        now=datetime(2026, 9, 17, 12, 30, tzinfo=UTC),
    )["campaigns"][0]
    assert merged["counts"]["running"] == 1
    assert merged["counts"]["stale"] == 0


def test_heartbeat_boundary_is_timezone_aware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = populated_snapshot(tmp_path, monkeypatch)
    snapshot["generated_at"] = (
        datetime(2026, 9, 17, 12, tzinfo=UTC) - timedelta(minutes=5)
    ).isoformat()
    resign(snapshot)
    merged = merge_snapshots([snapshot], now=datetime(2026, 9, 17, 12, tzinfo=UTC))["campaigns"][0]
    assert merged["sources"][0]["age_seconds"] == 300
