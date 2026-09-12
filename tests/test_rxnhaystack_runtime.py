from __future__ import annotations

import json
from pathlib import Path

import pytest

from rxnhaystack.manifest import ManifestError
from rxnhaystack.metrics import RunMetrics, cost_chf_from_usd, write_run_metrics
from rxnhaystack.runtime import (
    WorkerConfig,
    atomic_write_json,
    load_secret_specs,
    resolve_required_secrets,
)


def test_load_secret_specs_reads_private_files_without_returning_paths(tmp_path: Path) -> None:
    secret_path = tmp_path / "api-key"
    secret_path.write_text("super-secret\n")
    secret_path.chmod(0o600)

    secrets = load_secret_specs([f"OPENROUTER_API_KEY={secret_path}"])

    assert secrets == {"OPENROUTER_API_KEY": "super-secret"}


def test_load_secret_specs_rejects_insecure_permissions(tmp_path: Path) -> None:
    secret_path = tmp_path / "api-key"
    secret_path.write_text("super-secret\n")
    secret_path.chmod(0o644)

    with pytest.raises(ManifestError, match="require mode 0600"):
        load_secret_specs([f"OPENROUTER_API_KEY={secret_path}"])


@pytest.mark.parametrize("specification", ["missing-equals", "lowercase=/tmp/key", "NAME="])
def test_load_secret_specs_rejects_invalid_assignments(specification: str) -> None:
    with pytest.raises(ManifestError, match="NAME=/path"):
        load_secret_specs([specification])


def test_atomic_write_json_replaces_destination_and_leaves_no_temporary_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "metadata.json"
    destination.write_text("old")

    atomic_write_json(destination, {"status": "complete", "count": 2})

    assert json.loads(destination.read_text()) == {"status": "complete", "count": 2}
    assert list(tmp_path.glob("*.tmp")) == []


def test_run_metrics_checks_token_accounting() -> None:
    with pytest.raises(ManifestError, match="total_tokens"):
        RunMetrics(
            calls=1,
            input_tokens=10,
            output_tokens=5,
            total_tokens=14,
            latency_seconds=1,
            tool_time_seconds=0,
            cost_chf=0.1,
        )


def test_write_run_metrics_uses_launcher_path(tmp_path: Path) -> None:
    destination = tmp_path / "metrics.json"
    metrics = RunMetrics(
        calls=1,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        latency_seconds=1.5,
        tool_time_seconds=0.25,
        cost_usd=0.1,
        cost_chf=0.08,
        results={"macro_f1": 0.75},
    )

    result = write_run_metrics(metrics, environ={"RXNHAYSTACK_METRICS_PATH": str(destination)})

    assert result == destination.resolve()
    assert json.loads(destination.read_text())["results"] == {"macro_f1": 0.75}


def test_cost_conversion_uses_frozen_campaign_rate() -> None:
    assert cost_chf_from_usd(10, environ={"RXNHAYSTACK_USD_TO_CHF": "0.8"}) == 8


def complete_worker_environment(tmp_path: Path) -> dict[str, str]:
    raw = tmp_path / "raw.txt"
    cleaned = tmp_path / "cleaned.txt"
    raw.write_text("C>>CO\n")
    cleaned.write_text("C>>CO\n")
    return {
        "RXNHAYSTACK_RUN_ID": "test-run",
        "RXNHAYSTACK_RUN_DIR": str(tmp_path / "run"),
        "RXNHAYSTACK_METRICS_PATH": str(tmp_path / "run" / "metrics.json"),
        "RXNHAYSTACK_TASK": "tier1/task1",
        "RXNHAYSTACK_CONDITION": "baseline",
        "RXNHAYSTACK_METHOD": "rlm",
        "RXNHAYSTACK_MODEL": "provider/model",
        "RXNHAYSTACK_CORPUS_SIZE": "500",
        "RXNHAYSTACK_POSITIVE_CARDINALITY": "3",
        "RXNHAYSTACK_SEED": "42",
        "RXNHAYSTACK_REPETITION": "2",
        "RXNHAYSTACK_USD_TO_CHF": "0.8",
        "RXNHAYSTACK_RAW_DATASET": str(raw),
        "RXNHAYSTACK_CLEANED_DATASET": str(cleaned),
        "RXNHAYSTACK_QUESTION_PARALLELISM": "3",
        "RXNHAYSTACK_RESOURCE_TRACE_PATH": str(tmp_path / "run" / "resource-trace.jsonl"),
    }


def test_worker_config_parses_complete_launcher_contract(tmp_path: Path) -> None:
    config = WorkerConfig.from_environment(complete_worker_environment(tmp_path))

    assert config.run_id == "test-run"
    assert config.model == "provider/model"
    assert config.corpus_size == 500
    assert config.positive_cardinality == 3
    assert config.seed == 42
    assert config.repetition == 2
    assert config.question_parallelism == 3
    assert config.require_dataset().cleaned == (tmp_path / "cleaned.txt").resolve()


def test_worker_config_preserves_named_full_corpus(tmp_path: Path) -> None:
    environ = complete_worker_environment(tmp_path)
    environ["RXNHAYSTACK_CORPUS_SIZE"] = "full"

    assert WorkerConfig.from_environment(environ).corpus_size == "full"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("RXNHAYSTACK_SEED", "not-an-int", "must be an integer"),
        ("RXNHAYSTACK_REPETITION", "0", "must be >= 1"),
        ("RXNHAYSTACK_USD_TO_CHF", "0", "must be > 0"),
        ("RXNHAYSTACK_QUESTION_PARALLELISM", "0", "must be >= 1"),
    ],
)
def test_worker_config_rejects_invalid_values(
    tmp_path: Path, name: str, value: str, message: str
) -> None:
    environ = complete_worker_environment(tmp_path)
    environ[name] = value

    with pytest.raises(ManifestError, match=message):
        WorkerConfig.from_environment(environ)


def test_required_secrets_accepts_supplied_files_and_exported_values(tmp_path: Path) -> None:
    from rxnhaystack.manifest import load_manifest

    manifest_path = tmp_path / "campaign.toml"
    manifest_path.write_text(
        """
schema_version = 1
[campaign]
name = "secrets"
budget_chf = 0
usd_to_chf = 0.8
require_dataset = false
require_clean_git = false
required_secrets = ["OPENROUTER_API_KEY", "WANDB_API_KEY"]
[[runs]]
id = "test"
task = "test"
condition = "test"
method = "test"
model = "none"
corpus_size = 1
seed = 0
estimated_cost_chf = 0
command = ["python"]
""".strip()
        + "\n"
    )
    manifest = load_manifest(manifest_path)

    resolved = resolve_required_secrets(
        manifest,
        {"OPENROUTER_API_KEY": "from-file"},
        environ={"WANDB_API_KEY": "from-environment"},
    )

    assert resolved == {
        "OPENROUTER_API_KEY": "from-file",
        "WANDB_API_KEY": "from-environment",
    }


def test_required_secrets_fail_before_launch_with_actionable_message(tmp_path: Path) -> None:
    from rxnhaystack.manifest import load_manifest

    manifest_path = tmp_path / "campaign.toml"
    manifest_path.write_text(
        """
schema_version = 1
[campaign]
name = "secrets"
budget_chf = 0
usd_to_chf = 0.8
require_dataset = false
require_clean_git = false
required_secrets = ["OPENROUTER_API_KEY"]
[[runs]]
id = "test"
task = "test"
condition = "test"
method = "test"
model = "none"
corpus_size = 1
seed = 0
estimated_cost_chf = 0
command = ["python"]
""".strip()
        + "\n"
    )

    with pytest.raises(ManifestError, match="Missing required campaign secret"):
        resolve_required_secrets(load_manifest(manifest_path), {}, environ={})


def test_cli_reports_missing_required_secret_before_creating_artifacts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from rxnhaystack.cli import main

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    manifest_path = tmp_path / "campaign.toml"
    artifact_dir = tmp_path / "artifacts"
    manifest_path.write_text(
        f"""
schema_version = 1
[campaign]
name = "secrets"
artifact_dir = {json.dumps(str(artifact_dir))}
budget_chf = 0
usd_to_chf = 0.8
require_dataset = false
require_clean_git = false
required_secrets = ["OPENROUTER_API_KEY"]
[[runs]]
id = "test"
task = "test"
condition = "test"
method = "test"
model = "none"
corpus_size = 1
seed = 0
estimated_cost_chf = 0
command = ["python"]
""".strip()
        + "\n"
    )

    assert main(["run", str(manifest_path)]) == 2
    assert "Missing required campaign secret" in capsys.readouterr().err
    assert not artifact_dir.exists()
