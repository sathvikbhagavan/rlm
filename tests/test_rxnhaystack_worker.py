from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rxnhaystack.manifest import ManifestError
from rxnhaystack.worker import (
    RLM_LOCAL_MEMORY_LIMIT_ENV,
    BenchmarkRuntime,
    instrument_rlm_from_environment,
)


def test_benchmark_runtime_preserves_standalone_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RXNHAYSTACK_RUN_ID", raising=False)

    runtime = BenchmarkRuntime.from_defaults(
        model="default/model",
        dataset_path=tmp_path / "dataset.txt",
        seed=42,
    )

    assert runtime.model == "default/model"
    assert runtime.seed == 42
    assert runtime.question_parallelism == 1
    assert not runtime.launched


def test_benchmark_runtime_uses_typed_campaign_values(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw.txt"
    cleaned = tmp_path / "cleaned.txt"
    raw.write_text("C>>CO\n")
    cleaned.write_text("C>>CO\n")
    values = {
        "RXNHAYSTACK_RUN_ID": "run",
        "RXNHAYSTACK_RUN_DIR": str(tmp_path / "run"),
        "RXNHAYSTACK_METRICS_PATH": str(tmp_path / "run" / "metrics.json"),
        "RXNHAYSTACK_TASK": "tier1/task1",
        "RXNHAYSTACK_CONDITION": "baseline",
        "RXNHAYSTACK_METHOD": "rlm",
        "RXNHAYSTACK_MODEL": "campaign/model",
        "RXNHAYSTACK_CORPUS_SIZE": "full",
        "RXNHAYSTACK_SEED": "7",
        "RXNHAYSTACK_REPETITION": "1",
        "RXNHAYSTACK_USD_TO_CHF": "0.8",
        "RXNHAYSTACK_RAW_DATASET": str(raw),
        "RXNHAYSTACK_CLEANED_DATASET": str(cleaned),
        "RXNHAYSTACK_QUESTION_PARALLELISM": "4",
        "RXNHAYSTACK_RESOURCE_TRACE_PATH": str(tmp_path / "run" / "resource-trace.jsonl"),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    runtime = BenchmarkRuntime.from_defaults(
        model="default/model",
        dataset_path="ignored.txt",
        seed=42,
    )
    kwargs = runtime.instrument_rlm_kwargs(
        {"backend": "openrouter", "backend_kwargs": {"model_name": "old"}},
        sample_id=2,
    )

    assert runtime.model == "campaign/model"
    assert runtime.dataset_path == cleaned.resolve()
    assert runtime.seed == 7
    assert runtime.question_parallelism == 4
    assert runtime.launched
    assert kwargs["backend_kwargs"]["model_name"] == "campaign/model"
    assert set(kwargs) >= {
        "on_subcall_start",
        "on_subcall_complete",
        "on_iteration_start",
        "on_iteration_complete",
    }


def test_local_rlm_sets_address_space_limit_and_records_it(
    tmp_path: Path, monkeypatch
) -> None:
    trace_path = tmp_path / "resource-trace.jsonl"
    monkeypatch.setenv("RXNHAYSTACK_RESOURCE_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("RXNHAYSTACK_PROVIDER", "openrouter")
    monkeypatch.setenv(RLM_LOCAL_MEMORY_LIMIT_ENV, "4096")

    with (
        patch("rxnhaystack.worker.resource.getrlimit", return_value=(-1, -1)),
        patch("rxnhaystack.worker.resource.setrlimit") as setrlimit,
    ):
        configured = instrument_rlm_from_environment(
            {"backend": "openrouter", "environment": "local"}
        )

    setrlimit.assert_called_once()
    assert setrlimit.call_args.args[1][0] == 4096 * 1024 * 1024
    assert configured["environment"] == "local"
    assert '"event":"rlm_local_memory_limit_set"' in trace_path.read_text()


def test_local_rlm_rejects_invalid_memory_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(
        "RXNHAYSTACK_RESOURCE_TRACE_PATH", str(tmp_path / "resource-trace.jsonl")
    )
    monkeypatch.setenv("RXNHAYSTACK_PROVIDER", "openrouter")
    monkeypatch.setenv(RLM_LOCAL_MEMORY_LIMIT_ENV, "0")

    with pytest.raises(ManifestError, match=RLM_LOCAL_MEMORY_LIMIT_ENV):
        instrument_rlm_from_environment(
            {"backend": "openrouter", "environment": "local"}
        )
