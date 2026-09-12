from __future__ import annotations

import json
from pathlib import Path

from rxnhaystack.smoke import main


def test_smoke_worker_reads_dataset_and_writes_zero_cost_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "cleaned.txt"
    dataset.write_text("C>>CO\n")
    metrics = tmp_path / "metrics.json"
    monkeypatch.setenv("RXNHAYSTACK_CLEANED_DATASET", str(dataset))
    monkeypatch.setenv("RXNHAYSTACK_RAW_DATASET", str(dataset))
    monkeypatch.setenv("RXNHAYSTACK_METRICS_PATH", str(metrics))
    monkeypatch.setenv("RXNHAYSTACK_RUN_ID", "smoke")
    monkeypatch.setenv("RXNHAYSTACK_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RXNHAYSTACK_TASK", "infrastructure/smoke")
    monkeypatch.setenv("RXNHAYSTACK_CONDITION", "test")
    monkeypatch.setenv("RXNHAYSTACK_METHOD", "deterministic")
    monkeypatch.setenv("RXNHAYSTACK_MODEL", "none")
    monkeypatch.setenv("RXNHAYSTACK_CORPUS_SIZE", "full")
    monkeypatch.setenv("RXNHAYSTACK_POSITIVE_CARDINALITY", "0")
    monkeypatch.setenv("RXNHAYSTACK_SEED", "0")
    monkeypatch.setenv("RXNHAYSTACK_REPETITION", "1")
    monkeypatch.setenv("RXNHAYSTACK_USD_TO_CHF", "0.8")
    monkeypatch.setenv("RXNHAYSTACK_QUESTION_PARALLELISM", "1")
    monkeypatch.setenv("RXNHAYSTACK_RESOURCE_TRACE_PATH", str(tmp_path / "resources.jsonl"))

    main()

    payload = json.loads(metrics.read_text())
    assert payload["calls"] == 0
    assert payload["cost_chf"] == 0
    assert payload["results"]["check"] == "dataset-readable"
