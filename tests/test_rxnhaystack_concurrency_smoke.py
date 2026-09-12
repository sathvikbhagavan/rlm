from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rxnhaystack.concurrency_smoke import main


def test_concurrency_smoke_observes_bound_and_writes_ordered_results(
    tmp_path: Path, monkeypatch
) -> None:
    metrics = tmp_path / "metrics.json"
    values = {
        "RXNHAYSTACK_RUN_ID": "smoke",
        "RXNHAYSTACK_RUN_DIR": str(tmp_path),
        "RXNHAYSTACK_METRICS_PATH": str(metrics),
        "RXNHAYSTACK_TASK": "infrastructure/concurrency",
        "RXNHAYSTACK_CONDITION": "test",
        "RXNHAYSTACK_METHOD": "deterministic",
        "RXNHAYSTACK_MODEL": "none",
        "RXNHAYSTACK_CORPUS_SIZE": "1",
        "RXNHAYSTACK_SEED": "0",
        "RXNHAYSTACK_REPETITION": "1",
        "RXNHAYSTACK_USD_TO_CHF": "0.8",
        "RXNHAYSTACK_QUESTION_PARALLELISM": "3",
        "RXNHAYSTACK_RESOURCE_TRACE_PATH": str(tmp_path / "resource-trace.jsonl"),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    asyncio.run(main())

    payload = json.loads(metrics.read_text())
    assert payload["results"]["question_count"] == 6
    assert payload["results"]["maximum_active_questions"] == 3
    assert payload["results"]["result_order"] == list(range(6))
