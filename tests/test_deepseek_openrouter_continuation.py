from __future__ import annotations

import sqlite3
from pathlib import Path

from experiments.iclr2027.prepare_deepseek_openrouter_continuation import (
    OPENROUTER_MODEL,
    render,
    unfinished_source_ids,
)
from rxnhaystack.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_continuation_changes_transport_and_preserves_source_identity(tmp_path: Path) -> None:
    source = load_manifest(ROOT / "experiments/iclr2027/full-campaign.toml").runs
    run = next(item for item in source if item.run_id.startswith("full-deepseek") and item.method == "rlm")
    output = tmp_path / "continuation.toml"
    output.write_text(render([run]), encoding="utf-8")
    continuation = load_manifest(output)
    recovered = continuation.runs[0]
    assert recovered.model == OPENROUTER_MODEL
    assert recovered.env["RXNHAYSTACK_PROVIDER"] == "openrouter"
    assert recovered.env["RXNHAYSTACK_SOURCE_RUN_ID"] == run.run_id
    assert not any(key.startswith("RXNHAYSTACK_SWISSAI_") for key in recovered.env)
    assert recovered.command == run.command


def test_only_non_successes_are_selected(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(ledger)
    connection.execute("create table runs (run_id text, status text)")
    connection.executemany(
        "insert into runs values (?, ?)",
        [
            ("full-deepseek-v4-flash-tier1-task1-rlm-x100-r01", "succeeded"),
            ("full-deepseek-v4-flash-tier1-task1-rlm-x100-r02", "failed"),
            ("full-deepseek-v4-flash-tier1-task1-rlm-x100-r03", "pending"),
            ("full-gpt-5-mini-tier1-task1-rlm-x100-r01", "failed"),
        ],
    )
    connection.commit()
    connection.close()
    assert unfinished_source_ids(ledger) == {
        "full-deepseek-v4-flash-tier1-task1-rlm-x100-r02",
        "full-deepseek-v4-flash-tier1-task1-rlm-x100-r03",
    }
