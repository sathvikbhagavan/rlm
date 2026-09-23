from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from experiments.iclr2027.publish_x1000_succeeded_packs import (
    GEMINI_MODEL,
    QWEN_MODEL,
    build_pack_snapshot,
)
from rxnhaystack.control_room import validate_snapshot


def write_pack(path: Path, *, model: str) -> None:
    method = "codeact" if model == QWEN_MODEL else "rlm"
    tasks = (
        [("1", str(index)) for index in range(1, 31)]
        if model == QWEN_MODEL
        else [("4", task) for task in ("16", "17", "17b")]
    )
    rows = []
    for tier, task in tasks:
        for repetition in range(1, 6):
            rows.append(
                {
                    "run_id": (
                        f"full-{model}-tier{tier}-task{task}-{method}-x1000-r{repetition:02d}"
                    ),
                    "model": model,
                    "tier": tier,
                    "task": task,
                    "method": method,
                    "context": "1000",
                    "repetition": f"{repetition:02d}",
                    "status": "succeeded",
                    "attempt": 1,
                    "metrics.cost_chf": 0.1,
                    "metrics.total_tokens": 10,
                    "metrics.resources.process_wall_time_seconds": 2.0,
                    "metrics.results.macro_f1": 0.5,
                }
            )
    manifest = {
        "packed_at": "2026-09-23T08:00:00+00:00",
        "models": [model],
        "n_runs": len(rows),
        "rule": "test",
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in (("manifest.json", manifest), ("runs.json", rows)):
            data = json.dumps(payload).encode()
            info = tarfile.TarInfo(f"pack/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_build_pack_snapshot_validates_and_combines_both_packs(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen.tgz"
    gemini = tmp_path / "gemini.tgz"
    write_pack(qwen, model=QWEN_MODEL)
    write_pack(gemini, model=GEMINI_MODEL)

    snapshot = build_pack_snapshot(qwen, gemini)

    validate_snapshot(snapshot)
    assert len(snapshot["experiment"]["expected_runs"]) == 165
    assert len(snapshot["observations"]) == 165
    assert {item["status"] for item in snapshot["observations"]} == {"succeeded"}


def test_build_pack_snapshot_rejects_non_success(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen.tgz"
    gemini = tmp_path / "gemini.tgz"
    write_pack(qwen, model=QWEN_MODEL)
    write_pack(gemini, model=GEMINI_MODEL)
    with tarfile.open(gemini, "r:gz") as archive:
        manifest = json.load(archive.extractfile("pack/manifest.json"))
        rows = json.load(archive.extractfile("pack/runs.json"))
    rows[0]["status"] = "failed"
    with tarfile.open(gemini, "w:gz") as archive:
        for name, payload in (("manifest.json", manifest), ("runs.json", rows)):
            data = json.dumps(payload).encode()
            info = tarfile.TarInfo(f"pack/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    with pytest.raises(ValueError, match="Only successful x1000"):
        build_pack_snapshot(qwen, gemini)
