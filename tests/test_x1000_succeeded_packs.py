from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from experiments.iclr2027.publish_x1000_succeeded_packs import (
    DEEPSEEK_MODEL,
    GEMINI_MODEL,
    QWEN_MODEL,
    build_deepseek_repair_snapshot,
    build_pack_snapshot,
)
from rxnhaystack.control_room import snapshot_digest, validate_snapshot


def write_pack(path: Path, *, model: str, method: str, docker_only: bool = False) -> None:
    tasks = (
        [("4", task) for task in ("16", "17", "17b")]
        if docker_only
        else [("1", str(index)) for index in range(1, 31)]
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
    gemini_codeact = tmp_path / "gemini-codeact.tgz"
    gemini_rlm = tmp_path / "gemini-rlm.tgz"
    deepseek_rlm = tmp_path / "deepseek-rlm.tgz"
    write_pack(qwen, model=QWEN_MODEL, method="codeact")
    write_pack(gemini_codeact, model=GEMINI_MODEL, method="codeact")
    write_pack(gemini_rlm, model=GEMINI_MODEL, method="rlm", docker_only=True)
    write_pack(deepseek_rlm, model=DEEPSEEK_MODEL, method="rlm", docker_only=True)

    snapshot = build_pack_snapshot(qwen, gemini_codeact, gemini_rlm)

    validate_snapshot(snapshot)
    assert len(snapshot["experiment"]["expected_runs"]) == 315
    assert len(snapshot["observations"]) == 315
    assert {item["status"] for item in snapshot["observations"]} == {"succeeded"}


def test_build_pack_snapshot_rejects_non_success(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen.tgz"
    gemini_codeact = tmp_path / "gemini-codeact.tgz"
    gemini_rlm = tmp_path / "gemini-rlm.tgz"
    deepseek_rlm = tmp_path / "deepseek-rlm.tgz"
    write_pack(qwen, model=QWEN_MODEL, method="codeact")
    write_pack(gemini_codeact, model=GEMINI_MODEL, method="codeact")
    write_pack(gemini_rlm, model=GEMINI_MODEL, method="rlm", docker_only=True)
    write_pack(deepseek_rlm, model=DEEPSEEK_MODEL, method="rlm", docker_only=True)
    with tarfile.open(gemini_rlm, "r:gz") as archive:
        manifest = json.load(archive.extractfile("pack/manifest.json"))
        rows = json.load(archive.extractfile("pack/runs.json"))
    rows[0]["status"] = "failed"
    with tarfile.open(gemini_rlm, "w:gz") as archive:
        for name, payload in (("manifest.json", manifest), ("runs.json", rows)):
            data = json.dumps(payload).encode()
            info = tarfile.TarInfo(f"pack/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    with pytest.raises(ValueError, match="Only successful x1000"):
        build_pack_snapshot(qwen, gemini_codeact, gemini_rlm)


def test_build_deepseek_repair_uses_reference_campaign_identities(tmp_path: Path) -> None:
    pack = tmp_path / "deepseek-rlm.tgz"
    reference_path = tmp_path / "reference.json"
    write_pack(pack, model=DEEPSEEK_MODEL, method="rlm", docker_only=True)
    expected = []
    for task in ("16", "17", "17b"):
        for repetition in range(1, 6):
            run_id = (
                "x1000-openrouter-full-deepseek-v4-flash-"
                f"tier4-task{task}-rlm-x1000-r{repetition:02d}"
            )
            expected.append({"run_id": run_id, "spec_hash": f"hash-{task}-{repetition}"})
    reference = {
        "schema_version": 1,
        "generated_at": "2026-09-23T08:00:00+00:00",
        "source": {"id": "reference", "machine": "test", "owner": "test"},
        "experiment": {
            "name": "iclr2027-deepseek-rlm-x1000-v1",
            "definition_sha256": "definition",
            "expected_cost_chf": 0.0,
            "expected_runs": expected,
        },
        "observations": [],
    }
    reference["snapshot_id"] = snapshot_digest(reference)
    reference_path.write_text(json.dumps(reference))

    snapshot = build_deepseek_repair_snapshot(pack, reference_path)

    validate_snapshot(snapshot)
    assert snapshot["experiment"] == reference["experiment"]
    assert len(snapshot["observations"]) == 15
    assert {row["run_id"] for row in snapshot["observations"]} == {
        row["run_id"] for row in expected
    }
