from __future__ import annotations

import json
from pathlib import Path

import pytest

from human_eval.candidate_export import build_candidate_pack, write_candidate_pack
from human_eval.candidates import import_candidate_pack, public_candidates
from human_eval.db import Store


def prediction(path: Path, *, run_id: str = "run-1") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "tier4/task16",
                "run_id": run_id,
                "model": "private-model",
                "prompting_method": "rlm",
                "prompt_condition": "name_only",
                "questions": [
                    {
                        "canonical_question_id": "rxh-t4-task16-lactam-dipeptide",
                        "false_positive_chains": [[1, 2, 3, 4], [5, 6, 7, 8]],
                        "ground_truth_chains": [[10, 11, 12, 13]],
                    }
                ],
            }
        )
    )
    return path


def test_task16_candidate_export_is_deterministic_and_import_blinds_metadata(
    tmp_path: Path,
) -> None:
    source = prediction(tmp_path / "predictions.json")
    first = build_candidate_pack(
        [source],
        pack_id="task16-test",
        version="1.0.0",
        seed=7,
        max_false_positives=1,
    )
    second = build_candidate_pack(
        [source],
        pack_id="task16-test",
        version="1.0.0",
        seed=7,
        max_false_positives=1,
    )
    assert first == second
    assert len(first["candidates"]) == 2
    assert {item["control_type"] for item in first["candidates"]} == {None, "positive"}

    pack_path = write_candidate_pack(first, tmp_path / "candidate-pack.json")
    assert pack_path.stat().st_mode & 0o777 == 0o600
    store = Store(tmp_path / "state.sqlite3")
    result = import_candidate_pack(store, pack_path, tmp_path / "admin")
    assert result["candidate_count"] == 2
    public = public_candidates(store)
    assert all("model_id" not in item for item in public)
    assert all("run_id" not in item for item in public)
    assert all("prompt_condition" not in item for item in public)


def test_task16_candidate_export_rejects_wrong_artifact(tmp_path: Path) -> None:
    source = tmp_path / "wrong.json"
    source.write_text(json.dumps({"schema_version": 1, "task": "tier4/task17", "questions": []}))
    with pytest.raises(ValueError, match="Not a Task-16"):
        build_candidate_pack(
            [source],
            pack_id="task16-test",
            version="1.0.0",
            seed=7,
        )
