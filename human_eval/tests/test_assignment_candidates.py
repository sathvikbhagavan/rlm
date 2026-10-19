from __future__ import annotations

import json
from pathlib import Path

import pytest

from human_eval.assignments import assign_questions, blinded_order, resolve_assignment
from human_eval.candidates import import_candidate_pack, public_candidates
from human_eval.db import Store


def test_assignment_and_order_deterministic_and_stratified():
    questions = [{"question_id": f"q{x}", "category": "a" if x < 5 else "b"} for x in range(10)]
    one = assign_questions(questions, seed=42, quotas={"a": 2, "b": 3}, annotator_id="u1")
    assert one == assign_questions(questions, seed=42, quotas={"a": 2, "b": 3}, annotator_id="u1")
    assert len(one) == 5 and one != assign_questions(
        questions, seed=42, quotas={"a": 2, "b": 3}, annotator_id="u2"
    )
    assert blinded_order(["c", "a", "b"], seed=9, annotator_id="u") == blinded_order(
        ["b", "c", "a"], seed=9, annotator_id="u"
    )


def test_explicit_assignment_preserves_order_and_rejects_invalid_ids():
    questions = [{"question_id": f"q{x}", "category": "a"} for x in range(3)]

    assert resolve_assignment(
        questions, seed=42, annotator_id="u", question_ids=["q2", "q0"]
    ) == ["q2", "q0"]
    with pytest.raises(ValueError, match="duplicate"):
        resolve_assignment(questions, seed=42, annotator_id="u", question_ids=["q0", "q0"])
    with pytest.raises(ValueError, match="unknown"):
        resolve_assignment(questions, seed=42, annotator_id="u", question_ids=["missing"])


def test_candidate_pack_is_blinded_recursively(tmp_path: Path):
    pack = {
        "schema_version": "1",
        "pack_id": "secret-model-pack",
        "version": "1",
        "seed": 7,
        "candidates": [
            {
                "candidate_id": "model-name-run-1",
                "question_id": "q",
                "candidate_answer": [[1, 2]],
                "model_id": "secret-model",
                "control_type": "positive",
                "duplicate_group": "d",
                "metadata": {"run_id": "secret-run"},
            }
        ],
    }
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(pack))
    store = Store(tmp_path / "db.sqlite")
    result = import_candidate_pack(store, path, tmp_path / "admin")
    public = json.dumps(public_candidates(store))
    assert (
        "secret" not in public
        and "positive" not in public
        and "duplicate" not in public
        and "model-name" not in public
    )
    assert public_candidates(store)[0]["candidate_id"].startswith("cand-")
    admin = Path(result["unblinding_path"])
    assert admin.stat().st_mode & 0o777 == 0o600
    assert "secret-model" in admin.read_text() and "model-name-run-1" in admin.read_text()
