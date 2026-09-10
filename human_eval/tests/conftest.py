from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "clean.txt"
    path.write_text("CCO>O>CC=O\nBrC.CN>>CNC\nN#N>>N=N\n", encoding="utf-8")
    return path


@pytest.fixture
def tiny_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    admin = bundle / "admin"
    admin.mkdir(parents=True)
    question = {
        "schema_version": "1.0.0",
        "question_id": "rxh-t1-fixture",
        "tier": 1,
        "category": "structural-lookup",
        "subcategory": "fixture",
        "canonical_prompt": "Find the fixture product.",
        "answer_type": "index_set",
        "source_files": ["fixture.py"],
        "dataset_sha256": "abc",
        "ground_truth_ref": "gt:rxh-t1-fixture",
        "suggested_time_minutes": 10,
        "scoring": {},
        "metadata": {},
    }
    truth = {
        "schema_version": "1.0.0",
        "ground_truth_ref": "gt:rxh-t1-fixture",
        "question_id": "rxh-t1-fixture",
        "representation": [987654321],
        "relevant_reaction_indices": [987654321],
        "evaluator": {},
        "source_files": ["fixture.py"],
    }
    (bundle / "questions.jsonl").write_text(json.dumps(question) + "\n")
    (admin / "ground_truth.jsonl").write_text(json.dumps(truth) + "\n")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_version": "fixture",
                "dataset_sha256": "abc",
                "questions_sha256": "q",
                "ground_truth_location": "admin/ground_truth.jsonl",
                "audit_sampling": {"seed": 20270910, "method": "hash"},
            }
        )
    )
    return bundle
