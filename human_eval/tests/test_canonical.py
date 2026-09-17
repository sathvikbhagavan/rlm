from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from human_eval.canonical import (
    EXPECTED,
    SUGGESTED_MINUTES_BY_TIER,
    BundleBuilder,
    build_bundle,
)
from human_eval.schema import Question

ROOT = Path(__file__).resolve().parents[2]


def test_real_extraction_is_exact_and_deterministic(tmp_path: Path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    manifest_a = build_bundle(ROOT, first, "dataset-checksum")
    manifest_b = build_bundle(ROOT, second, "dataset-checksum")
    assert manifest_a["question_count"] == 100
    assert manifest_a["taxonomy"] == EXPECTED
    assert manifest_a["questions_sha256"] == manifest_b["questions_sha256"]
    assert (first / "questions.jsonl").read_bytes() == (second / "questions.jsonl").read_bytes()
    public = [json.loads(line) for line in (first / "questions.jsonl").read_text().splitlines()]
    assert all("representation" not in item for item in public)
    assert all("relevant_reaction_indices" not in item for item in public)
    assert all(
        item["suggested_time_minutes"] == SUGGESTED_MINUTES_BY_TIER[item["tier"]] for item in public
    )


def test_schema_round_trip_and_stable_ids():
    questions, _ = BundleBuilder(ROOT, "checksum").build()
    assert len({q.question_id for q in questions}) == 100
    assert [q.question_id for q in questions] == [
        Question.from_dict(q.to_dict()).question_id for q in questions
    ]
    assert Counter(q.tier for q in questions) == Counter(EXPECTED["tiers"])
    warnings = [q for q in questions if q.metadata.get("extraction_warning")]
    assert len(warnings) == 4
    task16 = [q for q in questions if "-task16-" in q.question_id and q.tier == 4]
    sequential = [
        q
        for q in questions
        if q.tier == 4 and ("-task17-" in q.question_id or "-task17b-" in q.question_id)
    ]
    assert len(task16) == 10
    assert len(sequential) == 10
    assert {q.metadata["conceptual_family"] for q in task16} == {"prospective-truncated-synthesis"}
    assert {q.metadata["conceptual_family"] for q in sequential} == {
        "multi-constraint-sequential-template"
    }
