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
        item["suggested_time_minutes"] == SUGGESTED_MINUTES_BY_TIER[item["tier"]]
        for item in public
    )
    ring_question = next(item for item in public if item["question_id"] == "rxh-t2-task3-1")
    assert "max(rings(product_component)) minus" in ring_question["canonical_prompt"]


def test_schema_round_trip_and_stable_ids():
    questions, _ = BundleBuilder(ROOT, "checksum").build()
    assert len({q.question_id for q in questions}) == 100
    assert [q.question_id for q in questions] == [
        Question.from_dict(q.to_dict()).question_id for q in questions
    ]
    assert Counter(q.tier for q in questions) == Counter(EXPECTED["tiers"])
    warnings = [q for q in questions if q.metadata.get("extraction_warning")]
    assert len(warnings) == 4
