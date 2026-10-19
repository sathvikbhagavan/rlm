from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDIES = ROOT / "human_eval/studies/2026-10-followup"


def test_followup_studies_complete_coverage_and_preserve_cross_checks() -> None:
    questions = {
        json.loads(line)["question_id"]
        for line in (ROOT / "human_eval/generated/canonical-v7/questions.jsonl").read_text().splitlines()
    }
    with (ROOT / "paper_plots/gold/iclr2027/human_validation/item_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        existing = {row["item_id"] for row in csv.DictReader(stream)}

    manifests = [json.loads(path.read_text()) for path in sorted(STUDIES.glob("*.json"))]
    core: list[str] = []
    cross_checks: list[str] = []
    for manifest in manifests:
        expected = (
            manifest["core_question_ids"] + manifest["validation_cross_check_question_ids"]
        )
        assert manifest["question_ids"] == expected
        assert len(expected) == len(set(expected))
        assert set(expected) <= questions
        core.extend(manifest["core_question_ids"])
        cross_checks.extend(manifest["validation_cross_check_question_ids"])

    assert len(core) == len(set(core)) == 56
    assert set(core).isdisjoint(existing)
    assert existing | set(core) == questions
    assert len(cross_checks) == 10
    assert set(cross_checks) <= existing
