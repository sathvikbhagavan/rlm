from __future__ import annotations

import json
from pathlib import Path

from human_eval.db import Store


def test_autosave_resume_revision_and_anonymous_id(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3")
    profile = store.profile()
    assert profile["annotator_id"].startswith("rxh-")
    store.save_draft(
        profile["annotator_id"],
        "baseline",
        "q1",
        {"answer_exact": "1"},
        annotation_context={"questions_sha256": "questions-v1"},
    )
    assert (
        Store(tmp_path / "state.sqlite3").draft(profile["annotator_id"], "baseline", "q1")[
            "payload"
        ]["answer_exact"]
        == "1"
    )
    store.save_draft(profile["annotator_id"], "baseline", "q1", {"answer_exact": "2"}, submit=True)
    first_submitted = store.draft(profile["annotator_id"], "baseline", "q1")["submitted_at"]
    store.save_draft(profile["annotator_id"], "baseline", "q1", {"answer_exact": "3"}, submit=True)
    assert store.draft(profile["annotator_id"], "baseline", "q1")["submitted_at"] == first_submitted
    with store.connect() as db:
        revisions = db.execute(
            "SELECT payload_json,annotation_context_json FROM revisions ORDER BY revision_id"
        ).fetchall()
    assert [json.loads(x[0])["answer_exact"] for x in revisions] == ["1", "2", "3"]
    assert json.loads(revisions[0][1])["questions_sha256"] == "questions-v1"


def test_pause_resume_timing_accounting(tmp_path: Path, monkeypatch):
    store = Store(tmp_path / "state.sqlite3")
    user = store.profile()["annotator_id"]
    store.timer(user, "baseline", "q1", "start")
    result = store.timer(user, "baseline", "q1", "heartbeat", elapsed_seconds=2)
    assert result["active_seconds"] >= 0
    paused = store.timer(user, "baseline", "q1", "pause", elapsed_seconds=1)
    assert paused["wall_seconds"] >= paused["active_seconds"]
    store.timer(user, "baseline", "q1", "start")
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM timing_sessions").fetchone()[0] == 2


def test_latest_submitted_payload_only_uses_eligible_completed_items(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3")
    user = store.profile()["annotator_id"]
    store.save_draft(user, "baseline", "q-draft", {"confidence": "1"})
    store.save_draft(user, "baseline", "q-other", {"confidence": "2"}, submit=True)
    store.save_draft(user, "baseline", "q-same", {"confidence": "4"}, submit=True)

    latest = store.latest_submitted_payload(
        user,
        "baseline",
        ["q-draft", "q-same", "q-current"],
        exclude_item_id="q-current",
    )
    assert latest and latest["item_id"] == "q-same"
    assert latest["payload"]["confidence"] == "4"
