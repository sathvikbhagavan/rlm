from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from human_eval.app import create_app
from human_eval.candidates import import_candidate_pack


def test_question_filters_accept_empty_values_and_combine(
    tmp_path: Path, tiny_bundle: Path, tiny_dataset: Path
):
    app = create_app(
        bundle_dir=tiny_bundle, state_dir=tmp_path / "filters", dataset_path=tiny_dataset
    )
    client = TestClient(app)

    one_result = [
        "/questions?mode=baseline&tier=&category=&status=&search=",
        "/questions?mode=baseline&tier=1",
        "/questions?mode=baseline&tier=1&category=structural-lookup&status=not-started&search=fixture",
    ]
    for url in one_result:
        response = client.get(url)
        assert response.status_code == 200
        assert "1 questions shown." in response.text
        assert "Some filter values were ignored." not in response.text
        assert "Case-insensitive substring search" in response.text
        assert 'role="tooltip"' in response.text

    invalid = client.get(
        "/questions?mode=invalid&tier=invalid&category=invalid&status=invalid"
    )
    assert invalid.status_code == 200
    assert "1 questions shown." in invalid.text
    assert "Some filter values were ignored." in invalid.text
    assert "tier filter was invalid" in invalid.text

    for url in (
        "/questions?mode=baseline&tier=2",
        "/questions?mode=baseline&tier=1&search=absent",
        "/questions?mode=baseline&status=completed",
    ):
        response = client.get(url)
        assert response.status_code == 200
        assert "0 questions shown." in response.text

    headers = {"X-RXH-CSRF": app.state.csrf}
    client.post(
        "/api/save/baseline/rxh-t1-fixture",
        json={"answer_exact": "1"},
        headers=headers,
    )
    assert "1 questions shown." in client.get(
        "/questions?mode=baseline&tier=1&category=structural-lookup&status=started"
    ).text
    client.post(
        "/api/submit/baseline/rxh-t1-fixture",
        json={"answer_exact": "1"},
        headers=headers,
    )
    assert "1 questions shown." in client.get(
        "/questions?mode=baseline&tier=1&category=structural-lookup&status=completed"
    ).text


def test_browser_smoke_leakage_autosave_and_export(
    tmp_path: Path, tiny_bundle: Path, tiny_dataset: Path
):
    state = tmp_path / "state"
    app = create_app(bundle_dir=tiny_bundle, state_dir=state, dataset_path=tiny_dataset)
    app.state.dataset_index.build()
    client = TestClient(app)
    csrf = app.state.csrf
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Benchmark validation" in dashboard.text
    assert "Important safeguard for independent answers" in dashboard.text
    assert "do not let it search or inspect this repository" in dashboard.text
    assert 'id="theme-select"' in dashboard.text
    assert "Prospective plausibility review" not in dashboard.text
    assert "Ground-truth audit" not in dashboard.text
    assert "rxh-theme" in client.get("/static/theme.js").text
    application_js = client.get("/static/app.js").text
    assert "action-row" in application_js
    assert "Does your answer reveal a possible benchmark error?" in application_js
    baseline = client.get("/question/rxh-t1-fixture?mode=baseline")
    assert baseline.status_code == 200 and "987654321" not in baseline.text
    api = client.get("/api/question/rxh-t1-fixture?mode=baseline")
    assert "ground_truth" not in api.text and "987654321" not in api.text
    assert client.get("/question/rxh-t1-fixture?mode=audit").text.count("987654321") >= 1
    assert (
        client.post("/api/save/baseline/rxh-t1-fixture", json={"answer_exact": "1, 2"}).status_code
        == 403
    )
    headers = {"X-RXH-CSRF": csrf}
    assert (
        client.post(
            "/api/save/baseline/rxh-t1-fixture", json={"answer_exact": "1, 2"}, headers=headers
        ).status_code
        == 200
    )
    response = client.post(
        "/api/submit/baseline/rxh-t1-fixture",
        json={"answer_exact": "1, 2", "tools": ["rdkit"], "verified_logic": False},
        headers=headers,
    )
    assert response.status_code == 422
    assert (
        client.post(
            "/api/submit/baseline/rxh-t1-fixture",
            json={"answer_exact": "1, 2", "tools": ["rdkit"], "verified_logic": True},
            headers=headers,
        ).status_code
        == 200
    )
    submitted_page = client.get("/question/rxh-t1-fixture?mode=baseline")
    assert "987654321" in submitted_page.text
    assert "Post-submission reference" in submitted_page.text
    assert "ground_truth_disagreement" in submitted_page.text
    assert (
        client.post(
            "/api/submit/baseline/rxh-t1-fixture",
            json={
                "answer_exact": "1, 2",
                "ground_truth_disagreement": True,
                "rationale": "Stored reference may be incomplete",
            },
            headers=headers,
        ).status_code
        == 200
    )
    revised = app.state.store.draft(
        app.state.profile["annotator_id"], "baseline", "rxh-t1-fixture"
    )
    assert revised["payload"]["ground_truth_disagreement"] is True
    assert revised["payload"]["post_ground_truth_reveal_revision"] is True
    assert client.get("/dataset?page=1").status_code == 200
    assert client.get("/depict/0.png").headers["content-type"] == "image/png"
    assert client.get("/download/evidence/baseline/rxh-t1-fixture").status_code == 403
    assert client.get("/download/evidence/audit/rxh-t1-fixture").status_code == 200
    assert client.get("/download/../../etc/passwd").status_code == 404
    assert (
        client.post(
            "/api/attachment/baseline/rxh-t1-fixture",
            files={"attachment": ("note.txt", b"fixture evidence", "text/plain")},
            headers=headers,
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/profile",
            json={"role": "student", "name": "must-be-dropped"},
            headers=headers,
        ).status_code
        == 200
    )
    exported = client.get("/export")
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        assert {
            "annotations.csv",
            "annotations.jsonl",
            "manifest.json",
            "revisions.jsonl",
            "timing.jsonl",
            "attachments.json",
            "assignments.jsonl",
            "study_manifests.json",
        } <= set(archive.namelist())
        all_text = b"".join(archive.read(x) for x in archive.namelist())
        assert b"987654321" not in all_text and b"admin_ground_truth" not in all_text
        assert b"fixture evidence" in all_text and b"must-be-dropped" not in all_text
        annotation = json.loads(archive.read("annotations.jsonl").splitlines()[0])
        assert annotation["annotation_context"]["questions_sha256"] == "q"


def test_export_can_restore_and_merge_idempotently(
    tmp_path: Path, tiny_bundle: Path, tiny_dataset: Path
):
    source = create_app(
        bundle_dir=tiny_bundle, state_dir=tmp_path / "source", dataset_path=tiny_dataset
    )
    source_client = TestClient(source)
    headers = {"X-RXH-CSRF": source.state.csrf}
    source_client.post(
        "/api/save/baseline/rxh-t1-fixture",
        json={"answer_exact": "4, 5", "rationale": "saved work"},
        headers=headers,
    )
    source_client.post(
        "/api/attachment/baseline/rxh-t1-fixture",
        files={"attachment": ("note.txt", b"restored attachment", "text/plain")},
        headers=headers,
    )
    exported = source_client.get("/export").content
    source_id = source.state.profile["annotator_id"]

    target = create_app(
        bundle_dir=tiny_bundle, state_dir=tmp_path / "target", dataset_path=tiny_dataset
    )
    target_client = TestClient(target)
    target_headers = {"X-RXH-CSRF": target.state.csrf}
    assert target_client.get("/restore").status_code == 200
    first = target_client.post(
        "/api/restore",
        files={"archive": ("annotations.zip", exported, "application/zip")},
        headers=target_headers,
    )
    assert first.status_code == 200
    assert first.json()["annotator_id"] == source_id
    assert target.state.profile["annotator_id"] == source_id
    draft = target.state.store.draft(source_id, "baseline", "rxh-t1-fixture")
    assert draft["payload"]["answer_exact"] == "4, 5"
    assert target.state.store.export_attachments(source_id)[0][1] == b"restored attachment"

    second = target_client.post(
        "/api/restore",
        files={"archive": ("annotations.zip", exported, "application/zip")},
        headers=target_headers,
    )
    assert second.status_code == 200
    assert second.json()["revisions"] == 0

    with zipfile.ZipFile(io.BytesIO(exported)) as original:
        members = {name: original.read(name) for name in original.namelist()}
    members["annotations.jsonl"] += b"{}\n"
    damaged = io.BytesIO()
    with zipfile.ZipFile(damaged, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    rejected = target_client.post(
        "/api/restore",
        files={"archive": ("damaged.zip", damaged.getvalue(), "application/zip")},
        headers=target_headers,
    )
    assert rejected.status_code == 422 and "mismatch" in rejected.text.lower()


def test_prospective_response_has_no_unblinding(
    tmp_path: Path, tiny_bundle: Path, tiny_dataset: Path
):
    state = tmp_path / "state"
    app = create_app(bundle_dir=tiny_bundle, state_dir=state, dataset_path=tiny_dataset)
    pack = tmp_path / "pack.json"
    pack.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "pack_id": "p",
                "version": "v",
                "seed": 1,
                "candidates": [
                    {
                        "candidate_id": "source-secret",
                        "question_id": "rxh-t1-fixture",
                        "candidate_answer": [[0]],
                        "model_id": "SECRET_MODEL",
                        "evaluator_outcome": "incorrect",
                    }
                ],
            }
        )
    )
    import_candidate_pack(app.state.store, pack, state / "admin")
    client = TestClient(app)
    listing = client.get("/prospective")
    assert (
        "SECRET_MODEL" not in listing.text
        and "source-secret" not in listing.text
        and "incorrect" not in listing.text
    )
    from human_eval.candidates import public_candidates

    public_id = public_candidates(app.state.store)[0]["candidate_id"]
    detail = client.get(f"/prospective/{public_id}")
    assert detail.status_code == 200 and "SECRET_MODEL" not in detail.text
