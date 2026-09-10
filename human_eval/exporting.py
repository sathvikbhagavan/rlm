from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from typing import Any

from . import SCHEMA_VERSION
from .db import Store


def export_zip(store: Store, annotator_id: str, bundle_manifest: dict[str, Any]) -> bytes:
    data = store.all_export_rows(annotator_id)
    profile = store.profile()
    first_submissions: dict[tuple[str, str], dict[str, Any]] = {}
    for revision in data["revisions"]:
        key = (revision["mode"], revision["item_id"])
        if revision["reason"] == "submission" and key not in first_submissions:
            first_submissions[key] = {
                "payload": json.loads(revision["payload_json"]),
                "annotation_context": json.loads(revision["annotation_context_json"]),
            }
    lossless = []
    for row in data["drafts"]:
        lossless.append(
            {
                **{key: row[key] for key in ("mode", "item_id", "updated_at", "submitted_at")},
                "payload": json.loads(row["payload_json"]),
                "annotation_context": json.loads(row["annotation_context_json"]),
                "first_submission_payload": (
                    first_submissions.get((row["mode"], row["item_id"])) or {}
                ).get("payload"),
                "first_submission_context": (
                    first_submissions.get((row["mode"], row["item_id"])) or {}
                ).get("annotation_context"),
            }
        )
    tidy = []
    for row in lossless:
        payload = (
            row["first_submission_payload"]
            if row["mode"] == "baseline" and row["first_submission_payload"]
            else row["payload"]
        )
        tidy.append(
            {
                "annotator_id": annotator_id,
                "mode": row["mode"],
                "item_id": row["item_id"],
                "submitted_at": row["submitted_at"] or "",
                "answer_exact": payload.get("answer_exact", ""),
                "abstention": payload.get("abstention", ""),
                "confidence": payload.get("confidence", ""),
                "label": payload.get("overall_label", ""),
                "offline_minutes": payload.get("offline_minutes", ""),
                "tools": "|".join(payload.get("tools", []))
                if isinstance(payload.get("tools"), list)
                else payload.get("tools", ""),
                "rationale": payload.get("rationale", ""),
                "has_later_revision": row["payload"] != row["first_submission_payload"]
                if row["first_submission_payload"]
                else False,
            }
        )
    files: dict[str, bytes] = {}
    files["annotations.jsonl"] = (
        "\n".join(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in lossless) + "\n"
    ).encode()
    files["revisions.jsonl"] = (
        "\n".join(json.dumps(x, sort_keys=True) for x in data["revisions"])
        + ("\n" if data["revisions"] else "")
    ).encode()
    files["timing.jsonl"] = (
        "\n".join(json.dumps(x, sort_keys=True) for x in data["timing"])
        + ("\n" if data["timing"] else "")
    ).encode()
    files["assignments.jsonl"] = (
        "\n".join(json.dumps(x, sort_keys=True) for x in data["assignments"])
        + ("\n" if data["assignments"] else "")
    ).encode()
    study_manifests = []
    for row in data["studies"]:
        study = json.loads(row["manifest_json"])
        if isinstance(study.get("annotators"), list):
            study["annotators"] = [
                value
                for value in study["annotators"]
                if str(value.get("annotator_id")) == annotator_id
            ]
        study_manifests.append(study)
    files["study_manifests.json"] = (
        json.dumps(study_manifests, indent=2, sort_keys=True) + "\n"
    ).encode()
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(tidy[0]) if tidy else ["annotator_id", "mode", "item_id"]
    )
    writer.writeheader()
    writer.writerows(tidy)
    files["annotations.csv"] = buffer.getvalue().encode()
    attachment_manifest = []
    for metadata, content in store.export_attachments(annotator_id):
        name = f"attachments/{metadata['attachment_id']}_{metadata['original_name']}"
        files[name] = content
        attachment_manifest.append(
            {
                key: metadata[key]
                for key in (
                    "attachment_id",
                    "mode",
                    "item_id",
                    "original_name",
                    "media_type",
                    "created_at",
                )
            }
        )
    files["attachments.json"] = (
        json.dumps(attachment_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(UTC).isoformat(),
        "anonymous_annotator_id": annotator_id,
        "expertise": json.loads(profile["expertise_json"]),
        "bundle": {
            key: bundle_manifest.get(key)
            for key in ("bundle_version", "questions_sha256", "dataset_sha256")
        },
        "contents": {},
    }
    for name, content in files.items():
        manifest["contents"][name] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def export_filename(annotator_id: str) -> str:
    return f"rxnhaystack_human_annotation_{annotator_id}.zip"
