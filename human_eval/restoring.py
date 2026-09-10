from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .db import Store

MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
VALID_MODES = {"baseline", "audit", "prospective"}


def restore_export(store: Store, archive_bytes: bytes) -> dict[str, Any]:
    """Validate and merge an annotator export into local SQLite state.

    Import is idempotent. It can replace a freshly generated, unused anonymous
    profile, but refuses to mix two annotators in a state directory that already
    contains work.
    """
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("Restore ZIP exceeds the 50 MiB compressed limit")
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as error:
        raise ValueError("Not a valid ZIP archive") from error
    with archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Restore ZIP contains duplicate member names")
        if any(not safe_member_name(name) for name in names):
            raise ValueError("Restore ZIP contains an unsafe member path")
        if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Restore ZIP exceeds the 100 MiB uncompressed limit")
        if "manifest.json" not in names:
            raise ValueError("Restore ZIP has no manifest.json")
        manifest = parse_json(archive.read("manifest.json"), "manifest.json")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("contents"), dict):
            raise ValueError("Invalid RxnHaystack export manifest")
        annotator_id = str(manifest.get("anonymous_annotator_id", ""))
        if not re.fullmatch(r"rxh-[0-9a-f]{12}", annotator_id):
            raise ValueError("Invalid anonymous annotator ID")
        declared = set(manifest["contents"])
        if set(names) != declared | {"manifest.json"}:
            raise ValueError("Restore ZIP members do not match its manifest")
        for name, metadata in manifest["contents"].items():
            content = archive.read(name)
            if len(content) != int(metadata["bytes"]):
                raise ValueError(f"Size mismatch for {name}")
            if hashlib.sha256(content).hexdigest() != metadata["sha256"]:
                raise ValueError(f"Checksum mismatch for {name}")

        annotations = parse_jsonl(archive.read("annotations.jsonl"), "annotations.jsonl")
        revisions = parse_jsonl(archive.read("revisions.jsonl"), "revisions.jsonl")
        timing = parse_jsonl(archive.read("timing.jsonl"), "timing.jsonl")
        assignments = parse_jsonl(archive.read("assignments.jsonl"), "assignments.jsonl")
        studies = parse_json(archive.read("study_manifests.json"), "study_manifests.json")
        attachment_manifest = parse_json(archive.read("attachments.json"), "attachments.json")
        if not isinstance(studies, list) or not isinstance(attachment_manifest, list):
            raise ValueError("Invalid study or attachment manifest")
        attachment_data = load_attachments(archive, attachment_manifest)

    replace_unused_profile(store, annotator_id, manifest.get("expertise", {}))
    counts = merge_rows(
        store,
        annotator_id,
        annotations,
        revisions,
        timing,
        assignments,
        studies,
        attachment_manifest,
        attachment_data,
    )
    return {"restored": True, "annotator_id": annotator_id, **counts}


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and "\\" not in name and not path.is_absolute() and ".." not in path.parts


def parse_json(content: bytes, name: str) -> Any:
    try:
        return json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON in {name}") from error


def parse_jsonl(content: bytes, name: str) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = content.decode("utf-8").splitlines()
        for line in lines:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Non-object row in {name}")
                rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSONL in {name}") from error
    return rows


def replace_unused_profile(store: Store, annotator_id: str, expertise: Any) -> None:
    if not isinstance(expertise, dict):
        raise ValueError("Invalid expertise metadata")
    current = store.profile()
    current_id = current["annotator_id"]
    if current_id == annotator_id:
        store.save_expertise(annotator_id, expertise)
        return
    with store.connect() as db:
        tables = ("drafts", "revisions", "timing_sessions", "assignments", "attachments")
        if any(
            db.execute(f"SELECT 1 FROM {table} WHERE annotator_id=? LIMIT 1", (current_id,)).fetchone()
            for table in tables
        ):
            raise ValueError(
                "This state directory already contains work for a different anonymous annotator"
            )
        db.execute("DELETE FROM profiles WHERE annotator_id=?", (current_id,))
        db.execute(
            "INSERT OR REPLACE INTO profiles VALUES (?,?,?)",
            (annotator_id, current["created_at"], json.dumps(expertise, sort_keys=True)),
        )


def load_attachments(
    archive: zipfile.ZipFile, attachment_manifest: list[Any]
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for raw in attachment_manifest:
        if not isinstance(raw, dict):
            raise ValueError("Invalid attachment metadata")
        attachment_id = str(raw.get("attachment_id", ""))
        original_name = Path(str(raw.get("original_name", "")).replace("\\", "/")).name
        expected = f"attachments/{attachment_id}_{original_name}"
        if expected not in archive.namelist():
            raise ValueError(f"Missing attachment member {expected}")
        result[attachment_id] = archive.read(expected)
    return result


def merge_rows(
    store: Store,
    annotator_id: str,
    annotations: list[dict[str, Any]],
    revisions: list[dict[str, Any]],
    timing: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    studies: list[dict[str, Any]],
    attachment_manifest: list[dict[str, Any]],
    attachment_data: dict[str, bytes],
) -> dict[str, int]:
    restored = {"annotations": 0, "revisions": 0, "timing_sessions": 0, "attachments": 0}
    with store.connect() as db:
        for row in annotations:
            mode, item_id = validate_item(row)
            payload = row.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("Annotation payload must be an object")
            updated_at = str(row["updated_at"])
            current = db.execute(
                "SELECT updated_at,submitted_at FROM drafts WHERE annotator_id=? AND mode=? AND item_id=?",
                (annotator_id, mode, item_id),
            ).fetchone()
            if current is not None and current["updated_at"] > updated_at:
                continue
            submitted_values = [
                str(value)
                for value in (row.get("submitted_at"), current["submitted_at"] if current else None)
                if value
            ]
            submitted_at = min(submitted_values) if submitted_values else None
            context = row.get("annotation_context", {})
            if not isinstance(context, dict):
                raise ValueError("Annotation context must be an object")
            db.execute(
                """INSERT INTO drafts(
                  annotator_id,mode,item_id,payload_json,updated_at,submitted_at,annotation_context_json
                ) VALUES (?,?,?,?,?,?,?) ON CONFLICT(annotator_id,mode,item_id) DO UPDATE SET
                  payload_json=excluded.payload_json, updated_at=excluded.updated_at,
                  submitted_at=excluded.submitted_at,
                  annotation_context_json=excluded.annotation_context_json""",
                (
                    annotator_id,
                    mode,
                    item_id,
                    json.dumps(payload, sort_keys=True, ensure_ascii=False),
                    updated_at,
                    submitted_at,
                    json.dumps(context, sort_keys=True),
                ),
            )
            restored["annotations"] += 1

        for row in revisions:
            mode, item_id = validate_item(row)
            payload_json = canonical_json_field(row, "payload_json")
            context_json = canonical_json_field(row, "annotation_context_json", default={})
            values = (
                annotator_id,
                mode,
                item_id,
                payload_json,
                str(row["created_at"]),
                str(row["reason"]),
                context_json,
            )
            exists = db.execute(
                """SELECT 1 FROM revisions WHERE annotator_id=? AND mode=? AND item_id=?
                AND payload_json=? AND created_at=? AND reason=? AND annotation_context_json=?""",
                values,
            ).fetchone()
            if exists is None:
                db.execute(
                    """INSERT INTO revisions(
                      annotator_id,mode,item_id,payload_json,created_at,reason,annotation_context_json
                    ) VALUES (?,?,?,?,?,?,?)""",
                    values,
                )
                restored["revisions"] += 1

        for row in timing:
            mode, item_id = validate_item(row)
            values = (
                annotator_id,
                mode,
                item_id,
                str(row["started_at"]),
                row.get("ended_at"),
                str(row["last_tick_at"]),
                float(row.get("active_seconds", 0)),
                float(row.get("wall_seconds", 0)),
                str(row["state"]),
            )
            exists = db.execute(
                """SELECT 1 FROM timing_sessions WHERE annotator_id=? AND mode=? AND item_id=?
                AND started_at=? AND ended_at IS ? AND last_tick_at=? AND active_seconds=?
                AND wall_seconds=? AND state=?""",
                values,
            ).fetchone()
            if exists is None:
                db.execute(
                    """INSERT INTO timing_sessions(
                      annotator_id,mode,item_id,started_at,ended_at,last_tick_at,
                      active_seconds,wall_seconds,state
                    ) VALUES (?,?,?,?,?,?,?,?,?)""",
                    values,
                )
                restored["timing_sessions"] += 1

        for study in studies:
            study_id = str(study["study_id"])
            db.execute(
                "INSERT OR IGNORE INTO studies VALUES (?,?,?)",
                (study_id, json.dumps(study, sort_keys=True), str(study.get("created_at", "restored"))),
            )
        for row in assignments:
            db.execute(
                "INSERT OR IGNORE INTO assignments VALUES (?,?,?,?,?)",
                (
                    annotator_id,
                    str(row["study_id"]),
                    str(row["mode"]),
                    str(row["item_id"]),
                    int(row["sequence"]),
                ),
            )
        restore_attachments(
            db, store, annotator_id, attachment_manifest, attachment_data, restored
        )
    return restored


def validate_item(row: dict[str, Any]) -> tuple[str, str]:
    mode = str(row.get("mode", ""))
    item_id = str(row.get("item_id", ""))
    if mode not in VALID_MODES or not item_id:
        raise ValueError("Invalid annotation mode or item ID")
    return mode, item_id


def canonical_json_field(row: dict[str, Any], key: str, default: Any = None) -> str:
    value = row.get(key, default)
    if isinstance(value, str):
        value = json.loads(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def restore_attachments(
    db: sqlite3.Connection,
    store: Store,
    annotator_id: str,
    metadata_rows: list[dict[str, Any]],
    contents: dict[str, bytes],
    restored: dict[str, int],
) -> None:
    directory = (store.path.parent / "attachments").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    allowed = {".txt", ".csv", ".json", ".pdf", ".png", ".jpg", ".jpeg"}
    for row in metadata_rows:
        attachment_id = str(row["attachment_id"])
        if not re.fullmatch(r"[0-9a-f]{24}", attachment_id):
            raise ValueError("Invalid attachment ID")
        if db.execute(
            "SELECT 1 FROM attachments WHERE attachment_id=?", (attachment_id,)
        ).fetchone():
            continue
        original_name = Path(str(row["original_name"]).replace("\\", "/")).name
        extension = Path(original_name).suffix.lower()
        if extension not in allowed:
            raise ValueError("Disallowed restored attachment type")
        target = (directory / f"{attachment_id}{extension}").resolve()
        if not target.is_relative_to(directory):
            raise ValueError("Unsafe restored attachment path")
        target.write_bytes(contents[attachment_id])
        db.execute(
            "INSERT INTO attachments VALUES (?,?,?,?,?,?,?,?)",
            (
                attachment_id,
                annotator_id,
                str(row["mode"]),
                str(row["item_id"]),
                original_name,
                str(row["media_type"]),
                str(target),
                str(row["created_at"]),
            ),
        )
        restored["attachments"] += 1
