from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS profiles (
  annotator_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, expertise_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS studies (
  study_id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assignments (
  annotator_id TEXT NOT NULL, study_id TEXT NOT NULL, mode TEXT NOT NULL, item_id TEXT NOT NULL,
  sequence INTEGER NOT NULL, PRIMARY KEY (annotator_id, study_id, mode, item_id)
);
CREATE TABLE IF NOT EXISTS drafts (
  annotator_id TEXT NOT NULL, mode TEXT NOT NULL, item_id TEXT NOT NULL,
  payload_json TEXT NOT NULL, updated_at TEXT NOT NULL, submitted_at TEXT,
  annotation_context_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (annotator_id, mode, item_id)
);
CREATE TABLE IF NOT EXISTS revisions (
  revision_id INTEGER PRIMARY KEY AUTOINCREMENT, annotator_id TEXT NOT NULL, mode TEXT NOT NULL,
  item_id TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL, reason TEXT NOT NULL,
  annotation_context_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS timing_sessions (
  session_id INTEGER PRIMARY KEY AUTOINCREMENT, annotator_id TEXT NOT NULL, mode TEXT NOT NULL,
  item_id TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT, last_tick_at TEXT NOT NULL,
  active_seconds REAL NOT NULL DEFAULT 0, wall_seconds REAL NOT NULL DEFAULT 0, state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
  candidate_id TEXT PRIMARY KEY, pack_id TEXT NOT NULL, question_id TEXT NOT NULL,
  candidate_json TEXT NOT NULL, control_type TEXT, duplicate_group TEXT, public_order INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS attachments (
  attachment_id TEXT PRIMARY KEY, annotator_id TEXT NOT NULL, mode TEXT NOT NULL,
  item_id TEXT NOT NULL, original_name TEXT NOT NULL, media_type TEXT NOT NULL,
  stored_path TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_drafts_profile_mode ON drafts(annotator_id, mode);
CREATE INDEX IF NOT EXISTS idx_timing_item ON timing_sessions(annotator_id, mode, item_id);
CREATE INDEX IF NOT EXISTS idx_candidates_pack_order ON candidates(pack_id, public_order);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)
            # Additive migration for pilot databases created before bundle provenance
            # was stamped on every autosave and submission.
            for table in ("drafts", "revisions"):
                columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                if "annotation_context_json" not in columns:
                    db.execute(
                        f"ALTER TABLE {table} ADD COLUMN annotation_context_json "
                        "TEXT NOT NULL DEFAULT '{}'"
                    )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def profile(self) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM profiles ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                anonymous_id = f"rxh-{secrets.token_hex(6)}"
                db.execute("INSERT INTO profiles VALUES (?, ?, '{}')", (anonymous_id, utcnow()))
                row = db.execute(
                    "SELECT * FROM profiles WHERE annotator_id=?", (anonymous_id,)
                ).fetchone()
            return dict(row)

    def save_expertise(self, annotator_id: str, payload: dict[str, Any]) -> None:
        allowed = {"role", "years_chemistry", "reaction_informatics", "uspto_familiarity"}
        clean = {key: value for key, value in payload.items() if key in allowed}
        with self.connect() as db:
            db.execute(
                "UPDATE profiles SET expertise_json=? WHERE annotator_id=?",
                (json.dumps(clean, sort_keys=True), annotator_id),
            )

    def save_draft(
        self,
        annotator_id: str,
        mode: str,
        item_id: str,
        payload: dict[str, Any],
        *,
        submit: bool = False,
        annotation_context: dict[str, Any] | None = None,
    ) -> None:
        now = utcnow()
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        context_encoded = json.dumps(annotation_context or {}, sort_keys=True)
        with self.connect() as db:
            previous = db.execute(
                "SELECT payload_json, submitted_at FROM drafts WHERE annotator_id=? AND mode=? AND item_id=?",
                (annotator_id, mode, item_id),
            ).fetchone()
            submitted_at = (
                previous["submitted_at"]
                if previous and previous["submitted_at"]
                else (now if submit else None)
            )
            db.execute(
                """INSERT INTO drafts(
                  annotator_id,mode,item_id,payload_json,updated_at,submitted_at,annotation_context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
              ON CONFLICT(annotator_id,mode,item_id) DO UPDATE SET payload_json=excluded.payload_json,
              updated_at=excluded.updated_at, submitted_at=excluded.submitted_at,
              annotation_context_json=excluded.annotation_context_json""",
                (annotator_id, mode, item_id, encoded, now, submitted_at, context_encoded),
            )
            if submit or previous is None or previous["payload_json"] != encoded:
                db.execute(
                    """INSERT INTO revisions(
                      annotator_id,mode,item_id,payload_json,created_at,reason,annotation_context_json
                    ) VALUES (?,?,?,?,?,?,?)""",
                    (
                        annotator_id,
                        mode,
                        item_id,
                        encoded,
                        now,
                        "submission" if submit else "autosave",
                        context_encoded,
                    ),
                )

    def draft(self, annotator_id: str, mode: str, item_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM drafts WHERE annotator_id=? AND mode=? AND item_id=?",
                (annotator_id, mode, item_id),
            ).fetchone()
        if row is None:
            return {
                "payload": {},
                "submitted_at": None,
                "updated_at": None,
                "annotation_context": {},
            }
        return {
            "payload": json.loads(row["payload_json"]),
            "submitted_at": row["submitted_at"],
            "updated_at": row["updated_at"],
            "annotation_context": json.loads(row["annotation_context_json"]),
        }

    def progress(self, annotator_id: str, mode: str) -> dict[str, int]:
        with self.connect() as db:
            row = db.execute(
                "SELECT COUNT(*) total, SUM(submitted_at IS NOT NULL) completed FROM drafts WHERE annotator_id=? AND mode=?",
                (annotator_id, mode),
            ).fetchone()
        return {"started": int(row["total"]), "completed": int(row["completed"] or 0)}

    def timer(
        self, annotator_id: str, mode: str, item_id: str, action: str, *, elapsed_seconds: float = 0
    ) -> dict[str, float | str | None]:
        now = datetime.now(UTC)
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM timing_sessions WHERE annotator_id=? AND mode=? AND item_id=? AND state='running' ORDER BY session_id DESC LIMIT 1",
                (annotator_id, mode, item_id),
            ).fetchone()
            if action == "start" and row is None:
                stamp = now.isoformat()
                db.execute(
                    "INSERT INTO timing_sessions(annotator_id,mode,item_id,started_at,last_tick_at,state) VALUES (?,?,?,?,?,'running')",
                    (annotator_id, mode, item_id, stamp, stamp),
                )
            elif action in {"heartbeat", "pause"} and row is not None:
                last = datetime.fromisoformat(row["last_tick_at"])
                observed = max(0.0, (now - last).total_seconds())
                increment = min(observed, max(0.0, elapsed_seconds), 60.0)
                active = float(row["active_seconds"]) + increment
                if action == "pause":
                    wall = (now - datetime.fromisoformat(row["started_at"])).total_seconds()
                    db.execute(
                        "UPDATE timing_sessions SET active_seconds=?,wall_seconds=?,last_tick_at=?,ended_at=?,state='paused' WHERE session_id=?",
                        (active, wall, now.isoformat(), now.isoformat(), row["session_id"]),
                    )
                else:
                    db.execute(
                        "UPDATE timing_sessions SET active_seconds=?,last_tick_at=? WHERE session_id=?",
                        (active, now.isoformat(), row["session_id"]),
                    )
            totals = db.execute(
                "SELECT COALESCE(SUM(active_seconds),0) active, COALESCE(SUM(CASE WHEN state='running' THEN (julianday(?) - julianday(started_at))*86400 ELSE wall_seconds END),0) wall FROM timing_sessions WHERE annotator_id=? AND mode=? AND item_id=?",
                (now.isoformat(), annotator_id, mode, item_id),
            ).fetchone()
        return {
            "active_seconds": round(float(totals["active"]), 3),
            "wall_seconds": round(float(totals["wall"]), 3),
        }

    def all_export_rows(self, annotator_id: str) -> dict[str, list[dict[str, Any]]]:
        with self.connect() as db:
            drafts = [
                dict(x)
                for x in db.execute(
                    "SELECT * FROM drafts WHERE annotator_id=? ORDER BY mode,item_id",
                    (annotator_id,),
                )
            ]
            revisions = [
                dict(x)
                for x in db.execute(
                    "SELECT * FROM revisions WHERE annotator_id=? ORDER BY revision_id",
                    (annotator_id,),
                )
            ]
            timing = [
                dict(x)
                for x in db.execute(
                    "SELECT * FROM timing_sessions WHERE annotator_id=? ORDER BY session_id",
                    (annotator_id,),
                )
            ]
            assignments = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM assignments WHERE annotator_id=? ORDER BY study_id,mode,sequence",
                    (annotator_id,),
                )
            ]
            studies = [
                dict(row)
                for row in db.execute(
                    "SELECT DISTINCT s.* FROM studies s JOIN assignments a ON a.study_id=s.study_id WHERE a.annotator_id=? ORDER BY s.study_id",
                    (annotator_id,),
                )
            ]
        for row in timing:
            if row["state"] == "running":
                started = datetime.fromisoformat(row["started_at"])
                last_tick = datetime.fromisoformat(row["last_tick_at"])
                row["wall_seconds"] = max(0.0, (last_tick - started).total_seconds())
                row["ended_at"] = row["last_tick_at"]
                row["state"] = "interrupted_at_last_heartbeat"
        return {
            "drafts": drafts,
            "revisions": revisions,
            "timing": timing,
            "assignments": assignments,
            "studies": studies,
        }

    def add_attachment(
        self,
        annotator_id: str,
        mode: str,
        item_id: str,
        original_name: str,
        media_type: str,
        content: bytes,
    ) -> dict[str, str]:
        safe_name = Path(original_name.replace("\\", "/")).name
        extension = Path(safe_name).suffix.lower()
        allowed = {".txt", ".csv", ".json", ".pdf", ".png", ".jpg", ".jpeg"}
        if extension not in allowed:
            raise ValueError(f"Attachment type not allowed: {extension}")
        attachment_id = secrets.token_hex(12)
        directory = self.path.parent / "attachments"
        directory.mkdir(parents=True, exist_ok=True)
        target = (directory / f"{attachment_id}{extension}").resolve()
        if not target.is_relative_to(directory.resolve()):
            raise ValueError("Unsafe attachment path")
        target.write_bytes(content)
        with self.connect() as db:
            db.execute(
                "INSERT INTO attachments VALUES (?,?,?,?,?,?,?,?)",
                (
                    attachment_id,
                    annotator_id,
                    mode,
                    item_id,
                    safe_name,
                    media_type,
                    str(target),
                    utcnow(),
                ),
            )
        return {"attachment_id": attachment_id, "name": safe_name}

    def export_attachments(self, annotator_id: str) -> list[tuple[dict[str, Any], bytes]]:
        root = (self.path.parent / "attachments").resolve()
        with self.connect() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM attachments WHERE annotator_id=? ORDER BY created_at",
                    (annotator_id,),
                )
            ]
        result = []
        for row in rows:
            path = Path(row["stored_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Attachment escaped allowlisted directory")
            result.append((row, path.read_bytes()))
        return result

    def install_study(self, manifest: dict[str, Any], assignments: dict[str, list[str]]) -> None:
        study_id = str(manifest["study_id"])
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO studies VALUES (?,?,?)",
                (study_id, json.dumps(manifest, sort_keys=True), utcnow()),
            )
            for annotator_id, item_ids in assignments.items():
                for sequence, item_id in enumerate(item_ids):
                    db.execute(
                        "INSERT OR REPLACE INTO assignments VALUES (?,?,?,?,?)",
                        (
                            annotator_id,
                            study_id,
                            str(manifest.get("mode", "baseline")),
                            item_id,
                            sequence,
                        ),
                    )

    def assigned_items(self, annotator_id: str, study_id: str, mode: str) -> list[str]:
        with self.connect() as db:
            return [
                str(row[0])
                for row in db.execute(
                    "SELECT item_id FROM assignments WHERE annotator_id=? AND study_id=? AND mode=? ORDER BY sequence",
                    (annotator_id, study_id, mode),
                )
            ]
