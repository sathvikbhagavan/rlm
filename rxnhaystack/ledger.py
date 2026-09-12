from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rxnhaystack.manifest import ManifestError, PlannedRun

STATUSES = ("pending", "running", "succeeded", "failed")
REQUIRED_EFFICIENCY_METRICS = (
    "calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_seconds",
    "tool_time_seconds",
    "cost_chf",
)


class LedgerError(RuntimeError):
    """Raised when a run would violate ledger state or identity invariants."""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    spec_hash: str
    status: str
    attempts: int
    estimated_cost_chf: float
    started_at: str | None
    finished_at: str | None
    return_code: int | None
    artifact_dir: str | None
    metrics: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True)
class AttemptRecord:
    run_id: str
    attempt: int
    status: str
    started_at: str
    finished_at: str | None
    return_code: int | None
    artifact_dir: str | None
    metrics: dict[str, Any] | None
    error: str | None


class RunLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    campaign TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    spec_hash TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_chf REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    return_code INTEGER,
                    artifact_dir TEXT,
                    metrics_json TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    return_code INTEGER,
                    artifact_dir TEXT,
                    metrics_json TEXT,
                    error TEXT,
                    PRIMARY KEY (run_id, attempt)
                )
                """
            )

    def sync_runs(self, runs: Iterable[PlannedRun], *, manifest_sha256: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for run in runs:
                existing = connection.execute(
                    "SELECT spec_hash FROM runs WHERE run_id = ?", (run.run_id,)
                ).fetchone()
                if existing is not None and existing["spec_hash"] != run.spec_hash:
                    raise LedgerError(
                        f"Run ID {run.run_id!r} already exists with a different specification. "
                        "Use a new run ID for changed science or execution settings."
                    )
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO runs (
                            run_id, campaign, manifest_sha256, spec_hash, spec_json,
                            status, attempts, estimated_cost_chf, created_at
                        ) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?)
                        """,
                        (
                            run.run_id,
                            run.campaign,
                            manifest_sha256,
                            run.spec_hash,
                            run.spec_json,
                            run.estimated_cost_chf,
                            now,
                        ),
                    )
                else:
                    connection.execute(
                        "UPDATE runs SET manifest_sha256 = ? WHERE run_id = ?",
                        (manifest_sha256, run.run_id),
                    )

    def claim(
        self,
        run_id: str,
        *,
        retry_failed: bool = False,
        recover_running: bool = False,
    ) -> int | None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, attempts FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise LedgerError(f"Unknown run ID: {run_id}")
            allowed = row["status"] == "pending"
            allowed = allowed or (retry_failed and row["status"] == "failed")
            allowed = allowed or (recover_running and row["status"] == "running")
            if not allowed:
                return None
            if row["status"] == "running":
                connection.execute(
                    """
                    UPDATE attempts
                    SET status = 'failed', finished_at = ?, error = ?
                    WHERE run_id = ? AND attempt = ? AND status = 'running'
                    """,
                    (
                        utc_now(),
                        "Marked interrupted during explicit recovery",
                        run_id,
                        row["attempts"],
                    ),
                )
            attempt = int(row["attempts"]) + 1
            started_at = utc_now()
            connection.execute(
                """
                UPDATE runs
                SET status = 'running', attempts = ?, started_at = ?, finished_at = NULL,
                    return_code = NULL, artifact_dir = NULL, metrics_json = NULL, error = NULL
                WHERE run_id = ?
                """,
                (attempt, started_at, run_id),
            )
            connection.execute(
                """
                INSERT INTO attempts (run_id, attempt, status, started_at)
                VALUES (?, ?, 'running', ?)
                """,
                (run_id, attempt, started_at),
            )
            return attempt

    def finish(
        self,
        run_id: str,
        *,
        return_code: int,
        artifact_dir: Path,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        status = "succeeded" if return_code == 0 and error is None else "failed"
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise LedgerError(f"Unknown run ID: {run_id}")
            if row["status"] != "running":
                raise LedgerError(
                    f"Cannot finish run {run_id!r} from status {row['status']!r}; expected 'running'"
                )
            finished_at = utc_now()
            metrics_json = json.dumps(metrics, sort_keys=True) if metrics is not None else None
            connection.execute(
                """
                UPDATE runs
                SET status = ?, finished_at = ?, return_code = ?, artifact_dir = ?,
                    metrics_json = ?, error = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    finished_at,
                    return_code,
                    str(artifact_dir.resolve()),
                    metrics_json,
                    error,
                    run_id,
                ),
            )
            attempt = connection.execute(
                "SELECT attempts FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()["attempts"]
            connection.execute(
                """
                UPDATE attempts
                SET status = ?, finished_at = ?, return_code = ?, artifact_dir = ?,
                    metrics_json = ?, error = ?
                WHERE run_id = ? AND attempt = ?
                """,
                (
                    status,
                    finished_at,
                    return_code,
                    str(artifact_dir.resolve()),
                    metrics_json,
                    error,
                    run_id,
                    attempt,
                ),
            )

    def get(self, run_id: str) -> RunRecord:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise LedgerError(f"Unknown run ID: {run_id}")
        return row_to_record(row)

    def list_runs(self, *, campaign: str | None = None) -> list[RunRecord]:
        query = "SELECT * FROM runs"
        arguments: tuple[str, ...] = ()
        if campaign is not None:
            query += " WHERE campaign = ?"
            arguments = (campaign,)
        query += " ORDER BY run_id"
        with self.connect() as connection:
            rows = connection.execute(query, arguments).fetchall()
        return [row_to_record(row) for row in rows]

    def list_attempts(self, *, campaign: str | None = None) -> list[AttemptRecord]:
        query = """
            SELECT attempts.*
            FROM attempts
            JOIN runs ON runs.run_id = attempts.run_id
        """
        arguments: tuple[str, ...] = ()
        if campaign is not None:
            query += " WHERE runs.campaign = ?"
            arguments = (campaign,)
        query += " ORDER BY attempts.run_id, attempts.attempt"
        with self.connect() as connection:
            rows = connection.execute(query, arguments).fetchall()
        return [attempt_row_to_record(row) for row in rows]


def row_to_record(row: sqlite3.Row) -> RunRecord:
    metrics = json.loads(row["metrics_json"]) if row["metrics_json"] is not None else None
    return RunRecord(
        run_id=row["run_id"],
        spec_hash=row["spec_hash"],
        status=row["status"],
        attempts=row["attempts"],
        estimated_cost_chf=row["estimated_cost_chf"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        return_code=row["return_code"],
        artifact_dir=row["artifact_dir"],
        metrics=metrics,
        error=row["error"],
    )


def attempt_row_to_record(row: sqlite3.Row) -> AttemptRecord:
    metrics = json.loads(row["metrics_json"]) if row["metrics_json"] is not None else None
    return AttemptRecord(
        run_id=row["run_id"],
        attempt=row["attempt"],
        status=row["status"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        return_code=row["return_code"],
        artifact_dir=row["artifact_dir"],
        metrics=metrics,
        error=row["error"],
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_metrics(metrics: Any, *, require_complete: bool = False) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise ManifestError("Run metrics must be a JSON object")
    integer_metrics = ("calls", "input_tokens", "output_tokens", "total_tokens")
    for key in integer_metrics:
        value = metrics.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ManifestError(f"Metric {key!r} must be a non-negative integer")
    non_negative_numbers = (
        "latency_seconds",
        "tool_time_seconds",
        "cost_usd",
        "cost_chf",
    )
    if require_complete:
        missing = [key for key in REQUIRED_EFFICIENCY_METRICS if key not in metrics]
        if missing:
            raise ManifestError(f"Run metrics missing required fields: {', '.join(missing)}")
    for key in non_negative_numbers:
        value = metrics.get(key)
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
        ):
            raise ManifestError(f"Metric {key!r} must be a non-negative number")
    wandb_url = metrics.get("wandb_url")
    if wandb_url is not None and (not isinstance(wandb_url, str) or not wandb_url):
        raise ManifestError("Metric 'wandb_url' must be a non-empty string")
    return metrics
