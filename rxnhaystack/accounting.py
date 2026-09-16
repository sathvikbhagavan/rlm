from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

ACCOUNTING_STATUSES = ("available", "recovered", "unavailable")
AccountingStatus = Literal["available", "recovered", "unavailable"]
RESPONSE_EVENTS_ENV = "RXNHAYSTACK_RESPONSE_EVENTS_PATH"
TRAJECTORY_EVENTS_ENV = "RXNHAYSTACK_TRAJECTORY_EVENTS_PATH"


def append_audit_event(path: str | Path, event: str, **fields: Any) -> None:
    """Append one durable, process-safe JSONL audit event."""

    destination = Path(path).expanduser().resolve()
    payload = {
        "schema_version": 1,
        "event": event,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **fields,
    }
    serialized = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str) + "\n"
    ).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        written = os.write(descriptor, serialized)
        if written != len(serialized):
            raise OSError(f"Short audit-event write: {written}/{len(serialized)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def merged_accounting_status(statuses: list[str]) -> AccountingStatus:
    if any(status == "unavailable" for status in statuses):
        return "unavailable"
    if any(status == "recovered" for status in statuses):
        return "recovered"
    return "available"
