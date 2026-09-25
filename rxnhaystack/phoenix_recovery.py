"""Read-only recovery of benchmark predictions from a Phoenix SQLite store."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rxnhaystack.score_recovery import parse_indices

PROJECT_NAMES: dict[tuple[str, str], str] = {
    ("tier3/task18", "llm"): "LLM-Task18_tier3",
    ("tier3/task18", "rlm"): "RLMs-Task18_tier3",
    ("tier3/task23", "rlm"): "RLMs-Task23_tier3",
}

COMMA_SEPARATED_INDICES_RE = re.compile(r"(?<![\d.])-?\d+(?:\s*,\s*-?\d+)+")
LABELED_ANSWER_RE = re.compile(
    r"^(?:Answer|All indices|Final answer(?: string)?):\s*([^\r\n]+)",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class PhoenixRunWindow:
    run_id: str
    task: str
    method: str
    model: str
    started_at: str
    finished_at: str


@dataclass(frozen=True)
class PhoenixPrediction:
    run_id: str
    question_id: str
    indices: tuple[int, ...]
    session_id: str
    project: str
    span_id: str
    extraction_method: str
    answer_sha256: str


def iso_to_sqlite_timestamp(value: str) -> str:
    """Convert a UTC ISO timestamp to Phoenix's naive UTC representation."""

    return value.replace("T", " ").replace("+00:00", "").replace("Z", "")


def json_attribute(value: str | dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ValueError("Phoenix span attributes must be a JSON object")
    return payload


def chat_content(attributes: dict[str, Any]) -> str:
    output = attributes.get("output") or {}
    serialized = output.get("value") if isinstance(output, dict) else None
    if not isinstance(serialized, str):
        return ""
    payload = json.loads(serialized)
    choices = payload.get("choices") or ()
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def input_message_content(message: dict[str, Any]) -> str:
    payload = message.get("message", message)
    content = payload.get("content", "") if isinstance(payload, dict) else ""
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True)


def prediction_candidates(
    *, method: str, root_chat_attributes: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    """Return ordered answer candidates, with the most recent candidate last."""

    if not root_chat_attributes:
        return []
    if method == "llm":
        return [("phoenix-direct-output", chat_content(root_chat_attributes[-1]))]
    if method != "rlm":
        return []

    llm = root_chat_attributes[-1].get("llm") or {}
    messages = llm.get("input_messages") or () if isinstance(llm, dict) else ()
    output: list[tuple[str, str]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = input_message_content(message)
        output.extend(
            (f"phoenix-rlm-message-{message_index}-comma-run", match.group())
            for match in COMMA_SEPARATED_INDICES_RE.finditer(content)
        )
        output.extend(
            (f"phoenix-rlm-message-{message_index}-answer-line", match.group(1))
            for match in LABELED_ANSWER_RE.finditer(content)
        )
    return output


def select_prediction_candidate(
    *, method: str, root_chat_attributes: list[dict[str, Any]], expected_count: int
) -> tuple[tuple[int, ...], str, str] | None:
    """Select the latest candidate agreeing with the evaluator's logged count."""

    for extraction_method, raw_answer in reversed(
        prediction_candidates(method=method, root_chat_attributes=root_chat_attributes)
    ):
        indices = parse_indices(raw_answer)
        if len(set(indices)) != expected_count:
            continue
        return (
            indices,
            extraction_method,
            hashlib.sha256(raw_answer.encode()).hexdigest(),
        )
    return None


def session_models(connection: sqlite3.Connection, session_id: str) -> frozenset[str]:
    models: set[str] = set()
    rows = connection.execute(
        """
        SELECT spans.attributes
        FROM spans
        JOIN traces ON traces.id = spans.trace_rowid
        JOIN project_sessions ON project_sessions.id = traces.project_session_rowid
        WHERE project_sessions.session_id = ? AND spans.span_kind = 'LLM'
        """,
        (session_id,),
    )
    for row in rows:
        attributes = json_attribute(row[0])
        llm = attributes.get("llm") or {}
        rlm = attributes.get("rlm") or {}
        model = None
        if isinstance(llm, dict):
            model = llm.get("model_name")
        if not model and isinstance(rlm, dict):
            model = rlm.get("final_model")
        if isinstance(model, str) and model:
            models.add(model)
    return frozenset(models)


def matching_session(
    connection: sqlite3.Connection, *, window: PhoenixRunWindow
) -> tuple[str, str] | None:
    """Match a run only when time, task project, and model identify one session."""

    project = PROJECT_NAMES.get((window.task, window.method))
    if project is None:
        return None
    rows = connection.execute(
        """
        SELECT project_sessions.session_id
        FROM project_sessions
        JOIN projects ON projects.id = project_sessions.project_id
        WHERE projects.name = ?
          AND project_sessions.start_time <= ?
          AND project_sessions.end_time >= ?
        ORDER BY project_sessions.start_time, project_sessions.id
        """,
        (
            project,
            iso_to_sqlite_timestamp(window.finished_at),
            iso_to_sqlite_timestamp(window.started_at),
        ),
    ).fetchall()
    matching = [
        str(row[0]) for row in rows if window.model in session_models(connection, str(row[0]))
    ]
    if len(matching) != 1:
        return None
    return matching[0], project


def root_chat_spans(
    connection: sqlite3.Connection, *, session_id: str
) -> list[tuple[str, dict[str, Any]]]:
    output: list[tuple[str, dict[str, Any]]] = []
    rows = connection.execute(
        """
        SELECT spans.span_id, spans.attributes
        FROM spans
        JOIN traces ON traces.id = spans.trace_rowid
        JOIN project_sessions ON project_sessions.id = traces.project_session_rowid
        WHERE project_sessions.session_id = ? AND spans.name = 'ChatCompletion'
        ORDER BY spans.start_time, spans.id
        """,
        (session_id,),
    )
    for span_id, serialized_attributes in rows:
        attributes = json_attribute(serialized_attributes)
        metadata = attributes.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("depth", 0) == 0:
            output.append((str(span_id), attributes))
    return output


def open_phoenix_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    result = connection.execute("PRAGMA quick_check").fetchone()
    if result is None or result[0] != "ok":
        connection.close()
        raise ValueError(f"Phoenix database failed quick_check: {path}")
    return connection
