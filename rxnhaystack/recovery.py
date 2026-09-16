from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MISSING_USAGE_ERROR = "No usage data received. Tracking tokens not possible."
WANDB_RUN_PATTERN = re.compile(r"/wandb/(?P<run>run-[^/\s]+)")
GENERATION_PATTERN = re.compile(r"^(?:gen|chatcmpl)-[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class RecoveryAudit:
    run_id: str
    attempt: int
    expected_trajectories: int
    complete_responses: int
    parsed_answers: int
    computed_scores: int
    trace_artifacts: int
    generation_ids: tuple[str, ...]
    accounting_status: str
    recoverability: str
    recovered_cost_usd: float | None


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _wandb_config(stderr: str, wandb_root: Path | None) -> dict[str, Any]:
    if wandb_root is None:
        return {}
    matches = WANDB_RUN_PATTERN.findall(stderr)
    if not matches:
        return {}
    path = wandb_root / matches[-1] / "files" / "config.yaml"
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^num_questions:\s*\n\s+value:\s*(\d+)\s*$", text, re.MULTILINE)
    return {"num_questions": int(match.group(1))} if match else {}


def audit_missing_usage(
    ledger_path: Path,
    *,
    artifact_root: Path,
    wandb_root: Path | None = None,
) -> list[RecoveryAudit]:
    """Read a ledger and artifacts without mutating either."""

    connection = sqlite3.connect(f"file:{ledger_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT run_id, attempts FROM runs WHERE status = 'failed' ORDER BY run_id"
        ).fetchall()
    finally:
        connection.close()

    audits: list[RecoveryAudit] = []
    for row in rows:
        run_id = str(row["run_id"])
        attempt = int(row["attempts"])
        attempt_dir = artifact_root / "runs" / run_id / f"attempt-{attempt:03d}"
        stderr = (attempt_dir / "stderr.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if MISSING_USAGE_ERROR not in stderr:
            continue
        stdout = (attempt_dir / "stdout.log").read_text(
            encoding="utf-8", errors="replace"
        )
        config = _wandb_config(stderr, wandb_root)
        expected = config.get("num_questions")
        # Some legacy task configs counted explicitly skipped query keys.  The
        # startup banner prints one ground-truth record for every trajectory
        # the worker will actually execute, before the first provider request.
        announced = len(re.findall(r"^Ground truth \[", stdout, re.MULTILINE))
        if announced:
            expected = announced
        if not isinstance(expected, int):
            match = re.search(r"^Question\s+1/(\d+)", stdout, re.MULTILINE)
            expected = int(match.group(1)) if match else 0

        trace = _events(attempt_dir / "resource-trace.jsonl")
        completion_events = [
            event for event in trace if event.get("event") == "rlm_completion_metrics"
        ]
        response_events = [
            event
            for event in _events(attempt_dir / "provider-responses.jsonl")
            if event.get("event") == "provider_response_saved"
        ]
        trajectory_events = _events(attempt_dir / "trajectory-events.jsonl")
        final_samples: set[str] = set()
        scored_samples: set[str] = set()
        for event in trajectory_events:
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            for key in data:
                match = re.match(r"^sample/([^/]+)/final_total_tokens$", str(key))
                if match:
                    final_samples.add(match.group(1))
                if re.match(
                    r"^sample/([^/]+)/(?:f1|is_exact_match|accuracy|score)$", str(key)
                ):
                    scored_samples.add(str(key).split("/")[1])

        legacy_scores = len(re.findall(r"^Metrics \[", stdout, re.MULTILINE))
        parsed = max(legacy_scores, len(scored_samples))
        scored = parsed
        complete_responses = max(parsed, min(len(completion_events), expected))
        trace_artifacts = min(max(len(completion_events), len(final_samples)), expected)
        generation_ids = tuple(
            sorted(
                {
                    value
                    for event in response_events
                    for value in [event.get("generation_id")]
                    if isinstance(value, str) and GENERATION_PATTERN.fullmatch(value)
                }
            )
        )
        response_costs = [
            event.get("usage", {}).get("cost_usd")
            for event in response_events
            if isinstance(event.get("usage"), dict)
        ]
        recovered_cost = (
            sum(float(value) for value in response_costs if value is not None)
            if response_costs and all(value is not None for value in response_costs)
            else None
        )
        fully_recoverable = (
            expected > 0
            and min(complete_responses, parsed, scored, trace_artifacts) >= expected
        )
        has_partial = any((complete_responses, parsed, scored, trace_artifacts))
        recoverability = (
            "fully_recoverable"
            if fully_recoverable
            else ("partially_recoverable" if has_partial else "irrecoverable")
        )
        audits.append(
            RecoveryAudit(
                run_id=run_id,
                attempt=attempt,
                expected_trajectories=expected,
                complete_responses=complete_responses,
                parsed_answers=parsed,
                computed_scores=scored,
                trace_artifacts=trace_artifacts,
                generation_ids=generation_ids,
                accounting_status="available" if recovered_cost is not None else "unavailable",
                recoverability=recoverability,
                recovered_cost_usd=recovered_cost,
            )
        )
    return audits


def render_dry_run(audits: list[RecoveryAudit], *, details: bool = False) -> str:
    counts = Counter(audit.recoverability for audit in audits)
    trajectories = Counter()
    for audit in audits:
        trajectories[audit.recoverability] += audit.expected_trajectories
    known_costs = [
        audit.recovered_cost_usd
        for audit in audits
        if audit.recovered_cost_usd is not None
    ]
    payload: dict[str, Any] = {
        "mode": "dry-run",
        "jobs": {
            "total": len(audits),
            "fully_recoverable": counts["fully_recoverable"],
            "partially_recoverable": counts["partially_recoverable"],
            "irrecoverable": counts["irrecoverable"],
        },
        "trajectories": {
            "total": sum(audit.expected_trajectories for audit in audits),
            "fully_recoverable": trajectories["fully_recoverable"],
            "partially_recoverable": trajectories["partially_recoverable"],
            "irrecoverable": trajectories["irrecoverable"],
        },
        "accounting": {
            "jobs_with_recovered_cost": len(known_costs),
            "jobs_with_unknown_cost": len(audits) - len(known_costs),
            "recovered_cost_usd": sum(known_costs) if known_costs else None,
        },
    }
    if details:
        payload["runs"] = [asdict(audit) for audit in audits]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return json.dumps(payload, indent=2, sort_keys=True)
