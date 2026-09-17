from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rxnhaystack.manifest import ExperimentManifest, ManifestError
from rxnhaystack.runtime import atomic_write_json, inspect_git

SNAPSHOT_SCHEMA_VERSION = 1
SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SAFE_METRIC_FIELDS = (
    "calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_seconds",
    "tool_time_seconds",
    "cost_usd",
    "cost_chf",
)
SAFE_RESOURCE_FIELDS = (
    "memory_limit_exceeded",
    "memory_limit_mib",
    "memory_reservation_mib",
    "peak_combined_memory_mib",
    "peak_docker_memory_mib",
    "peak_host_process_tree_rss_mib",
    "peak_process_tree_rss_mib",
    "process_wall_time_seconds",
)
STATUS_ORDER = ("succeeded", "running", "stale", "failed", "pending")


class ControlRoomError(RuntimeError):
    """Raised when a shared-status snapshot is invalid or unsafe to merge."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_snapshot(
    manifest: ExperimentManifest,
    *,
    source_id: str,
    machine: str,
    owner: str,
    scheduler_job_id: str | None = None,
    session_name: str | None = None,
    ledger_path: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Create a path-free, prompt-free status snapshot from one local ledger."""

    validate_source_identity(source_id, machine=machine, owner=owner)
    expected = [safe_run_spec(run) for run in manifest.runs]
    expected_by_id = {item["run_id"]: item for item in expected}
    selected_ledger_path = (
        ledger_path.expanduser().resolve()
        if ledger_path is not None
        else manifest.campaign.artifact_dir / "ledger.sqlite3"
    )
    observations: list[dict[str, Any]] = []
    ledger_manifest_hashes: set[str] = set()

    if selected_ledger_path.is_file():
        runs, attempts = read_ledger(selected_ledger_path, campaign=manifest.campaign.name)
        attempts_by_run: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for attempt in attempts:
            attempts_by_run[str(attempt["run_id"])].append(attempt)
        for row in runs:
            run_id = str(row["run_id"])
            if run_id not in expected_by_id:
                raise ControlRoomError(
                    f"Ledger contains run {run_id!r}, which is absent from the experiment file"
                )
            expected_spec = expected_by_id[run_id]
            if row["spec_hash"] != expected_spec["spec_hash"]:
                raise ControlRoomError(
                    f"Ledger specification for {run_id!r} differs from the experiment file"
                )
            ledger_manifest_hashes.add(str(row["manifest_sha256"]))
            if row["status"] == "pending" and int(row["attempts"]) == 0:
                continue
            observations.append(
                safe_observation(row, attempts_by_run.get(run_id, []), expected_spec)
            )

    git = inspect_git(manifest.campaign.project_root)
    timestamp = generated_at or utc_now()
    payload: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at": timestamp,
        "source": {
            "id": source_id,
            "machine": machine.strip(),
            "owner": owner.strip(),
            "scheduler_job_id": optional_label(scheduler_job_id),
            "session_name": optional_label(session_name),
            "git_commit": git.commit,
            "tracked_dirty": git.tracked_dirty,
        },
        "experiment": {
            "name": manifest.campaign.name,
            "definition_sha256": manifest.sha256,
            "recorded_definition_sha256": sorted(ledger_manifest_hashes),
            "expected_cost_chf": manifest.estimated_cost_chf,
            "expected_runs": expected,
        },
        "observations": sorted(observations, key=lambda item: item["run_id"]),
    }
    payload["snapshot_id"] = snapshot_digest(payload)
    validate_snapshot(payload)
    return payload


def validate_source_identity(source_id: str, *, machine: str, owner: str) -> None:
    if not SOURCE_ID_PATTERN.fullmatch(source_id):
        raise ManifestError(
            "Control-room source IDs must contain 1-64 lowercase letters, digits, '_' or '-', "
            "and must start with a letter or digit"
        )
    for name, value in (("machine", machine), ("owner", owner)):
        if not value.strip() or len(value.strip()) > 80 or any(char in value for char in "\r\n"):
            raise ManifestError(f"Control-room {name} must be a non-empty single-line label")


def optional_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > 120 or any(char in normalized for char in "\r\n"):
        raise ManifestError("Scheduler and session labels must be single-line values")
    return normalized


def safe_run_spec(run: Any) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "spec_hash": run.spec_hash,
        "model": run.model,
        "method": run.method,
        "task": run.task,
        "condition": run.condition,
        "corpus_size": run.corpus_size,
        "repetition": run.repetition,
        "estimated_cost_chf": run.estimated_cost_chf,
        "memory_limit_mib": run.memory_limit_mib,
        "question_parallelism": run.question_parallelism,
    }


def read_ledger(path: Path, *, campaign: str) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    uri = f"file:{path}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        with connection:
            runs = connection.execute(
                "SELECT * FROM runs WHERE campaign = ? ORDER BY run_id", (campaign,)
            ).fetchall()
            attempts = connection.execute(
                """
                SELECT attempts.*
                FROM attempts JOIN runs ON runs.run_id = attempts.run_id
                WHERE runs.campaign = ?
                ORDER BY attempts.run_id, attempts.attempt
                """,
                (campaign,),
            ).fetchall()
    except sqlite3.Error as error:
        raise ControlRoomError(f"Cannot read campaign ledger: {error}") from error
    finally:
        if "connection" in locals():
            connection.close()
    return runs, attempts


def safe_observation(
    row: sqlite3.Row,
    attempt_rows: list[sqlite3.Row],
    expected_spec: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = str(row["run_id"])
    spec_hash = str(expected_spec["spec_hash"])
    attempts = [safe_attempt(item, run_id=run_id, spec_hash=spec_hash) for item in attempt_rows]
    return {
        "run_id": run_id,
        "spec_hash": spec_hash,
        "status": row["status"],
        "attempt_count": int(row["attempts"]),
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "attempts": attempts,
    }


def safe_attempt(row: sqlite3.Row, *, run_id: str, spec_hash: str) -> dict[str, Any]:
    metrics = parse_json_object(row["metrics_json"])
    started_at = str(row["started_at"])
    attempt = int(row["attempt"])
    key_material = f"{run_id}\0{spec_hash}\0{attempt}\0{started_at}".encode()
    return {
        "attempt_key": hashlib.sha256(key_material).hexdigest(),
        "attempt": attempt,
        "status": row["status"],
        "started_at": started_at,
        "finished_at": row["finished_at"],
        "return_code": row["return_code"],
        "failure_category": classify_failure(row["error"]),
        "metrics": safe_metrics(metrics),
    }


def parse_json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ControlRoomError("Ledger contains invalid metrics JSON") from error
    if not isinstance(parsed, dict):
        raise ControlRoomError("Ledger metrics must be a JSON object")
    return parsed


def safe_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in SAFE_METRIC_FIELDS:
        value = metrics.get(key)
        if is_safe_number(value):
            safe[key] = value
    resources = metrics.get("resources")
    if isinstance(resources, Mapping):
        safe_resources = {
            key: resources[key]
            for key in SAFE_RESOURCE_FIELDS
            if key in resources
            and (is_safe_number(resources[key]) or isinstance(resources[key], bool))
        }
        if safe_resources:
            safe["resources"] = safe_resources
    results = metrics.get("results")
    if isinstance(results, Mapping):
        safe_results = {
            str(key): value
            for key, value in results.items()
            if is_safe_result_name(str(key)) and (is_safe_number(value) or isinstance(value, bool))
        }
        if safe_results:
            safe["results"] = safe_results
    wandb_url = metrics.get("wandb_url")
    if isinstance(wandb_url, str) and wandb_url.startswith("https://wandb.ai/"):
        safe["wandb_url"] = wandb_url
    return safe


def is_safe_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def is_safe_result_name(name: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name))


def classify_failure(error: Any) -> str | None:
    if not isinstance(error, str) or not error.strip():
        return None
    normalized = error.casefold()
    categories = (
        ("policy_refusal", ("policy violation", "refusal", "blocked for a previous policy")),
        ("rate_limit", ("rate limit", "http 429", "status code: 429", "status 429")),
        ("context_overflow", ("context window", "maximum context", "too many tokens")),
        ("memory_limit", ("memory limit", "out of memory", "cannot allocate memory", "oom")),
        ("workflow_timeout", ("workflow timeout", "wall-time limit", "wall time limit")),
        ("api_timeout", ("apitimeout", "api timeout", "request timed out", "read timeout")),
        ("http_5xx", ("http 500", "http 502", "http 503", "http 504", "status code: 5")),
        ("interrupted", ("return code 143", "sigterm", "interrupted", "keyboardinterrupt")),
        ("malformed_response", ("empty choices", "invalid choices", "malformed response")),
        ("provider_error", ("api status", "provider error", "http 4")),
    )
    for category, markers in categories:
        if any(marker in normalized for marker in markers):
            return category
    if "timeout" in normalized or "timed out" in normalized:
        return "timeout"
    return "unknown"


def snapshot_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("snapshot_id", None)
    serialized = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def snapshot_state_digest(payload: Mapping[str, Any]) -> str:
    """Hash experiment state while excluding the publication heartbeat time."""

    semantic = dict(payload)
    semantic.pop("snapshot_id", None)
    semantic.pop("generated_at", None)
    serialized = json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    validate_snapshot(snapshot)
    atomic_write_json(path, snapshot)
    path.chmod(0o600)


def load_snapshot(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlRoomError(f"Cannot read status snapshot {path.name}: {error}") from error
    validate_snapshot(payload)
    return payload


def validate_snapshot(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ControlRoomError("Unsupported control-room snapshot schema")
    if payload.get("snapshot_id") != snapshot_digest(payload):
        raise ControlRoomError("Control-room snapshot hash does not match its contents")
    source = payload.get("source")
    experiment = payload.get("experiment")
    observations = payload.get("observations")
    if not isinstance(source, dict) or not SOURCE_ID_PATTERN.fullmatch(str(source.get("id", ""))):
        raise ControlRoomError("Control-room snapshot has an invalid source")
    if not isinstance(experiment, dict) or not isinstance(experiment.get("expected_runs"), list):
        raise ControlRoomError("Control-room snapshot has an invalid experiment definition")
    if not isinstance(observations, list):
        raise ControlRoomError("Control-room snapshot observations must be an array")
    expected_ids = {
        item.get("run_id") for item in experiment["expected_runs"] if isinstance(item, dict)
    }
    if len(expected_ids) != len(experiment["expected_runs"]) or None in expected_ids:
        raise ControlRoomError("Control-room expected run IDs are invalid or duplicated")
    for observation in observations:
        if not isinstance(observation, dict) or observation.get("run_id") not in expected_ids:
            raise ControlRoomError("Control-room observation references an unexpected run")


def artifact_name(snapshot: Mapping[str, Any]) -> str:
    source_id = str(snapshot["source"]["id"])
    campaign = re.sub(r"[^a-z0-9-]+", "-", str(snapshot["experiment"]["name"]).casefold())
    base = f"rxnhaystack-status-{source_id}-{campaign}".strip("-")
    digest = hashlib.sha256(base.encode()).hexdigest()[:10]
    return f"{base[:90].rstrip('-')}-{digest}"


def publish_snapshot(
    path: Path,
    *,
    api_key: str,
    entity: str,
    project: str,
    wandb_module: Any | None = None,
) -> str:
    """Publish one immutable snapshot artifact and move its `latest` alias."""

    snapshot = load_snapshot(path)
    wandb = wandb_module
    if wandb is None:
        import wandb as imported_wandb

        wandb = imported_wandb
    name = artifact_name(snapshot)
    summary = summarize_snapshot(snapshot)
    config = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source_id": snapshot["source"]["id"],
        "machine": snapshot["source"]["machine"],
        "owner": snapshot["source"]["owner"],
        "campaign": snapshot["experiment"]["name"],
        "definition_sha256": snapshot["experiment"]["definition_sha256"],
        "snapshot_id": snapshot["snapshot_id"],
        "artifact_name": name,
    }
    old_key = os.environ.get("WANDB_API_KEY")
    old_disable_code = os.environ.get("WANDB_DISABLE_CODE")
    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_DISABLE_CODE"] = "true"
    try:
        run = wandb.init(
            entity=entity,
            project=project,
            job_type="status-publisher",
            group=str(snapshot["source"]["id"]),
            name=f"{snapshot['source']['id']}-{snapshot['generated_at'].replace(':', '')[:15]}",
            config=config,
            save_code=False,
            reinit="finish_previous",
        )
        if run is None:
            raise ControlRoomError("W&B did not create a status-publisher run")
        try:
            artifact = wandb.Artifact(name, type="rxnhaystack-status", metadata=summary)
            with tempfile.TemporaryDirectory(prefix="rxnhaystack-publish-") as temporary:
                compressed = Path(temporary) / "snapshot.json.gz"
                with (
                    path.open("rb") as source,
                    gzip.open(compressed, "wb", compresslevel=9) as target,
                ):
                    shutil_copyfileobj(source, target)
                artifact.add_file(str(compressed), name="snapshot.json.gz")
                logged = run.log_artifact(artifact, aliases=["latest"])
                if hasattr(logged, "wait"):
                    logged.wait()
            run.summary.update(summary)
            url = str(getattr(run, "url", ""))
        finally:
            run.finish()
    finally:
        restore_environment("WANDB_API_KEY", old_key)
        restore_environment("WANDB_DISABLE_CODE", old_disable_code)
    return url


def restore_environment(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def summarize_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(item["status"] for item in snapshot["observations"])
    return {
        "generated_at": snapshot["generated_at"],
        "expected_runs": len(snapshot["experiment"]["expected_runs"]),
        "observed_runs": len(snapshot["observations"]),
        **{f"local_{status}": counts[status] for status in ("running", "succeeded", "failed")},
    }


def sync_snapshots(
    *,
    api_key: str,
    entity: str,
    project: str,
    output_dir: Path,
    wandb_module: Any | None = None,
) -> list[Path]:
    """Download the latest artifact for every independently publishing source."""

    wandb = wandb_module
    if wandb is None:
        import wandb as imported_wandb

        wandb = imported_wandb
    old_key = os.environ.get("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = api_key
    try:
        return sync_snapshots_with_api(
            wandb=wandb,
            api_key=api_key,
            entity=entity,
            project=project,
            output_dir=output_dir,
        )
    finally:
        restore_environment("WANDB_API_KEY", old_key)


def sync_snapshots_with_api(
    *,
    wandb: Any,
    api_key: str,
    entity: str,
    project: str,
    output_dir: Path,
) -> list[Path]:
    api = wandb.Api(api_key=api_key)
    runs = api.runs(
        f"{entity}/{project}", filters={"jobType": "status-publisher"}, order="-created_at"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written: list[Path] = []
    for run in runs:
        config = dict(run.config)
        source_id = str(config.get("source_id", ""))
        name = str(config.get("artifact_name", ""))
        expected_snapshot_id = str(config.get("snapshot_id", ""))
        if source_id in seen or not SOURCE_ID_PATTERN.fullmatch(source_id) or not name:
            continue
        artifact = api.artifact(f"{entity}/{project}/{name}:latest")
        with tempfile.TemporaryDirectory(prefix="rxnhaystack-status-") as temporary:
            downloaded_root = Path(artifact.download(root=temporary))
            compressed = downloaded_root / "snapshot.json.gz"
            plain = downloaded_root / "snapshot.json"
            if compressed.is_file():
                try:
                    with gzip.open(compressed, "rt", encoding="utf-8") as handle:
                        snapshot = json.load(handle)
                except (OSError, json.JSONDecodeError) as error:
                    raise ControlRoomError(
                        f"Cannot read compressed status artifact for {source_id}: {error}"
                    ) from error
                validate_snapshot(snapshot)
            elif plain.is_file():
                snapshot = load_snapshot(plain)
            else:
                raise ControlRoomError(f"Status artifact for {source_id} has no snapshot file")
            if snapshot["source"]["id"] != source_id or artifact_name(snapshot) != name:
                raise ControlRoomError(
                    f"Downloaded status artifact for {source_id} is inconsistent"
                )
            if snapshot["snapshot_id"] != expected_snapshot_id:
                raise ControlRoomError(
                    f"Latest status artifact for {source_id} does not match its publisher run"
                )
            destination = output_dir / f"{source_id}.json"
            write_snapshot(destination, snapshot)
        seen.add(source_id)
        written.append(destination)
    return written


def shutil_copyfileobj(source: Any, target: Any, *, chunk_size: int = 1024 * 1024) -> None:
    while chunk := source.read(chunk_size):
        target.write(chunk)


def load_snapshot_directory(path: Path) -> list[dict[str, Any]]:
    if not path.is_dir():
        raise ControlRoomError(f"Status snapshot directory does not exist: {path}")
    latest: dict[str, dict[str, Any]] = {}
    for snapshot_path in sorted(path.glob("*.json")):
        snapshot = load_snapshot(snapshot_path)
        source_id = str(snapshot["source"]["id"])
        if source_id not in latest or snapshot["generated_at"] > latest[source_id]["generated_at"]:
            latest[source_id] = snapshot
    if not latest:
        raise ControlRoomError(f"No status snapshots found in {path}")
    return list(latest.values())


def merge_snapshots(
    snapshots: Iterable[dict[str, Any]],
    *,
    stale_after_seconds: float = 7200,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Merge sources by immutable run and attempt identity without double-counting."""

    reference_time = now or datetime.now(UTC)
    by_experiment: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        validate_snapshot(snapshot)
        experiment = snapshot["experiment"]
        key = (str(experiment["name"]), str(experiment["definition_sha256"]))
        by_experiment[key].append(snapshot)
    campaigns = [
        merge_experiment(group, stale_after_seconds=stale_after_seconds, now=reference_time)
        for _, group in sorted(by_experiment.items())
    ]
    return {"generated_at": reference_time.isoformat(), "campaigns": campaigns}


def merge_experiment(
    snapshots: list[dict[str, Any]], *, stale_after_seconds: float, now: datetime
) -> dict[str, Any]:
    first = snapshots[0]
    expected: dict[str, dict[str, Any]] = {}
    observations: dict[str, list[tuple[dict[str, Any], dict[str, Any], bool]]] = defaultdict(list)
    sources: list[dict[str, Any]] = []
    for snapshot in snapshots:
        source = snapshot["source"]
        age_seconds = max(0.0, (now - parse_timestamp(snapshot["generated_at"])).total_seconds())
        source_stale = age_seconds > stale_after_seconds
        sources.append(
            {
                **source,
                "generated_at": snapshot["generated_at"],
                "age_seconds": age_seconds,
                "stale": source_stale,
                "observed_runs": len(snapshot["observations"]),
            }
        )
        for spec in snapshot["experiment"]["expected_runs"]:
            previous = expected.setdefault(spec["run_id"], spec)
            if previous != spec:
                raise ControlRoomError(
                    f"Sources disagree on the specification of run {spec['run_id']!r}"
                )
        for observation in snapshot["observations"]:
            observations[observation["run_id"]].append((observation, source, source_stale))

    merged_runs = [
        merge_run(spec, observations.get(run_id, [])) for run_id, spec in expected.items()
    ]
    merged_runs.sort(key=lambda item: item["run_id"])
    counts = Counter(run["status"] for run in merged_runs)
    cells = aggregate_cells(merged_runs)
    metrics = aggregate_attempt_metrics(merged_runs)
    return {
        "name": first["experiment"]["name"],
        "definition_sha256": first["experiment"]["definition_sha256"],
        "expected_cost_chf": first["experiment"]["expected_cost_chf"],
        "counts": {status: counts[status] for status in STATUS_ORDER},
        "duplicate_runs": sum(bool(run["duplicate_execution"]) for run in merged_runs),
        "sources": sorted(sources, key=lambda item: item["id"]),
        "cells": cells,
        "metrics": metrics,
        "runs": merged_runs,
    }


def merge_run(
    spec: dict[str, Any],
    candidates: list[tuple[dict[str, Any], dict[str, Any], bool]],
) -> dict[str, Any]:
    all_attempts: dict[str, dict[str, Any]] = {}
    attempt_sources: dict[str, set[str]] = defaultdict(set)
    source_attempts: dict[str, set[str]] = defaultdict(set)
    current: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
    for observation, source, source_stale in candidates:
        current.append((observation, source, source_stale))
        for attempt in observation["attempts"]:
            key = attempt["attempt_key"]
            existing = all_attempts.setdefault(key, attempt)
            if existing != attempt:
                raise ControlRoomError(f"Sources disagree on attempt {key}")
            source_id = str(source["id"])
            attempt_sources[key].add(source_id)
            source_attempts[source_id].add(key)

    statuses = {observation["status"] for observation, _, _ in current}
    fresh_running = [item for item in current if item[0]["status"] == "running" and not item[2]]
    stale_running = [item for item in current if item[0]["status"] == "running" and item[2]]
    if "succeeded" in statuses:
        status = "succeeded"
    elif fresh_running:
        status = "running"
    elif stale_running:
        status = "stale"
    elif "failed" in statuses:
        status = "failed"
    else:
        status = "pending"

    execution_sources = {
        source_id for source_ids in attempt_sources.values() for source_id in source_ids
    }
    attempt_sets = list(source_attempts.values())
    duplicate_execution = any(
        not first.issubset(second) and not second.issubset(first)
        for index, first in enumerate(attempt_sets)
        for second in attempt_sets[index + 1 :]
    )
    distinct_attempts = list(all_attempts.values())
    failure_categories = Counter(
        attempt["failure_category"]
        for attempt in distinct_attempts
        if attempt.get("failure_category") is not None
    )
    active_sources = sorted(
        source["id"] for observation, source, _ in current if observation["status"] == "running"
    )
    return {
        **{key: value for key, value in spec.items() if key != "spec_hash"},
        "status": status,
        "sources": sorted(execution_sources),
        "active_sources": active_sources,
        "attempts": sorted(distinct_attempts, key=lambda item: item["started_at"]),
        "attempt_count": len(distinct_attempts),
        "duplicate_execution": duplicate_execution,
        "failure_categories": dict(sorted(failure_categories.items())),
    }


def aggregate_cells(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(friendly_model(run["model"]), run["method"])].append(run)
    cells: list[dict[str, Any]] = []
    for (model, method), items in sorted(grouped.items()):
        counts = Counter(item["status"] for item in items)
        durations = [
            attempt_duration(attempt)
            for item in items
            for attempt in item["attempts"]
            if attempt["status"] in {"succeeded", "failed"}
            and attempt_duration(attempt) is not None
        ]
        cells.append(
            {
                "model": model,
                "method": method,
                "expected": len(items),
                "counts": {status: counts[status] for status in STATUS_ORDER},
                "median_attempt_seconds": statistics.median(durations) if durations else None,
            }
        )
    return cells


def attempt_duration(attempt: Mapping[str, Any]) -> float | None:
    if not attempt.get("finished_at"):
        return None
    return max(
        0.0,
        (
            parse_timestamp(str(attempt["finished_at"]))
            - parse_timestamp(str(attempt["started_at"]))
        ).total_seconds(),
    )


def aggregate_attempt_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = {attempt["attempt_key"]: attempt for run in runs for attempt in run["attempts"]}
    totals = {key: 0.0 for key in SAFE_METRIC_FIELDS}
    present = Counter()
    peak_memory_mib = 0.0
    memory_limit_exceeded = 0
    unknown_cost_attempts = 0
    failures = Counter()
    for attempt in attempts.values():
        metrics = attempt["metrics"]
        for key in SAFE_METRIC_FIELDS:
            if is_safe_number(metrics.get(key)):
                totals[key] += float(metrics[key])
                present[key] += 1
        resources = metrics.get("resources", {})
        if isinstance(resources, Mapping):
            for key in ("peak_combined_memory_mib", "peak_process_tree_rss_mib"):
                if is_safe_number(resources.get(key)):
                    peak_memory_mib = max(peak_memory_mib, float(resources[key]))
            if resources.get("memory_limit_exceeded") is True:
                memory_limit_exceeded += 1
        if "cost_chf" not in metrics:
            unknown_cost_attempts += 1
        if attempt.get("failure_category"):
            failures[attempt["failure_category"]] += 1
    rendered_totals: dict[str, int | float] = {}
    for key, value in totals.items():
        rendered_totals[key] = (
            int(value)
            if key in {"calls", "input_tokens", "output_tokens", "total_tokens"}
            else value
        )
    return {
        **rendered_totals,
        "attempts": len(attempts),
        "unknown_cost_attempts": unknown_cost_attempts,
        "peak_memory_mib": peak_memory_mib,
        "memory_limit_exceeded_attempts": memory_limit_exceeded,
        "failure_categories": dict(sorted(failures.items())),
        "metric_coverage": dict(present),
    }


def parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ControlRoomError(f"Invalid timestamp in status snapshot: {value!r}") from error
    if parsed.tzinfo is None:
        raise ControlRoomError(f"Status timestamps must include a timezone: {value!r}")
    return parsed.astimezone(UTC)


def friendly_model(model: str) -> str:
    normalized = model.casefold()
    names = (
        ("deepseek", "DeepSeek V4 Flash"),
        ("glm-5.2", "GLM 5.2"),
        ("qwen3.5", "Qwen 3.5"),
        ("gemini", "Gemini Flash"),
        ("claude", "Claude Haiku"),
        ("gpt-5-mini", "GPT-5 mini"),
    )
    for marker, name in names:
        if marker in normalized:
            return name
    return model.rsplit("/", maxsplit=1)[-1]


def write_markdown(path: Path, merged: Mapping[str, Any]) -> None:
    lines = [
        "# RxnHaystack live experiment status",
        "",
        f"Generated: `{merged['generated_at']}`",
        "",
        "> Generated from sanitized machine snapshots. Local SQLite ledgers remain authoritative.",
        "",
    ]
    for campaign in merged["campaigns"]:
        counts = campaign["counts"]
        lines.extend(
            [
                f"## {campaign['name']}",
                "",
                f"Definition: `{campaign['definition_sha256']}`",
                "",
                f"Expected: **{sum(counts.values()):,}** jobs; succeeded: **{counts['succeeded']:,}**; "
                f"running: **{counts['running']:,}**; stale: **{counts['stale']:,}**; "
                f"failed: **{counts['failed']:,}**; pending: **{counts['pending']:,}**.",
                "",
                "| Model | Method | Expected | Succeeded | Running | Stale | Failed | Pending |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for cell in campaign["cells"]:
            item = cell["counts"]
            lines.append(
                f"| {cell['model']} | {cell['method']} | {cell['expected']} | "
                f"{item['succeeded']} | {item['running']} | {item['stale']} | "
                f"{item['failed']} | {item['pending']} |"
            )
        lines.extend(
            [
                "",
                "### Reporting sources",
                "",
                "| Source | Owner | Machine | Updated | Observed | State |",
                "| --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for source in campaign["sources"]:
            state = "STALE" if source["stale"] else "current"
            lines.append(
                f"| {source['id']} | {source['owner']} | {source['machine']} | "
                f"{source['generated_at']} | {source['observed_runs']} | {state} |"
            )
        if campaign["duplicate_runs"]:
            lines.extend(
                [
                    "",
                    f"Warning: **{campaign['duplicate_runs']}** run IDs have attempts from more than one source.",
                ]
            )
        lines.append("")
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def write_dashboard(path: Path, merged: Mapping[str, Any]) -> None:
    serialized = json.dumps(merged, sort_keys=True, separators=(",", ":")).replace("<", "\\u003c")
    document = DASHBOARD_TEMPLATE.replace("__CONTROL_ROOM_DATA__", serialized)
    atomic_write_text(path, document)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def dashboard_title(merged: Mapping[str, Any]) -> str:
    campaigns = merged.get("campaigns", [])
    return "RxnHaystack control room" if len(campaigns) != 1 else str(campaigns[0]["name"])


DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>RxnHaystack control room</title>
<style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--green:#3fb950;--blue:#58a6ff;--red:#f85149;--yellow:#d29922;--purple:#bc8cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}header{padding:22px 28px;border-bottom:1px solid var(--line);position:sticky;top:0;background:rgba(13,17,23,.96);z-index:3}h1{font-size:21px;margin:0 0 5px}.sub{color:var(--muted)}main{padding:22px 28px;max-width:1700px;margin:auto}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px}.card{padding:14px}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}.value{font-size:23px;font-weight:700;margin-top:4px}.panel{margin:14px 0;padding:16px;overflow:auto}h2{font-size:16px;margin:0 0 12px}.filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}input,select{background:var(--bg);border:1px solid var(--line);border-radius:6px;color:var(--text);padding:7px 9px}input{min-width:280px;flex:1}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{text-align:left;padding:8px 9px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em;position:sticky;top:0;background:var(--panel)}.ok{color:var(--green)}.running{color:var(--blue)}.failed{color:var(--red)}.stale{color:var(--yellow)}.pending{color:var(--muted)}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 7px;font-size:12px}.bar{height:7px;background:#21262d;border-radius:9px;overflow:hidden;min-width:130px}.bar>span{height:100%;display:block;background:var(--green)}.warn{color:var(--yellow)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}.hidden{display:none}a{color:var(--blue)}
</style>
</head>
<body>
<header><h1>RxnHaystack experiment control room</h1><div class="sub" id="updated"></div></header>
<main id="app"></main>
<script>const DATA=__CONTROL_ROOM_DATA__;
const fmt=n=>new Intl.NumberFormat().format(n||0); const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const duration=s=>s==null?'—':s<120?Math.round(s)+'s':s<7200?Math.round(s/60)+'m':(s/3600).toFixed(1)+'h';
const statusClass=s=>s==='succeeded'?'ok':s; document.getElementById('updated').textContent='Merged '+new Date(DATA.generated_at).toLocaleString()+' · read-only view';
function renderCampaign(c,idx){const total=Object.values(c.counts).reduce((a,b)=>a+b,0),done=c.counts.succeeded,metrics=c.metrics;
return `<section data-campaign="${idx}"><h2>${esc(c.name)} <span class="pill mono">${esc(c.definition_sha256.slice(0,12))}</span></h2>
<div class="cards"><div class="card"><div class="label">Completed</div><div class="value ok">${fmt(done)} / ${fmt(total)}</div></div><div class="card"><div class="label">Running</div><div class="value running">${fmt(c.counts.running)}</div></div><div class="card"><div class="label">Failed</div><div class="value failed">${fmt(c.counts.failed)}</div></div><div class="card"><div class="label">Pending</div><div class="value pending">${fmt(c.counts.pending)}</div></div><div class="card"><div class="label">Recorded cost</div><div class="value">CHF ${(metrics.cost_chf||0).toFixed(2)}</div><div class="sub">${fmt(metrics.unknown_cost_attempts)} attempts unknown</div></div><div class="card"><div class="label">Model calls</div><div class="value">${fmt(metrics.calls)}</div></div><div class="card"><div class="label">Tokens</div><div class="value">${fmt(metrics.total_tokens)}</div></div><div class="card"><div class="label">Peak memory</div><div class="value">${(metrics.peak_memory_mib/1024).toFixed(1)} GiB</div></div></div>
<div class="panel"><h2>Model and method matrix</h2><table><thead><tr><th>Model</th><th>Method</th><th>Progress</th><th>Success</th><th>Running</th><th>Stale</th><th>Failed</th><th>Pending</th><th>Median attempt</th></tr></thead><tbody>${c.cells.map(x=>{const pct=100*x.counts.succeeded/x.expected;return `<tr><td>${esc(x.model)}</td><td>${esc(x.method)}</td><td><div class="bar"><span style="width:${pct}%"></span></div></td><td class="ok">${x.counts.succeeded}/${x.expected}</td><td class="running">${x.counts.running}</td><td class="stale">${x.counts.stale}</td><td class="failed">${x.counts.failed}</td><td class="pending">${x.counts.pending}</td><td>${duration(x.median_attempt_seconds)}</td></tr>`}).join('')}</tbody></table></div>
<div class="panel"><h2>Reporting sources ${c.duplicate_runs?`<span class="warn">· ${c.duplicate_runs} duplicate run IDs</span>`:''}</h2><table><thead><tr><th>Source</th><th>Owner</th><th>Machine</th><th>Last update</th><th>Observed</th><th>Git</th><th>Scheduler / session</th><th>State</th></tr></thead><tbody>${c.sources.map(s=>`<tr><td class="mono">${esc(s.id)}</td><td>${esc(s.owner)}</td><td>${esc(s.machine)}</td><td>${esc(new Date(s.generated_at).toLocaleString())}</td><td>${fmt(s.observed_runs)}</td><td class="mono">${esc(s.git_commit.slice(0,12))}${s.tracked_dirty?' *':''}</td><td>${esc(s.scheduler_job_id||s.session_name||'—')}</td><td class="${s.stale?'stale':'ok'}">${s.stale?'STALE':'current'}</td></tr>`).join('')}</tbody></table></div>
<div class="panel"><h2>Run explorer</h2><div class="filters"><input class="search" placeholder="Filter run ID, model, task, source…"><select class="method"><option value="">all methods</option>${[...new Set(c.runs.map(r=>r.method))].map(v=>`<option>${esc(v)}</option>`).join('')}</select><select class="status"><option value="">all states</option>${['succeeded','running','stale','failed','pending'].map(v=>`<option>${v}</option>`).join('')}</select></div><table><thead><tr><th>Run</th><th>Model</th><th>Task</th><th>Method</th><th>Condition</th><th>Status</th><th>Attempts</th><th>Source</th><th>Failure class</th></tr></thead><tbody class="runs">${c.runs.map(r=>`<tr data-search="${esc((r.run_id+' '+r.model+' '+r.task+' '+r.sources.join(' ')).toLowerCase())}" data-method="${esc(r.method)}" data-status="${esc(r.status)}"><td class="mono">${esc(r.run_id)}</td><td>${esc(r.model)}</td><td>${esc(r.task)}</td><td>${esc(r.method)}</td><td>${esc(r.condition)}</td><td class="${statusClass(r.status)}">${esc(r.status)}</td><td>${r.attempt_count}</td><td>${esc(r.sources.join(', ')||'—')}</td><td>${esc(Object.keys(r.failure_categories).join(', ')||'—')}</td></tr>`).join('')}</tbody></table></div></section>`}
document.getElementById('app').innerHTML=DATA.campaigns.map(renderCampaign).join('');
document.querySelectorAll('section').forEach(section=>{const q=section.querySelector('.search'),m=section.querySelector('.method'),s=section.querySelector('.status'),rows=[...section.querySelectorAll('.runs tr')];const apply=()=>{const needle=q.value.toLowerCase();rows.forEach(r=>r.classList.toggle('hidden',!!((needle&&!r.dataset.search.includes(needle))||(m.value&&r.dataset.method!==m.value)||(s.value&&r.dataset.status!==s.value))))};q.oninput=apply;m.onchange=apply;s.onchange=apply});
</script></body></html>"""
