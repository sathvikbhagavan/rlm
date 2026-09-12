from __future__ import annotations

import fnmatch
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rxnhaystack.dataset import CLEANED_DATASET_ENV, RAW_DATASET_ENV
from rxnhaystack.ledger import RunLedger, validate_metrics
from rxnhaystack.manifest import ExperimentManifest, ManifestError, PlannedRun
from rxnhaystack.metrics import METRICS_PATH_ENV, USD_TO_CHF_ENV
from rxnhaystack.resources import MemoryBudget, wait_with_memory_watchdog
from rxnhaystack.runtime import (
    RESOURCE_TRACE_PATH_ENV,
    Preflight,
    atomic_write_json,
    preflight_as_dict,
)


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    status: str
    attempt: int | None
    return_code: int | None
    artifact_dir: Path | None
    error: str | None


def select_runs(runs: tuple[PlannedRun, ...], patterns: list[str]) -> list[PlannedRun]:
    if not patterns:
        return list(runs)
    selected = [run for run in runs if any(fnmatch.fnmatchcase(run.run_id, p) for p in patterns)]
    if not selected:
        raise ManifestError(f"No run IDs matched selection patterns: {', '.join(patterns)}")
    return selected


def enforce_remaining_budget(manifest: ExperimentManifest, ledger: RunLedger) -> float:
    estimates = {run.run_id: run.estimated_cost_chf for run in manifest.runs}
    records = {
        record.run_id: record for record in ledger.list_runs(campaign=manifest.campaign.name)
    }
    committed = 0.0
    for attempt in ledger.list_attempts(campaign=manifest.campaign.name):
        if attempt.metrics is not None and isinstance(
            attempt.metrics.get("cost_chf"), (int, float)
        ):
            committed += float(attempt.metrics["cost_chf"])
        else:
            committed += estimates[attempt.run_id]
    for run in manifest.runs:
        record = records[run.run_id]
        if record.status in {"pending", "failed"}:
            committed += run.estimated_cost_chf
    if committed > manifest.campaign.budget_chf:
        raise ManifestError(
            f"Recorded and remaining estimated cost CHF {committed:.2f} exceeds campaign budget "
            f"CHF {manifest.campaign.budget_chf:.2f}. Increase the explicit budget or reduce the matrix."
        )
    return committed


def execute_run(
    run: PlannedRun,
    *,
    manifest: ExperimentManifest,
    preflight: Preflight,
    ledger: RunLedger,
    secrets: dict[str, str],
    retry_failed: bool,
    recover_running: bool,
) -> ExecutionResult:
    attempt = ledger.claim(
        run.run_id,
        retry_failed=retry_failed,
        recover_running=recover_running,
    )
    if attempt is None:
        return ExecutionResult(
            run_id=run.run_id,
            status="skipped",
            attempt=None,
            return_code=None,
            artifact_dir=None,
            error=None,
        )

    attempt_dir = manifest.campaign.artifact_dir / "runs" / run.run_id / f"attempt-{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = attempt_dir / "metrics.json"
    metadata_path = attempt_dir / "metadata.json"
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    resource_trace_path = attempt_dir / "resource-trace.jsonl"
    generated_env = build_run_environment(
        run,
        manifest=manifest,
        preflight=preflight,
        attempt_dir=attempt_dir,
        metrics_path=metrics_path,
        resource_trace_path=resource_trace_path,
    )
    environment = os.environ.copy()
    environment.update(run.env)
    environment.update(generated_env)
    environment.update(secrets)
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    execution_metadata: dict[str, Any] = {
        "attempt": attempt,
        "command": list(run.command),
        "cwd": str(manifest.campaign.project_root),
        "environment": {**run.env, **generated_env},
        "secret_names": sorted(secrets),
        "started_at": started_at,
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "manifest": {"path": str(manifest.path), "sha256": manifest.sha256},
        "run": json.loads(run.spec_json),
        "provenance": preflight_as_dict(preflight),
        "execution": execution_metadata,
        "result": {"status": "running"},
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    atomic_write_json(metadata_path, metadata)

    return_code = 127
    error: str | None = None
    metrics: dict[str, Any] | None = None
    peak_rss_mib = 0.0
    memory_limit_exceeded = False
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                run.command,
                cwd=manifest.campaign.project_root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            usage = wait_with_memory_watchdog(
                process,
                memory_limit_mib=run.memory_limit_mib,
                trace_path=resource_trace_path,
                run_id=run.run_id,
            )
        return_code = usage.return_code
        peak_rss_mib = usage.peak_rss_mib
        memory_limit_exceeded = usage.memory_limit_exceeded
        if metrics_path.exists():
            try:
                metrics = validate_metrics(
                    json.loads(metrics_path.read_text(encoding="utf-8")),
                    require_complete=manifest.campaign.require_metrics,
                )
            except (OSError, json.JSONDecodeError, ManifestError) as metrics_error:
                error = f"Invalid metrics: {metrics_error}"
        elif manifest.campaign.require_metrics:
            error = f"Command did not write required metrics to {metrics_path}"
        if memory_limit_exceeded:
            error = f"Process-tree RSS exceeded the {run.memory_limit_mib} MiB memory limit" + (
                f"; {error}" if error is not None else ""
            )
        elif return_code != 0:
            error = f"Command exited with status {return_code}" + (
                f"; {error}" if error is not None else ""
            )
    except OSError as execution_error:
        error = f"Could not execute command: {execution_error}"
    finished_at = datetime.now(UTC).isoformat()
    duration = time.monotonic() - started
    status = "succeeded" if return_code == 0 and error is None else "failed"
    resource_usage = {
        "process_wall_time_seconds": duration,
        "peak_process_tree_rss_mib": peak_rss_mib,
        "memory_reservation_mib": run.memory_reservation_mib,
        "memory_limit_mib": run.memory_limit_mib,
        "memory_limit_exceeded": memory_limit_exceeded,
        "trace_path": str(resource_trace_path),
    }
    execution_metadata.update(
        {
            "finished_at": finished_at,
            "duration_seconds": duration,
            "resources": resource_usage,
        }
    )
    if metrics is not None:
        metrics["resources"] = resource_usage
        atomic_write_json(metrics_path, metrics)
    metadata["result"] = {
        "status": status,
        "return_code": return_code,
        "error": error,
        "metrics": metrics,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    atomic_write_json(metadata_path, metadata)
    ledger.finish(
        run.run_id,
        return_code=return_code,
        artifact_dir=attempt_dir,
        metrics=metrics,
        error=error,
    )
    return ExecutionResult(
        run_id=run.run_id,
        status=status,
        attempt=attempt,
        return_code=return_code,
        artifact_dir=attempt_dir,
        error=error,
    )


def execute_run_safely(
    run: PlannedRun,
    *,
    manifest: ExperimentManifest,
    preflight: Preflight,
    ledger: RunLedger,
    secrets: dict[str, str],
    retry_failed: bool,
    recover_running: bool,
) -> ExecutionResult:
    try:
        return execute_run(
            run,
            manifest=manifest,
            preflight=preflight,
            ledger=ledger,
            secrets=secrets,
            retry_failed=retry_failed,
            recover_running=recover_running,
        )
    except Exception as unexpected_error:
        record = ledger.get(run.run_id)
        if record.status != "running":
            raise
        attempt = record.attempts
        attempt_dir = (
            manifest.campaign.artifact_dir / "runs" / run.run_id / f"attempt-{attempt:03d}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        error = f"Launcher failure: {type(unexpected_error).__name__}: {unexpected_error}"
        atomic_write_json(
            attempt_dir / "launcher-error.json",
            {
                "run_id": run.run_id,
                "attempt": attempt,
                "status": "failed",
                "error": error,
            },
        )
        ledger.finish(
            run.run_id,
            return_code=125,
            artifact_dir=attempt_dir,
            error=error,
        )
        return ExecutionResult(
            run_id=run.run_id,
            status="failed",
            attempt=attempt,
            return_code=125,
            artifact_dir=attempt_dir,
            error=error,
        )


def build_run_environment(
    run: PlannedRun,
    *,
    manifest: ExperimentManifest,
    preflight: Preflight,
    attempt_dir: Path,
    metrics_path: Path,
    resource_trace_path: Path,
) -> dict[str, str]:
    environment = {
        "RXNHAYSTACK_RUN_ID": run.run_id,
        "RXNHAYSTACK_RUN_DIR": str(attempt_dir),
        METRICS_PATH_ENV: str(metrics_path),
        USD_TO_CHF_ENV: str(manifest.campaign.usd_to_chf),
        "RXNHAYSTACK_TASK": run.task,
        "RXNHAYSTACK_CONDITION": run.condition,
        "RXNHAYSTACK_METHOD": run.method,
        "RXNHAYSTACK_MODEL": run.model,
        "RXNHAYSTACK_CORPUS_SIZE": str(run.corpus_size),
        "RXNHAYSTACK_SEED": str(run.seed),
        "RXNHAYSTACK_REPETITION": str(run.repetition),
        "RXNHAYSTACK_QUESTION_PARALLELISM": str(run.question_parallelism),
        RESOURCE_TRACE_PATH_ENV: str(resource_trace_path),
    }
    if run.positive_cardinality is not None:
        environment["RXNHAYSTACK_POSITIVE_CARDINALITY"] = str(run.positive_cardinality)
    if preflight.dataset is not None:
        environment[RAW_DATASET_ENV] = preflight.dataset.raw_path
        environment[CLEANED_DATASET_ENV] = preflight.dataset.cleaned_path
    return environment


def run_selected(
    manifest: ExperimentManifest,
    *,
    preflight: Preflight,
    selected: list[PlannedRun],
    secrets: dict[str, str],
    max_parallel: int,
    retry_failed: bool,
    recover_running: bool,
) -> list[ExecutionResult]:
    if max_parallel < 1:
        raise ManifestError("max_parallel must be at least 1")
    ledger = RunLedger(manifest.campaign.artifact_dir / "ledger.sqlite3")
    ledger.sync_runs(manifest.runs, manifest_sha256=manifest.sha256)
    enforce_remaining_budget(manifest, ledger)
    results: list[ExecutionResult] = []
    memory_budget = MemoryBudget(manifest.campaign.max_parallel_memory_mib)

    def execute_with_reservation(run: PlannedRun) -> ExecutionResult:
        with memory_budget.reserve(run.memory_reservation_mib):
            return execute_run_safely(
                run,
                manifest=manifest,
                preflight=preflight,
                ledger=ledger,
                secrets=secrets,
                retry_failed=retry_failed,
                recover_running=recover_running,
            )

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(execute_with_reservation, run): run.run_id for run in selected}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda result: result.run_id)
