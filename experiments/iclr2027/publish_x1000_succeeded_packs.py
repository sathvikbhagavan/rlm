from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from rxnhaystack.control_room import (
    load_snapshot,
    publish_snapshot,
    safe_metrics,
    snapshot_digest,
    validate_snapshot,
    write_snapshot,
)

CAMPAIGN = "iclr2027-sathvik-x1000-succeeded-pack-v1"
SOURCE_ID = "sathvik-x1000-succeeded-packs-v1"
DEEPSEEK_SOURCE_ID = "sathvik-deepseek-x1000-docker-pack-v1"
QWEN_MODEL = "qwen3.5-397b"
GEMINI_MODEL = "gemini-3.7-flash"
DEEPSEEK_MODEL = "deepseek-v4-flash"


def nested_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in row.items():
        if not key.startswith("metrics.") or value is None:
            continue
        cursor = metrics
        parts = key.removeprefix("metrics.").split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    if row.get("wandb_url") and "wandb_url" not in metrics:
        metrics["wandb_url"] = row["wandb_url"]
    return metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_pack(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            parts = PurePosixPath(member.name).parts
            if member.name.startswith("/") or ".." in parts or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member: {member.name}")
        manifests = [member for member in members if member.name.endswith("/manifest.json")]
        runs_files = [member for member in members if member.name.endswith("/runs.json")]
        if len(manifests) != 1 or len(runs_files) != 1:
            raise ValueError(f"{path.name} must contain one manifest.json and one runs.json")
        manifest_handle = archive.extractfile(manifests[0])
        runs_handle = archive.extractfile(runs_files[0])
        if manifest_handle is None or runs_handle is None:
            raise ValueError(f"Cannot read {path.name}")
        manifest = json.load(manifest_handle)
        rows = json.load(runs_handle)
    if not isinstance(manifest, dict) or not isinstance(rows, list):
        raise ValueError(f"Invalid JSON payload in {path.name}")
    if manifest.get("n_runs") != len(rows):
        raise ValueError(f"Row count in {path.name} disagrees with its manifest")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Every packed run in {path.name} must be an object")
    return manifest, rows


def validate_rows(
    rows: list[dict[str, Any]], *, model: str, method: str, docker_only: bool
) -> None:
    expected_count = 15 if docker_only else 150
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} {model} {method} rows, found {len(rows)}")
    identities: set[tuple[str, str, int]] = set()
    run_ids: set[str] = set()
    for row in rows:
        run_id = str(row.get("run_id", ""))
        repetition = int(row.get("repetition", 0))
        tier = str(row.get("tier", ""))
        task = str(row.get("task", ""))
        expected_id = f"full-{model}-tier{tier}-task{task}-{method}-x1000-r{repetition:02d}"
        if run_id != expected_id:
            raise ValueError(f"Unexpected packed run identity: {run_id}")
        if row.get("model") != model or row.get("method") != method:
            raise ValueError(f"Unexpected model/method in {run_id}")
        if str(row.get("context")) != "1000" or row.get("status") != "succeeded":
            raise ValueError(f"Only successful x1000 records are accepted: {run_id}")
        if repetition not in range(1, 6):
            raise ValueError(f"Unexpected repetition in {run_id}")
        is_docker_task = tier == "4" and task in {"16", "17", "17b"}
        if docker_only and not is_docker_task:
            raise ValueError(f"Docker-only pack contains a non-Docker task: {run_id}")
        identities.add((tier, task, repetition))
        run_ids.add(run_id)
    if len(run_ids) != len(rows) or len(identities) != len(rows):
        raise ValueError(f"Duplicate run in the {model} pack")
    if not docker_only:
        task_counts = Counter((tier, task) for tier, task, _ in identities)
        if len(task_counts) != 30 or set(task_counts.values()) != {5}:
            raise ValueError(
                "Full x1000 pack must contain five repetitions of all 30 task configurations"
            )
    else:
        expected = {
            ("4", task, repetition) for task in ("16", "17", "17b") for repetition in range(1, 6)
        }
        if identities != expected:
            raise ValueError("Docker pack must contain five repetitions of Tasks 16/17/17b")


def stable_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def build_pack_snapshot(
    qwen_codeact_path: Path,
    gemini_codeact_path: Path,
    gemini_rlm_docker_path: Path,
) -> dict[str, Any]:
    inputs = (
        (qwen_codeact_path, QWEN_MODEL, "codeact", False, ""),
        (gemini_codeact_path, GEMINI_MODEL, "codeact", False, ""),
        (gemini_rlm_docker_path, GEMINI_MODEL, "rlm", True, ""),
    )
    expected_runs: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    packed_times: list[datetime] = []
    archive_hashes: list[str] = []
    for path, model, method, docker_only, canonical_prefix in inputs:
        manifest, rows = load_pack(path)
        validate_rows(rows, model=model, method=method, docker_only=docker_only)
        packed_at = datetime.fromisoformat(str(manifest["packed_at"]))
        if packed_at.tzinfo is None:
            packed_at = packed_at.replace(tzinfo=UTC)
        packed_times.append(packed_at)
        archive_hashes.append(sha256_file(path))
        for row in rows:
            metrics = safe_metrics(nested_metrics(row))
            resources = metrics.get("resources", {})
            run_id = canonical_prefix + str(row["run_id"])
            repetition = int(row["repetition"])
            spec_material = {
                "run_id": run_id,
                "model": model,
                "method": str(row["method"]),
                "task": f"tier{row['tier']}/task{row['task']}",
                "condition": "x1000-extension",
                "corpus_size": 1000,
                "repetition": repetition,
            }
            spec_hash = stable_hash(spec_material)
            expected_runs.append(
                {
                    **spec_material,
                    "spec_hash": spec_hash,
                    "estimated_cost_chf": float(metrics.get("cost_chf", 0.0)),
                    "memory_limit_mib": resources.get("memory_limit_mib"),
                    "question_parallelism": 1,
                }
            )
            wall_seconds = float(resources.get("process_wall_time_seconds", 0.0))
            finished_at = packed_at
            started_at = finished_at - timedelta(seconds=wall_seconds)
            attempt = int(row.get("attempt", 1))
            attempt_key = stable_hash(
                {
                    "run_id": run_id,
                    "spec_hash": spec_hash,
                    "attempt": attempt,
                    "started_at": started_at.isoformat(),
                    "archive_sha256": archive_hashes[-1],
                }
            )
            observations.append(
                {
                    "run_id": run_id,
                    "spec_hash": spec_hash,
                    "status": "succeeded",
                    "attempt_count": 1,
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "attempts": [
                        {
                            "attempt_key": attempt_key,
                            "attempt": attempt,
                            "status": "succeeded",
                            "started_at": started_at.isoformat(),
                            "finished_at": finished_at.isoformat(),
                            "return_code": 0,
                            "failure_category": None,
                            "metrics": metrics,
                        }
                    ],
                }
            )
    generated_at = max(packed_times).astimezone(UTC).isoformat()
    definition_sha256 = stable_hash(sorted(archive_hashes))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": generated_at,
        "source": {
            "id": SOURCE_ID,
            "machine": "liacpc15",
            "owner": "Sathvik",
            "scheduler_job_id": None,
            "session_name": "provided-succeeded-packs",
            "git_commit": "external-result-pack",
            "tracked_dirty": False,
        },
        "experiment": {
            "name": CAMPAIGN,
            "definition_sha256": definition_sha256,
            "recorded_definition_sha256": sorted(archive_hashes),
            "expected_cost_chf": sum(float(spec["estimated_cost_chf"]) for spec in expected_runs),
            "expected_runs": sorted(expected_runs, key=lambda item: item["run_id"]),
        },
        "observations": sorted(observations, key=lambda item: item["run_id"]),
    }
    payload["snapshot_id"] = snapshot_digest(payload)
    validate_snapshot(payload)
    return payload


def build_deepseek_repair_snapshot(pack_path: Path, reference_path: Path) -> dict[str, Any]:
    """Attach the successful Docker pack to the existing 150-cell campaign identity."""
    reference = load_snapshot(reference_path)
    manifest, rows = load_pack(pack_path)
    validate_rows(rows, model=DEEPSEEK_MODEL, method="rlm", docker_only=True)
    expected = {item["run_id"]: item for item in reference["experiment"]["expected_runs"]}
    observations: list[dict[str, Any]] = []
    packed_at = datetime.fromisoformat(str(manifest["packed_at"]))
    if packed_at.tzinfo is None:
        packed_at = packed_at.replace(tzinfo=UTC)
    archive_hash = sha256_file(pack_path)
    for row in rows:
        run_id = "x1000-openrouter-" + str(row["run_id"])
        if run_id not in expected:
            raise ValueError(f"Packed run is absent from the reference campaign: {run_id}")
        metrics = safe_metrics(nested_metrics(row))
        resources = metrics.get("resources", {})
        wall_seconds = float(resources.get("process_wall_time_seconds", 0.0))
        started_at = packed_at - timedelta(seconds=wall_seconds)
        attempt = int(row.get("attempt", 1))
        attempt_key = stable_hash(
            {
                "run_id": run_id,
                "spec_hash": expected[run_id]["spec_hash"],
                "attempt": attempt,
                "started_at": started_at.isoformat(),
                "archive_sha256": archive_hash,
            }
        )
        observations.append(
            {
                "run_id": run_id,
                "spec_hash": expected[run_id]["spec_hash"],
                "status": "succeeded",
                "attempt_count": 1,
                "started_at": started_at.isoformat(),
                "finished_at": packed_at.isoformat(),
                "attempts": [
                    {
                        "attempt_key": attempt_key,
                        "attempt": attempt,
                        "status": "succeeded",
                        "started_at": started_at.isoformat(),
                        "finished_at": packed_at.isoformat(),
                        "return_code": 0,
                        "failure_category": None,
                        "metrics": metrics,
                    }
                ],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": packed_at.astimezone(UTC).isoformat(),
        "source": {
            "id": DEEPSEEK_SOURCE_ID,
            "machine": "liacpc15",
            "owner": "Sathvik",
            "scheduler_job_id": None,
            "session_name": "provided-deepseek-x1000-docker-pack",
            "git_commit": "external-result-pack",
            "tracked_dirty": False,
        },
        "experiment": reference["experiment"],
        "observations": sorted(observations, key=lambda item: item["run_id"]),
    }
    payload["snapshot_id"] = snapshot_digest(payload)
    validate_snapshot(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and publish Sathvik's successful x1000 result packs."
    )
    parser.add_argument("--qwen-codeact-pack", type=Path, required=True)
    parser.add_argument("--gemini-codeact-pack", type=Path, required=True)
    parser.add_argument("--gemini-rlm-docker-pack", type=Path, required=True)
    parser.add_argument("--deepseek-rlm-docker-pack", type=Path, required=True)
    parser.add_argument("--deepseek-reference-snapshot", type=Path, required=True)
    parser.add_argument("--deepseek-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-key-file", type=Path)
    parser.add_argument("--entity", default="liac")
    parser.add_argument("--project", default="rxnhaystack-dashboard")
    args = parser.parse_args()
    snapshot = build_pack_snapshot(
        args.qwen_codeact_pack.resolve(),
        args.gemini_codeact_pack.resolve(),
        args.gemini_rlm_docker_pack.resolve(),
    )
    write_snapshot(args.output.resolve(), snapshot)
    print(
        f"Validated {len(snapshot['observations'])} successful x1000 records; "
        f"snapshot {snapshot['snapshot_id'][:12]}"
    )
    deepseek_snapshot = build_deepseek_repair_snapshot(
        args.deepseek_rlm_docker_pack.resolve(), args.deepseek_reference_snapshot.resolve()
    )
    write_snapshot(args.deepseek_output.resolve(), deepseek_snapshot)
    print(
        f"Validated {len(deepseek_snapshot['observations'])} DeepSeek Docker records; "
        f"snapshot {deepseek_snapshot['snapshot_id'][:12]}"
    )
    if args.wandb_key_file is not None:
        key = args.wandb_key_file.expanduser().read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("W&B key file is empty")
        url = publish_snapshot(
            args.output.resolve(), api_key=key, entity=args.entity, project=args.project
        )
        print(f"Published {SOURCE_ID}: {url}")
        deepseek_url = publish_snapshot(
            args.deepseek_output.resolve(), api_key=key, entity=args.entity, project=args.project
        )
        print(f"Published {DEEPSEEK_SOURCE_ID}: {deepseek_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
