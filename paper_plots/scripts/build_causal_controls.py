#!/usr/bin/env python3
"""Freeze the completed matched-cardinality and oracle controls for the paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rxnhaystack.control_room import (
    FULL_CAMPAIGN,
    MATCHED_CAMPAIGN,
    ORACLE_CAMPAIGN,
    load_snapshot_directory,
    merge_snapshots,
    scientific_dashboard_view,
)
from rxnhaystack.score_recovery import (
    SUBMISSION_SCORE_FREEZE_ID,
    apply_corrected_score_recoveries,
    carry_forward_pending_corrected_scores,
)

EXECUTOR_CAMPAIGN = "iclr2027-oracle-executor-v1"
GPT = "openai/gpt-5-mini"
QWEN = "RCP-AIaaS/Qwen/Qwen3.5-397B-A17B"
CLAUDE = "anthropic/claude-haiku-4.5"
MODEL_LABELS = {GPT: "GPT-5 mini", QWEN: "Qwen 3.5", CLAUDE: "Claude Haiku 4.5"}

QUESTION_COUNTS = {
    "tier1/task1": 10,
    "tier2/task2": 6,
    "tier2/task3": 5,
    "tier2/task4": 5,
    "tier2/task5": 4,
    "tier3/task6": 4,
    "tier3/task7": 5,
    "tier3/task8": 2,
    "tier3/task9": 4,
    "tier3/task10": 5,
    "tier3/task10b": 5,
    "tier3/task13": 1,
    "tier3/task14": 1,
    "tier3/task15": 1,
    "tier3/task17": 1,
    "tier3/task18": 1,
    "tier3/task20": 1,
    "tier3/task21": 1,
    "tier3/task22": 1,
    "tier3/task23": 1,
    "tier3/task24": 1,
    "tier4/task13": 4,
    "tier4/task14": 2,
}
ORACLE_TASKS = frozenset(
    {"tier3/task6", "tier3/task10", "tier3/task23", "tier4/task13", "tier4/task14"}
)
FIELDNAMES = (
    "study",
    "arm",
    "model",
    "model_label",
    "context",
    "condition",
    "tier",
    "task",
    "repetition",
    "question_count",
    "f1",
    "score_available",
    "original_f1",
    "score_correction_status",
    "ground_truth_version",
    "calls",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_seconds",
    "tool_time_seconds",
    "process_wall_time_seconds",
    "peak_combined_memory_mib",
    "run_id",
    "result_updated_at",
    "sources",
)
PROVISIONAL_FIELDNAMES = (*FIELDNAMES[:10], "status", *FIELDNAMES[10:])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def successful_attempt(run: dict[str, Any]) -> dict[str, Any]:
    attempts = [attempt for attempt in run["attempts"] if attempt["status"] == "succeeded"]
    if run["status"] != "succeeded" or not attempts:
        raise ValueError(f"Control record is not scientifically complete: {run['run_id']}")
    return max(attempts, key=lambda item: item["started_at"])


def successful_score(run: dict[str, Any]) -> float:
    attempt = successful_attempt(run)
    results = (attempt.get("metrics") or {}).get("results") or {}
    value = results.get("macro_f1")
    if value is None:
        raise ValueError(f"Successful control has no macro_f1: {run['run_id']}")
    return float(value)


def context_label(value: Any) -> str:
    return "full" if str(value) == "full" else str(int(value))


def row(
    run: dict[str, Any], *, study: str, arm: str, model_label: str | None = None
) -> dict[str, Any]:
    task = str(run["task"])
    attempt = successful_attempt(run)
    metrics = attempt.get("metrics") or {}
    resources = metrics.get("resources") or {}
    return {
        "study": study,
        "arm": arm,
        "model": run["model"],
        "model_label": model_label or MODEL_LABELS[str(run["model"])],
        "context": context_label(run["corpus_size"]),
        "condition": run["condition"],
        "tier": int(task.removeprefix("tier").split("/", 1)[0]),
        "task": task,
        "repetition": int(run["repetition"]),
        "question_count": QUESTION_COUNTS[task],
        "f1": successful_score(run),
        "score_available": True,
        "original_f1": "",
        "score_correction_status": "",
        "ground_truth_version": "",
        **{
            field: metrics.get(field)
            for field in (
                "calls",
                "cost_usd",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "latency_seconds",
                "tool_time_seconds",
            )
        },
        "process_wall_time_seconds": resources.get("process_wall_time_seconds"),
        "peak_combined_memory_mib": resources.get("peak_combined_memory_mib"),
        "run_id": run["run_id"],
        "result_updated_at": run.get("result_updated_at") or "",
        "sources": ";".join(run.get("sources", ())),
    }


def provisional_matched_row(run: dict[str, Any]) -> dict[str, Any]:
    """Preserve an unfinished Qwen cell without treating it as final evidence."""
    task = str(run["task"])
    status = str(run["status"])
    output: dict[str, Any] = {
        "study": "matched_cardinality",
        "arm": "matched-provisional",
        "model": run["model"],
        "model_label": MODEL_LABELS[str(run["model"])],
        "context": context_label(run["corpus_size"]),
        "condition": run["condition"],
        "tier": int(task.removeprefix("tier").split("/", 1)[0]),
        "task": task,
        "repetition": int(run["repetition"]),
        "question_count": QUESTION_COUNTS[task],
        "status": status,
        "f1": "",
        "score_available": False,
        "original_f1": "",
        "score_correction_status": "",
        "ground_truth_version": "",
        "calls": "",
        "cost_usd": "",
        "input_tokens": "",
        "output_tokens": "",
        "total_tokens": "",
        "latency_seconds": "",
        "tool_time_seconds": "",
        "process_wall_time_seconds": "",
        "peak_combined_memory_mib": "",
        "run_id": run["run_id"],
        "result_updated_at": run.get("result_updated_at") or "",
        "sources": ";".join(run.get("sources", ())),
    }
    if status == "succeeded":
        completed = row(run, study="matched_cardinality", arm="matched-provisional")
        output.update(completed)
        output["status"] = status
    return output


def build_rows(campaigns: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    matched = [run for run in campaigns[MATCHED_CAMPAIGN]["runs"] if run["model"] == GPT]
    if len(matched) != 725 or any(run["status"] != "succeeded" for run in matched):
        raise ValueError("GPT-5-mini matched-cardinality arm must be exactly 725/725 succeeded")

    predicate = campaigns[ORACLE_CAMPAIGN]["runs"]
    if len(predicate) != 150 or any(run["status"] != "succeeded" for run in predicate):
        raise ValueError("Oracle-predicate study must be exactly 150/150 succeeded")

    executor = campaigns[EXECUTOR_CAMPAIGN]["runs"]
    if len(executor) != 15 or any(run["status"] != "succeeded" for run in executor):
        raise ValueError("Oracle executor must be exactly 15/15 succeeded")

    ordinary = [
        run
        for run in campaigns[FULL_CAMPAIGN]["runs"]
        if run["method"] == "rlm" and run["model"] in {QWEN, CLAUDE} and run["task"] in ORACLE_TASKS
    ]
    if len(ordinary) != 150 or any(run["status"] != "succeeded" for run in ordinary):
        raise ValueError("Ordinary oracle-comparison cells must be exactly 150/150 succeeded")

    rows = [row(run, study="matched_cardinality", arm="matched") for run in matched]
    rows.extend(row(run, study="oracle_predicate", arm="ordinary") for run in ordinary)
    rows.extend(row(run, study="oracle_predicate", arm="predicate") for run in predicate)
    rows.extend(
        row(run, study="oracle_predicate", arm="executor", model_label="Deterministic executor")
        for run in executor
    )
    if len(rows) != 1040:
        raise AssertionError(f"Expected 1,040 frozen records, got {len(rows)}")
    if any(record["study"] == "matched_cardinality" and record["model"] != GPT for record in rows):
        raise AssertionError("Unfinished Qwen matched-cardinality records entered the freeze")
    return sorted(rows, key=lambda item: (item["study"], item["arm"], item["run_id"]))


def apply_ground_truth_corrections(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Invalidate stale scores while preserving immutable control-run evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    affected = {str(task) for task in payload["affected_tasks"]}
    for record in rows:
        record["ground_truth_version"] = str(payload["corrected_bundle"])
        record["original_f1"] = record.get("f1", "")
        if str(record["task"]) not in affected:
            record["score_correction_status"] = "not_affected"
            continue
        if bool(record.get("score_available")):
            record["score_available"] = False
            record["f1"] = ""
            record["score_correction_status"] = "historical_score_invalidated"
            record["sources"] = f"{record['sources']};ground-truth:{payload['correction_id']}"
        else:
            record["score_correction_status"] = "affected_without_historical_score"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/control-room/shared"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper_plots/gold/iclr2027/causal_controls")
    )
    parser.add_argument(
        "--ground-truth-corrections",
        type=Path,
        default=Path("paper_plots/gold/ground_truth_corrections.json"),
    )
    parser.add_argument(
        "--corrected-score-recoveries",
        type=Path,
        default=Path("paper_plots/gold/corrected_score_recoveries.json"),
    )
    args = parser.parse_args()

    snapshots = load_snapshot_directory(args.cache_dir)
    merged = scientific_dashboard_view(
        merge_snapshots(snapshots, stale_after_seconds=10 * 365 * 24 * 3600)
    )
    campaigns = {campaign["name"]: campaign for campaign in merged["campaigns"]}
    required = {FULL_CAMPAIGN, MATCHED_CAMPAIGN, ORACLE_CAMPAIGN, EXECUTOR_CAMPAIGN}
    missing = required - campaigns.keys()
    if missing:
        raise ValueError(f"Missing dashboard studies: {sorted(missing)}")

    rows = build_rows(campaigns)
    qwen_matched = [run for run in campaigns[MATCHED_CAMPAIGN]["runs"] if run["model"] == QWEN]
    if len(qwen_matched) != 725:
        raise ValueError(
            f"Qwen matched-cardinality grid must contain 725 jobs, got {len(qwen_matched)}"
        )
    qwen_rows = sorted(
        (provisional_matched_row(run) for run in qwen_matched), key=lambda item: item["run_id"]
    )
    correction_manifest = apply_ground_truth_corrections(rows, args.ground_truth_corrections)
    apply_ground_truth_corrections(qwen_rows, args.ground_truth_corrections)
    recovery_manifest, recovered_final = apply_corrected_score_recoveries(
        rows, args.corrected_score_recoveries
    )
    _provisional_manifest, recovered_provisional = apply_corrected_score_recoveries(
        qwen_rows, args.corrected_score_recoveries
    )
    carried_final = carry_forward_pending_corrected_scores(rows)
    carried_provisional = carry_forward_pending_corrected_scores(qwen_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.csv"
    with records_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provisional_path = args.output_dir / "matched_qwen_provisional.csv"
    with provisional_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROVISIONAL_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(qwen_rows)

    source_ids = sorted(
        {source for record in rows for source in record["sources"].split(";") if source}
    )
    snapshot_files = {}
    for path in sorted(args.cache_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source", {}).get("id") in source_ids:
            snapshot_files[str(path)] = sha256_file(path)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "record_count": len(rows),
        "records_sha256": sha256_file(records_path),
        "matched_qwen_provisional_sha256": sha256_file(provisional_path),
        "ground_truth_corrections": {
            "path": str(args.ground_truth_corrections),
            "sha256": sha256_file(args.ground_truth_corrections),
            "correction_id": correction_manifest["correction_id"],
            "corrected_bundle": correction_manifest["corrected_bundle"],
            "policy": correction_manifest["policy"],
            "final_records": dict(Counter(row["score_correction_status"] for row in rows)),
            "provisional_records": dict(
                Counter(row["score_correction_status"] for row in qwen_rows)
            ),
        },
        "corrected_score_recoveries": {
            "path": str(args.corrected_score_recoveries),
            "sha256": sha256_file(args.corrected_score_recoveries),
            "recovery_id": recovery_manifest["recovery_id"],
            "available": len(recovery_manifest["recoveries"]),
            "applied_final": recovered_final,
            "applied_provisional": recovered_provisional,
        },
        "submission_score_freeze": {
            "freeze_id": SUBMISSION_SCORE_FREEZE_ID,
            "policy": (
                "Use exact corrected rescores where available; otherwise carry forward the "
                "preserved historical score while retaining historical_score_invalidated "
                "status in the internal post-submission queue."
            ),
            "carried_final": carried_final,
            "carried_provisional": carried_provisional,
        },
        "selection": {
            "matched_cardinality": "GPT-5 mini only; 725/725 succeeded",
            "oracle_predicate": "Qwen 3.5 and Claude Haiku 4.5; 150/150 succeeded",
            "ordinary_comparison": "same models, tasks, contexts, and repetitions; 150/150 succeeded",
            "deterministic_executor": "15/15 succeeded",
            "matched_qwen_provisional": dict(Counter(row["status"] for row in qwen_rows)),
            "excluded_from_final_inference": "unfinished Qwen matched-cardinality arm",
        },
        "snapshot_files": snapshot_files,
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} causal-control records to {records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
