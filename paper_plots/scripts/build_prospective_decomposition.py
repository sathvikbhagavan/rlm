#!/usr/bin/env python3
"""Freeze the completed Task-16 prospective decomposition for the paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rxnhaystack.control_room import (
    PROSPECTIVE_CAMPAIGN,
    load_snapshot_directory,
    merge_snapshots,
    scientific_dashboard_view,
)

MODEL_LABELS = {
    "RCP-AIaaS/Qwen/Qwen3.5-397B-A17B": "Qwen",
    "anthropic/claude-haiku-4.5": "Claude",
}
CONDITIONS = ("name_only", "structure_only", "structure_plus_class")
RECORD_FIELDS = (
    "model",
    "model_label",
    "condition",
    "repetition",
    "question_count",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "exact_match_accuracy",
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
TARGET_FIELDS = (
    "model",
    "model_label",
    "condition",
    "repetition",
    "target",
    "canonical_question_id",
    "f1",
    "precision",
    "recall",
    "exact_match",
    "ground_truth_chain_count",
    "parsed_chain_count",
    "valid_chain_count",
    "false_positive_chain_count",
    "run_id",
)
AGGREGATE_FIELDS = (
    "model_label",
    "condition",
    "runs",
    "trajectories",
    "mean_macro_f1",
    "sd_macro_f1",
    "mean_precision",
    "mean_recall",
    "mean_exact_match_accuracy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def successful_metrics(run: dict[str, Any]) -> dict[str, float]:
    attempts = [attempt for attempt in run["attempts"] if attempt["status"] == "succeeded"]
    if run["status"] != "succeeded" or not attempts:
        raise ValueError(f"Prospective record is not scientifically complete: {run['run_id']}")
    attempt = max(attempts, key=lambda item: item["started_at"])
    results = (attempt.get("metrics") or {}).get("results") or {}
    required = ("macro_f1", "macro_precision", "macro_recall", "exact_match_accuracy")
    missing = [name for name in required if results.get(name) is None]
    if missing:
        raise ValueError(f"Successful record lacks {missing}: {run['run_id']}")
    if int(results.get("queries_evaluated", 0)) != 3:
        raise ValueError(f"Expected three targets in {run['run_id']}")
    return {name: float(results[name]) for name in required}


def successful_attempt(run: dict[str, Any]) -> dict[str, Any]:
    attempts = [attempt for attempt in run["attempts"] if attempt["status"] == "succeeded"]
    if run["status"] != "succeeded" or not attempts:
        raise ValueError(f"Prospective record is not scientifically complete: {run['run_id']}")
    return max(attempts, key=lambda item: item["started_at"])


def build_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    runs = campaign["runs"]
    if len(runs) != 30 or any(run["status"] != "succeeded" for run in runs):
        raise ValueError("Prospective decomposition must be exactly 30/30 succeeded")

    rows: list[dict[str, Any]] = []
    for run in runs:
        model = str(run["model"])
        condition = str(run["condition"])
        if model not in MODEL_LABELS or condition not in CONDITIONS:
            raise ValueError(f"Unexpected prospective arm: {model}, {condition}")
        metrics = successful_metrics(run)
        attempt_metrics = successful_attempt(run).get("metrics") or {}
        resources = attempt_metrics.get("resources") or {}
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "condition": condition,
                "repetition": int(run["repetition"]),
                "question_count": 3,
                **metrics,
                **{
                    field: attempt_metrics.get(field)
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
        )

    expected = {
        (label, condition, repetition)
        for label in MODEL_LABELS.values()
        for condition in CONDITIONS
        for repetition in range(1, 6)
    }
    observed = {(row["model_label"], row["condition"], row["repetition"]) for row in rows}
    if observed != expected:
        raise ValueError("Prospective model/condition/repetition grid is incomplete")
    return sorted(rows, key=lambda item: item["run_id"])


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates = []
    for model_label in MODEL_LABELS.values():
        for condition in CONDITIONS:
            group = [
                row
                for row in rows
                if row["model_label"] == model_label and row["condition"] == condition
            ]
            f1 = [float(row["macro_f1"]) for row in group]
            aggregates.append(
                {
                    "model_label": model_label,
                    "condition": condition,
                    "runs": len(group),
                    "trajectories": sum(int(row["question_count"]) for row in group),
                    "mean_macro_f1": statistics.mean(f1),
                    "sd_macro_f1": statistics.stdev(f1),
                    "mean_precision": statistics.mean(
                        float(row["macro_precision"]) for row in group
                    ),
                    "mean_recall": statistics.mean(float(row["macro_recall"]) for row in group),
                    "mean_exact_match_accuracy": statistics.mean(
                        float(row["exact_match_accuracy"]) for row in group
                    ),
                }
            )
    return aggregates


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/control-room/shared"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_plots/gold/iclr2027/prospective_decomposition"),
    )
    parser.add_argument(
        "--target-score-pack",
        type=Path,
        default=Path("paper_plots/gold/source_packs/task16-target-scores.json"),
    )
    args = parser.parse_args()

    snapshots = load_snapshot_directory(args.cache_dir)
    merged = scientific_dashboard_view(
        merge_snapshots(snapshots, stale_after_seconds=10 * 365 * 24 * 3600)
    )
    campaigns = {campaign["name"]: campaign for campaign in merged["campaigns"]}
    if PROSPECTIVE_CAMPAIGN not in campaigns:
        raise ValueError(f"Missing dashboard study: {PROSPECTIVE_CAMPAIGN}")

    rows = build_rows(campaigns[PROSPECTIVE_CAMPAIGN])
    aggregates = aggregate_rows(rows)
    target_payload = json.loads(args.target_score_pack.read_text(encoding="utf-8"))
    target_rows = target_payload.get("rows") or []
    if len(target_rows) != 90:
        raise ValueError("Task-16 target score pack must contain exactly 90 rows")
    labels = {model: label for model, label in MODEL_LABELS.items()}
    target_rows = [
        {**target, "model_label": labels[str(target["model"])]} for target in target_rows
    ]
    expected_run_ids = {row["run_id"] for row in rows}
    if {row["run_id"] for row in target_rows} != expected_run_ids:
        raise ValueError("Task-16 target score pack run IDs do not match the frozen study")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.csv"
    aggregates_path = args.output_dir / "aggregates.csv"
    targets_path = args.output_dir / "target_records.csv"
    write_csv(records_path, RECORD_FIELDS, rows)
    write_csv(aggregates_path, AGGREGATE_FIELDS, aggregates)
    write_csv(targets_path, TARGET_FIELDS, target_rows)

    source_ids = {source for row in rows for source in row["sources"].split(";") if source}
    snapshot_files = {}
    for path in sorted(args.cache_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source", {}).get("id") in source_ids:
            snapshot_files[str(path)] = sha256_file(path)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "campaign": PROSPECTIVE_CAMPAIGN,
        "selection": "Qwen and Claude; three conditions; five repetitions; three targets",
        "jobs": len(rows),
        "trajectories": sum(int(row["question_count"]) for row in rows),
        "records_sha256": sha256_file(records_path),
        "aggregates_sha256": sha256_file(aggregates_path),
        "target_records_sha256": sha256_file(targets_path),
        "target_score_pack": {
            "path": str(args.target_score_pack),
            "sha256": sha256_file(args.target_score_pack),
        },
        "snapshot_files": snapshot_files,
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} prospective records and {len(aggregates)} aggregates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
