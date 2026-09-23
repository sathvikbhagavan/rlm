#!/usr/bin/env python3
"""Freeze the completed matched-cardinality and oracle controls for the paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
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
    "run_id",
    "result_updated_at",
    "sources",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def successful_score(run: dict[str, Any]) -> float:
    attempts = [attempt for attempt in run["attempts"] if attempt["status"] == "succeeded"]
    if run["status"] != "succeeded" or not attempts:
        raise ValueError(f"Control record is not scientifically complete: {run['run_id']}")
    attempt = max(attempts, key=lambda item: item["started_at"])
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
        "run_id": run["run_id"],
        "result_updated_at": run.get("result_updated_at") or "",
        "sources": ";".join(run.get("sources", ())),
    }


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/control-room/shared"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper_plots/gold/iclr2027/causal_controls")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.csv"
    with records_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

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
        "selection": {
            "matched_cardinality": "GPT-5 mini only; 725/725 succeeded",
            "oracle_predicate": "Qwen 3.5 and Claude Haiku 4.5; 150/150 succeeded",
            "ordinary_comparison": "same models, tasks, contexts, and repetitions; 150/150 succeeded",
            "deterministic_executor": "15/15 succeeded",
            "excluded": "unfinished Qwen matched-cardinality arm",
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
