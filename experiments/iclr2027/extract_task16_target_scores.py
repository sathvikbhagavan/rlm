#!/usr/bin/env python3
"""Extract compact per-target Task-16 scores from large prediction artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_row(run_id: str, payload: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    score = question["scores"]
    return {
        "run_id": run_id,
        "model": payload["model"],
        "condition": payload["prompt_condition"],
        "repetition": int(run_id.rsplit("-r", 1)[1]),
        "target": question["question_id"],
        "canonical_question_id": question["canonical_question_id"],
        "f1": float(score["f1"]),
        "precision": float(score["precision"]),
        "recall": float(score["recall"]),
        "exact_match": float(score["is_exact_match"]),
        "ground_truth_chain_count": int(score["ground_truth_chain_count"]),
        "parsed_chain_count": int(score["parsed_chain_count"]),
        "valid_chain_count": int(score["valid_chain_count"]),
        "false_positive_chain_count": len(question["false_positive_chains"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    paths = sorted(args.artifact_root.glob("runs/*/attempt-*/task16-predictions.json"))
    if len(paths) != 30:
        raise ValueError(f"Expected 30 Task-16 prediction artifacts, found {len(paths)}")

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    for path in paths:
        run_id = path.parents[1].name
        if run_id in seen_runs:
            raise ValueError(f"Multiple prediction artifacts found for {run_id}")
        seen_runs.add(run_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = payload.get("questions") or []
        if len(questions) != 3 or not payload.get("excluded_all_target_products"):
            raise ValueError(f"Invalid prospective artifact: {path}")
        rows.extend(compact_row(run_id, payload, question) for question in questions)
        sources.append(
            {
                "run_id": run_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    if len(rows) != 90:
        raise AssertionError(f"Expected 90 target trajectories, got {len(rows)}")
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "selection": "30 completed jobs; three targets per job; direct score extraction",
        "rows": sorted(rows, key=lambda item: (item["run_id"], item["target"])),
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} target-level records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
