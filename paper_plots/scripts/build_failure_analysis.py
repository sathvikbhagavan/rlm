#!/usr/bin/env python3
"""Build the corrected, trace-backed ICLR failure-analysis tables."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METHODS = ("llm", "codeact", "rlm")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(
    path: Path, rows: list[dict[str, Any]], *, fieldnames: tuple[str, ...] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else list(fieldnames or ())
    if not columns:
        raise ValueError(f"Empty table requires explicit columns: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def outcome(row: dict[str, str]) -> str:
    if row["status"] != "succeeded":
        return "execution failure"
    score = float(row["f1"])
    if score >= 1.0 - 1e-12:
        return "exact"
    if score <= 1e-12:
        return "zero score"
    return "partial"


def error_signature(question: dict[str, Any]) -> str | None:
    required = {"corrected_precision", "corrected_recall", "predicted_count"}
    if not required.issubset(question):
        return None
    precision = float(question["corrected_precision"])
    recall = float(question["corrected_recall"])
    if float(question["corrected_f1"]) >= 1.0 - 1e-12:
        return "exact"
    if int(question["predicted_count"]) == 0:
        return "empty answer"
    if precision >= 1.0 - 1e-12 and recall < 1.0 - 1e-12:
        return "omissions only"
    if recall >= 1.0 - 1e-12 and precision < 1.0 - 1e-12:
        return "extra selections only"
    return "mixed omissions and extras"


def build(root: Path) -> dict[str, Any]:
    gold = root / "paper_plots/gold/iclr2027"
    records_path = gold / "final_arm_records.csv"
    pending_path = gold / "post_submission/pending_corrected_rescores.csv"
    recoveries_path = root / "paper_plots/gold/corrected_score_recoveries.json"
    output = gold / "failure_analysis"

    records = read_csv(records_path)
    pending = {row["run_id"] for row in read_csv(pending_path)}
    valid = [row for row in records if row["run_id"] not in pending]
    by_run = {row["run_id"]: row for row in valid}

    outcome_rows: list[dict[str, Any]] = []
    for method in METHODS:
        members = [row for row in valid if row["method"] == method]
        counts = Counter(outcome(row) for row in members)
        for category in ("exact", "partial", "zero score", "execution failure"):
            outcome_rows.append(
                {
                    "method": method,
                    "outcome": category,
                    "jobs": counts[category],
                    "all_valid_jobs": len(members),
                    "fraction": counts[category] / len(members),
                }
            )

    recovery_data = json.loads(recoveries_path.read_text())
    signatures: list[dict[str, str]] = []
    unavailable = 0
    for recovery in recovery_data["recoveries"]:
        run = by_run.get(recovery["run_id"])
        if run is None:
            continue
        for question in recovery["question_scores"]:
            signature = error_signature(question)
            if signature is None:
                unavailable += 1
                continue
            signatures.append(
                {
                    "run_id": recovery["run_id"],
                    "model": run["model_label"],
                    "method": run["method"],
                    "task": run["task"],
                    "question_id": question["question_id"],
                    "signature": signature,
                    "corrected_f1": str(question["corrected_f1"]),
                    "corrected_precision": str(question["corrected_precision"]),
                    "corrected_recall": str(question["corrected_recall"]),
                    "predicted_count": str(question["predicted_count"]),
                    "ground_truth_count": str(question["corrected_ground_truth_count"]),
                    "recovery": question["recovery"],
                }
            )

    signature_rows: list[dict[str, Any]] = []
    for method in METHODS:
        members = [row for row in signatures if row["method"] == method]
        errors = [row for row in members if row["signature"] != "exact"]
        counts = Counter(row["signature"] for row in errors)
        for category in (
            "empty answer",
            "omissions only",
            "extra selections only",
            "mixed omissions and extras",
        ):
            signature_rows.append(
                {
                    "method": method,
                    "signature": category,
                    "question_outputs": counts[category],
                    "erroneous_question_outputs": len(errors),
                    "all_classifiable_question_outputs": len(members),
                    "fraction_of_errors": counts[category] / len(errors) if errors else 0.0,
                }
            )

    task_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in signatures:
        grouped[(row["method"], row["task"])].append(row)
    for (method, task), members in sorted(grouped.items()):
        counts = Counter(row["signature"] for row in members)
        task_rows.append(
            {
                "method": method,
                "task": task,
                "question_outputs": len(members),
                "exact": counts["exact"],
                "empty_answer": counts["empty answer"],
                "omissions_only": counts["omissions only"],
                "extra_selections_only": counts["extra selections only"],
                "mixed_omissions_and_extras": counts["mixed omissions and extras"],
            }
        )

    paths = {
        "job_outcomes": output / "job_outcomes.csv",
        "trace_records": output / "trace_error_records.csv",
        "trace_signatures": output / "trace_error_signatures.csv",
        "task_signatures": output / "task_error_signatures.csv",
    }
    write_csv(paths["job_outcomes"], outcome_rows)
    write_csv(
        paths["trace_records"],
        signatures,
        fieldnames=(
            "run_id",
            "model",
            "method",
            "task",
            "question_id",
            "signature",
            "corrected_f1",
            "corrected_precision",
            "corrected_recall",
            "predicted_count",
            "ground_truth_count",
            "recovery",
        ),
    )
    write_csv(
        paths["trace_signatures"],
        signature_rows,
        fieldnames=(
            "method",
            "signature",
            "question_outputs",
            "erroneous_question_outputs",
            "all_classifiable_question_outputs",
            "fraction_of_errors",
        ),
    )
    write_csv(
        paths["task_signatures"],
        task_rows,
        fieldnames=(
            "method",
            "task",
            "question_outputs",
            "exact",
            "empty_answer",
            "omissions_only",
            "extra_selections_only",
            "mixed_omissions_and_extras",
        ),
    )

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "paper_models": sorted({row["model_label"] for row in valid}),
        "frozen_jobs": len(records),
        "excluded_pending_corrected_rescores": len(records) - len(valid),
        "valid_jobs": len(valid),
        "valid_expected_trajectories": sum(int(row["question_count"]) for row in valid),
        "trace_classifiable_question_outputs": len(signatures),
        "trace_outputs_without_exact_precision_recall": unavailable,
        "trace_tasks": sorted({row["task"] for row in signatures}),
        "ground_truth_version": "rxnhaystack-human-1.6.0",
        "inputs": {
            str(path.relative_to(root)): sha256(path)
            for path in (records_path, pending_path, recoveries_path)
        },
        "outputs": {str(path.relative_to(root)): sha256(path) for path in paths.values()},
        "interpretation": {
            "job_outcomes": "All valid final-arm jobs after mandatory pending-rescore exclusion.",
            "trace_signatures": (
                "Corrected-task question outputs with retained predictions and exact corrected "
                "precision/recall. Signatures describe set errors, not inferred chemical causes."
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    summary = build(Path(__file__).resolve().parents[2])
    print(json.dumps(summary, indent=2, sort_keys=True))
