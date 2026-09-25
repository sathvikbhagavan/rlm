#!/usr/bin/env python3
"""Build per-model appendix tables from the frozen full-benchmark records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METHOD_ORDER = ("llm", "codeact", "rlm")
CONTEXT_ORDER = {"100": 0, "500": 1, "1000": 2, "full": 3}
ADDITIVE_METRICS = (
    "calls",
    "cost_usd",
    "total_tokens",
    "latency_seconds",
    "tool_time_seconds",
    "process_wall_time_seconds",
)
TASK_LABELS = {
    "tier1/task1": "T1.1 Exact product lookup (10Q)",
    "tier2/task2": "T2.1 Molecular-weight change (6Q)",
    "tier2/task3": "T2.2 Ring-count change (5Q)",
    "tier2/task4": "T2.3 Aromatic-ring formation (5Q)",
    "tier2/task5": "T2.4 Combined weight + rings (4Q)",
    "tier3/task13": "T3.1 New nitrogen atom (1Q)",
    "tier3/task14": "T3.2 C-N formed / C-O broken (1Q)",
    "tier3/task15": "T3.3 Single C-C bond formation (1Q)",
    "tier3/task17": "T3.4 Fused heterocycle (1Q)",
    "tier3/task18": "T3.5 New ring system (1Q)",
    "tier3/task20": "T3.6 Fused-ring construction (1Q)",
    "tier3/task21": "T3.7 Transition-metal reagent (1Q)",
    "tier3/task22": "T3.8 HATU / T3P reagent (1Q)",
    "tier3/task23": "T3.9 New stereocenter (1Q)",
    "tier3/task24": "T3.10 E-alkene formation (1Q)",
    "tier3/task6": "T3.11-14 Amide couplings (4Q)",
    "tier3/task7": "T3.15-19 Group transformations (5Q)",
    "tier3/task8": "T3.20-21 Protecting groups (2Q)",
    "tier3/task9": "T3.22-25 Named reactions (4Q)",
    "tier3/task10": "T3.26-30 Mechanisms I (5Q)",
    "tier3/task10b": "T3.31-35 Mechanisms II (5Q)",
    "tier4/task11": "T4.1-2 Fixed-length chains (2Q)",
    "tier4/task12": "T4.3-4 Longest chains (2Q)",
    "tier4/task12b": "T4.5 Hub molecules (1Q)",
    "tier4/task13": "T4.6-9 Group-constrained chains (4Q)",
    "tier4/task14": "T4.10-11 Protecting-group pairs (2Q)",
    "tier4/task15": "T4.12-15 Ring-construction chains (4Q)",
    "tier4/task16": "T4.16-25 Truncated routes (10Q)",
    "tier4/task17": "T4.26-30 SMIRKS chains I (5Q)",
    "tier4/task17b": "T4.31-35 SMIRKS chains II (5Q)",
}
TASK_ORDER = {task: index for index, task in enumerate(TASK_LABELS)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def population_std(values: list[float]) -> float:
    return statistics.pstdev(values) if values else 0.0


def as_bool(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def row_score_available(row: dict[str, Any]) -> bool:
    """Read the explicit score flag, with legacy-table compatibility."""
    if "score_available" in row:
        return as_bool(row["score_available"])
    return row.get("status") == "succeeded" and row.get("f1") not in {None, ""}


def summarize_model(
    rows: list[dict[str, str]], model: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return tier/resource and task-level summaries for one terminal model."""
    model_rows = [row for row in rows if row["model"] == model]
    if not model_rows:
        raise ValueError(f"No records found for model {model!r}")
    unresolved = [row for row in model_rows if row["status"] not in {"succeeded", "failed"}]
    if unresolved:
        raise ValueError(f"Model {model!r} has {len(unresolved)} unresolved records")

    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in model_rows:
        grouped[(row["method"], row["context"], int(row["tier"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (method, context, tier), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][2],
            METHOD_ORDER.index(item[0][0]),
            CONTEXT_ORDER[item[0][1]],
        ),
    ):
        by_repetition: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in group:
            by_repetition[int(row["repetition"])].append(row)
        repetition_values: dict[str, list[float]] = defaultdict(list)
        for repetition_group in by_repetition.values():
            scoreable = [
                row
                for row in repetition_group
                if row["status"] == "failed"
                or (row["status"] == "succeeded" and row_score_available(row))
            ]
            scoreable_trajectories = sum(int(row["question_count"]) for row in scoreable)
            successful = [row for row in repetition_group if row["status"] == "succeeded"]
            scored_successful = [row for row in successful if row_score_available(row)]
            successful_trajectories = sum(int(row["question_count"]) for row in successful)
            f1_numerator = sum(
                float(row["f1"]) * int(row["question_count"]) for row in scored_successful
            )
            if scoreable_trajectories:
                repetition_values["f1"].append(f1_numerator / scoreable_trajectories)
            if successful_trajectories:
                for metric in ADDITIVE_METRICS:
                    measured = [row for row in successful if row[metric] != ""]
                    measured_trajectories = sum(int(row["question_count"]) for row in measured)
                    if measured_trajectories != successful_trajectories:
                        raise ValueError(
                            f"Incomplete {metric} coverage for "
                            f"{model}/{method}/{context}/tier{tier}"
                        )
                    repetition_values[metric].append(
                        sum(float(row[metric]) for row in measured) / measured_trajectories
                    )
                memory = [float(row["peak_combined_memory_mib"]) for row in successful]
                if len(memory) != len(successful):
                    raise ValueError("Incomplete peak-memory coverage")
                repetition_values["peak_combined_memory_mib"].append(statistics.fmean(memory))

        output: dict[str, Any] = {
            "model": model,
            "method": method,
            "context": context,
            "tier": tier,
            "jobs": len(group),
            "successful_jobs": sum(row["status"] == "succeeded" for row in group),
            "failed_jobs": sum(row["status"] == "failed" for row in group),
            "question_trajectories": sum(int(row["question_count"]) for row in group),
            "scoreable_trajectories": sum(
                int(row["question_count"])
                for row in group
                if row["status"] == "failed"
                or (row["status"] == "succeeded" and row_score_available(row))
            ),
            "repetitions": len(by_repetition),
        }
        output["score_coverage"] = (
            output["scoreable_trajectories"] / output["question_trajectories"]
        )
        for metric in ("f1", *ADDITIVE_METRICS, "peak_combined_memory_mib"):
            values = repetition_values[metric]
            output[f"{metric}_mean"] = statistics.fmean(values) if values else None
            output[f"{metric}_std"] = population_std(values) if values else None
        summaries.append(output)

    task_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in model_rows:
        task_groups[(row["task"], row["method"], row["context"])].append(row)
    task_summaries: list[dict[str, Any]] = []
    for (task, method, context), group in sorted(
        task_groups.items(),
        key=lambda item: (
            TASK_ORDER[item[0][0]],
            METHOD_ORDER.index(item[0][1]),
            CONTEXT_ORDER[item[0][2]],
        ),
    ):
        scoreable = [
            row
            for row in group
            if row["status"] == "failed"
            or (row["status"] == "succeeded" and row_score_available(row))
        ]
        resolved_scores = [
            float(row["f1"]) if row["status"] == "succeeded" else 0.0 for row in scoreable
        ]
        task_summaries.append(
            {
                "model": model,
                "tier": int(group[0]["tier"]),
                "task": task,
                "task_label": TASK_LABELS[task],
                "method": method,
                "context": context,
                "repetitions": len(group),
                "successful_jobs": sum(row["status"] == "succeeded" for row in group),
                "failed_jobs": sum(row["status"] == "failed" for row in group),
                "scoreable_jobs": len(scoreable),
                "score_coverage": len(scoreable) / len(group),
                "f1_mean": statistics.fmean(resolved_scores) if resolved_scores else None,
                "f1_std": population_std(resolved_scores) if resolved_scores else None,
            }
        )
    return summaries, task_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--source", type=Path, default=Path("paper_plots/gold/iclr2027/full_benchmark_records.csv")
    )
    parser.add_argument(
        "--additional-source",
        action="append",
        type=Path,
        default=[],
        help="Additional canonical record table, such as an x1000 extension.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper_plots/gold/iclr2027/model_appendix")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [args.source, *args.additional_source]
    rows = [row for source in sources for row in read_csv(source)]
    summaries, task_summaries = summarize_model(rows, args.model)
    output = args.output / args.model
    summary_path = output / "tier_metrics.csv"
    task_path = output / "task_f1.csv"
    write_csv(summary_path, summaries)
    write_csv(task_path, task_summaries)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "sources": [{"path": str(source), "sha256": sha256_file(source)} for source in sources],
        "outputs": [
            {"path": path.name, "sha256": sha256_file(path)} for path in (summary_path, task_path)
        ],
        "aggregation": {
            "f1": (
                "question-weighted within each repetition; terminal failures score zero; "
                "pending corrected rescores use the submission-freeze historical value and "
                "retain their internal queue status"
            ),
            "additive_resources": ("sum divided by all successful question trajectories"),
            "peak_memory": "mean combined process-tree plus Docker peak per successful job",
            "variation": "population standard deviation across five repetitions",
        },
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(summaries)} tier rows and {len(task_summaries)} task rows to {output}")


if __name__ == "__main__":
    main()
