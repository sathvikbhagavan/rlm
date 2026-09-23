#!/usr/bin/env python3
"""Build frozen efficiency tables from the gold ICLR benchmark records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METHOD_ORDER = ("llm", "codeact", "rlm")
MODEL_ORDER = (
    "qwen3.5",
    "deepseek-v4-flash",
    "glm-5.2",
    "gemini-3.7-flash",
    "gpt-5-mini",
    "claude-haiku-4.5",
)
CONTEXT_ORDER = {"100": 0, "500": 1, "full": 2}

PER_TRAJECTORY_METRICS = {
    "calls_per_trajectory": "calls",
    "tokens_per_trajectory": "total_tokens",
    "latency_seconds_per_trajectory": "latency_seconds",
    "tool_time_seconds_per_trajectory": "tool_time_seconds",
    "wall_time_seconds_per_trajectory": "process_wall_time_seconds",
}
PER_JOB_METRICS = {"peak_memory_mib_per_job": "peak_combined_memory_mib"}
ALL_METRICS = (*PER_TRAJECTORY_METRICS, *PER_JOB_METRICS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty efficiency table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def per_model_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize additive metrics per trajectory and peak memory per successful job."""
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not as_bool(row.get("arm_final")):
            continue
        context = str(row["context"])
        if context not in CONTEXT_ORDER:
            continue
        grouped[(str(row["model"]), str(row["method"]), context, int(row["tier"]))].append(row)

    output: list[dict[str, Any]] = []
    for (model, method, context, tier), group in sorted(
        grouped.items(),
        key=lambda item: (
            MODEL_ORDER.index(item[0][0]),
            item[0][3],
            METHOD_ORDER.index(item[0][1]),
            CONTEXT_ORDER[item[0][2]],
        ),
    ):
        successful = [
            row
            for row in group
            if row["status"] == "succeeded" and as_bool(row.get("score_available"))
        ]
        successful_trajectories = sum(int(row["question_count"]) for row in successful)
        row_out: dict[str, Any] = {
            "model": model,
            "method": method,
            "context": context,
            "tier": tier,
            "successful_jobs": len(successful),
            "successful_trajectories": successful_trajectories,
            "failed_jobs": sum(row["status"] == "failed" for row in group),
        }
        for output_name, source_name in PER_TRAJECTORY_METRICS.items():
            measured = [row for row in successful if as_float(row.get(source_name)) is not None]
            measured_trajectories = sum(int(row["question_count"]) for row in measured)
            row_out[output_name] = (
                sum(float(row[source_name]) for row in measured) / measured_trajectories
                if measured_trajectories
                else None
            )
            row_out[f"{output_name}_coverage"] = (
                measured_trajectories / successful_trajectories if successful_trajectories else 0.0
            )
        for output_name, source_name in PER_JOB_METRICS.items():
            measured = [row for row in successful if as_float(row.get(source_name)) is not None]
            row_out[output_name] = (
                statistics.fmean(float(row[source_name]) for row in measured) if measured else None
            )
            row_out[f"{output_name}_coverage"] = (
                len(measured) / len(successful) if successful else 0.0
            )
        output.append(row_out)
    return output


def across_model_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average per-model efficiency without weighting large models or tasks more heavily."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["context"]), int(row["tier"]))].append(row)

    output: list[dict[str, Any]] = []
    for (method, context, tier), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][2],
            METHOD_ORDER.index(item[0][0]),
            CONTEXT_ORDER[item[0][1]],
        ),
    ):
        models = {str(row["model"]) for row in group}
        row_out: dict[str, Any] = {
            "method": method,
            "context": context,
            "tier": tier,
            "n_models": len(models),
            "included_models": ";".join(model for model in MODEL_ORDER if model in models),
            "successful_jobs": sum(int(row["successful_jobs"]) for row in group),
            "failed_jobs": sum(int(row["failed_jobs"]) for row in group),
        }
        for metric in ALL_METRICS:
            measured = [row for row in group if as_float(row.get(metric)) is not None]
            values = [float(row[metric]) for row in measured]
            standard_deviation = statistics.stdev(values) if len(values) > 1 else None
            row_out[f"mean_{metric}"] = statistics.fmean(values) if values else None
            row_out[f"sem_{metric}"] = (
                standard_deviation / math.sqrt(len(values))
                if standard_deviation is not None
                else None
            )
            row_out[f"minimum_{metric}_coverage"] = min(
                (float(row[f"{metric}_coverage"]) for row in measured), default=0.0
            )
            row_out[f"n_{metric}_models"] = len(measured)
        output.append(row_out)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("paper_plots/gold/iclr2027"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_plots/gold/iclr2027/efficiency_appendix"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.gold / "full_benchmark_records.csv"
    parent_manifest = args.gold / "source_manifest.json"
    model_rows = per_model_summaries(read_csv(source))
    across_rows = across_model_summaries(model_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    per_model_path = args.output / "per_model.csv"
    across_path = args.output / "across_models.csv"
    write_csv(per_model_path, model_rows)
    write_csv(across_path, across_rows)

    parent = json.loads(parent_manifest.read_text())
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_snapshot_as_of": parent["as_of"],
        "source_repository_commit": parent["repository_commit"],
        "sources": [
            {
                "path": str(source),
                "sha256": sha256_file(source),
            },
            {
                "path": str(parent_manifest),
                "sha256": sha256_file(parent_manifest),
            },
        ],
        "outputs": [
            {"path": path.name, "sha256": sha256_file(path)}
            for path in (per_model_path, across_path)
        ],
        "aggregation": {
            "included": "successful scored jobs from terminal model/method arms",
            "additive_metrics": "sum per job divided by successful question trajectories",
            "peak_memory": "mean peak combined host-process-tree plus Docker memory per job",
            "cross_model": "unweighted mean and SEM across terminal model arms",
            "failed_jobs": "excluded from resource means and counted separately",
        },
    }
    (args.output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(model_rows)} per-model and {len(across_rows)} cross-model rows")


if __name__ == "__main__":
    main()
