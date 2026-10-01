#!/usr/bin/env python3
"""Build the multi-model full-corpus capability-split table."""

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

MODEL_ORDER = (
    "qwen3.5",
    "deepseek-v4-flash",
    "glm-5.2",
    "gemini-3.7-flash",
    "gpt-5-mini",
    "claude-haiku-4.5",
)
CAPABILITY_GROUPS = {
    "bond_changes": ("tier3/task13", "tier3/task14", "tier3/task15"),
    "stereochemistry": ("tier3/task23", "tier3/task24"),
    "mechanisms": ("tier3/task10", "tier3/task10b"),
    "mechanical_graph": ("tier4/task11", "tier4/task12", "tier4/task12b"),
    "chemically_constrained_graph": ("tier4/task13", "tier4/task14", "tier4/task15"),
    "route_and_multi_constraint": ("tier4/task16", "tier4/task17", "tier4/task17b"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty capability table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def capability_summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [
        row
        for row in rows
        if row["method"] == "rlm"
        and str(row["context"]) == "full"
        and as_bool(row.get("arm_final"))
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        for group, tasks in CAPABILITY_GROUPS.items():
            if row["task"] in tasks:
                grouped[(str(row["model"]), group)].append(row)
                break

    per_model: list[dict[str, Any]] = []
    for (model, group), members in sorted(
        grouped.items(),
        key=lambda item: (
            MODEL_ORDER.index(item[0][0]),
            tuple(CAPABILITY_GROUPS).index(item[0][1]),
        ),
    ):
        numerator = 0.0
        denominator = 0
        failures = 0
        for row in members:
            # A terminal failure has no model answer to rescore, but it remains a
            # resolved benchmark trajectory and therefore contributes zero.  The
            # score-availability gate applies only to successful runs whose
            # historical prediction could not be scored under the frozen evaluator.
            if (
                row["status"] == "succeeded"
                and "score_available" in row
                and not as_bool(row.get("score_available"))
            ):
                continue
            weight = int(row["question_count"])
            denominator += weight
            if row["status"] == "succeeded" and row.get("f1") not in {None, ""}:
                numerator += float(row["f1"]) * weight
            else:
                failures += 1
        if denominator == 0:
            continue
        per_model.append(
            {
                "model": model,
                "capability_group": group,
                "mean_f1": numerator / denominator,
                "weighted_question_runs": denominator,
                "terminal_failed_jobs": failures,
            }
        )

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_model:
        by_group[str(row["capability_group"])].append(row)
    across_models: list[dict[str, Any]] = []
    for group in CAPABILITY_GROUPS:
        members = by_group[group]
        if not members:
            continue
        values = [float(row["mean_f1"]) for row in members]
        across_models.append(
            {
                "capability_group": group,
                "across_model_mean_f1": statistics.fmean(values),
                "minimum_model_f1": min(values),
                "maximum_model_f1": max(values),
                "n_models": len(values),
                "included_models": ";".join(
                    model for model in MODEL_ORDER if any(row["model"] == model for row in members)
                ),
            }
        )
    return per_model, across_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("paper_plots/gold/iclr2027"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_plots/gold/iclr2027/capability_split"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.gold / "full_benchmark_records.csv"
    parent_manifest = args.gold / "source_manifest.json"
    per_model, across_models = capability_summaries(read_csv(source))
    args.output.mkdir(parents=True, exist_ok=True)
    per_model_path = args.output / "per_model.csv"
    across_path = args.output / "across_models.csv"
    write_csv(per_model_path, per_model)
    write_csv(across_path, across_models)
    parent = json.loads(parent_manifest.read_text())
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_snapshot_as_of": parent["as_of"],
        "source_repository_commit": parent["repository_commit"],
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "outputs": [
            {"path": path.name, "sha256": sha256_file(path)}
            for path in (per_model_path, across_path)
        ],
        "aggregation": {
            "scope": "terminal full-corpus RLM arms",
            "within_model": "question-weighted mean across task scripts and repetitions",
            "terminal_failures": "score zero",
            "corrected_tasks": (
                "exact corrected rescores where recoverable; otherwise the explicit "
                "submission-freeze historical value"
            ),
            "across_models": "unweighted mean and observed model range",
        },
    }
    (args.output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(per_model)} per-model and {len(across_models)} group rows")


if __name__ == "__main__":
    main()
