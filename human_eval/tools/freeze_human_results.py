"""Freeze checksum-verified human exports into paper-ready derived tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from human_eval.analysis import analyze, load_export, mean, write_csv


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def parse_export(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("exports must be LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="append", type=parse_export, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model-gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    labels_by_id: dict[str, str] = {}
    inputs: list[dict[str, object]] = []
    paths: list[Path] = []
    for label, path in args.export:
        manifest, _annotations, _timing = load_export(path)
        reviewer_id = str(manifest["anonymous_annotator_id"])
        if reviewer_id in labels_by_id or label in labels_by_id.values():
            raise ValueError(f"Duplicate reviewer label or ID: {label}")
        labels_by_id[reviewer_id] = label
        paths.append(path)
        inputs.append(
            {
                "reviewer": label,
                "anonymous_annotator_id": reviewer_id,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "original_bundle": manifest.get("bundle"),
                "exported_at": manifest.get("exported_at"),
            }
        )

    analyze(
        paths,
        args.bundle / "questions.jsonl",
        args.bundle / "admin/ground_truth.jsonl",
        args.output,
    )
    rows = read_csv(args.output / "item_metrics.csv")
    for row in rows:
        row["annotator_id"] = labels_by_id[row["annotator_id"]]
    write_csv(args.output / "item_metrics.csv", rows)
    pairwise = read_csv(args.output / "baseline_pairwise_agreement.csv")
    for row in pairwise:
        row["annotator_a"] = labels_by_id[row["annotator_a"]]
        row["annotator_b"] = labels_by_id[row["annotator_b"]]
    write_csv(args.output / "baseline_pairwise_agreement.csv", pairwise)
    summary_path = args.output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for record in summary.get("agreement", {}).get("baseline", {}).get("pairwise", []):
        record["annotator_a"] = labels_by_id[record["annotator_a"]]
        record["annotator_b"] = labels_by_id[record["annotator_b"]]
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    annotator_rows: list[dict[str, object]] = []
    for _reviewer_id, label in sorted(labels_by_id.items(), key=lambda item: item[1]):
        reviewer_rows = [row for row in rows if row["annotator_id"] == label]
        submitted = [row for row in reviewer_rows if row["submitted"] == "True"]
        scored = [row for row in submitted if row.get("exact_match", "") != ""]
        source = next(item for item in inputs if item["reviewer"] == label)
        _, annotations, _ = load_export(
            next(path for path in paths if file_sha256(path) == source["sha256"])
        )
        tools: Counter[str] = Counter()
        for annotation in annotations:
            if not annotation.get("submitted_at"):
                continue
            payload = annotation.get("first_submission_payload") or annotation["payload"]
            tools.update(str(tool) for tool in payload.get("tools", []))
        annotator_rows.append(
            {
                "reviewer": label,
                "submitted_items": len(submitted),
                "scored_nonabstained_items": len(scored),
                "abstentions": sum(bool(row.get("abstention")) for row in submitted),
                "exact_match_accuracy": mean([float(row["exact_match"]) for row in scored]),
                "macro_precision": mean([float(row["precision"]) for row in scored]),
                "macro_recall": mean([float(row["recall"]) for row in scored]),
                "macro_f1": mean([float(row["f1"]) for row in scored]),
                "active_hours": sum(float(row["active"] or 0) for row in submitted) / 3600,
                "wall_hours": sum(float(row["wall"] or 0) for row in submitted) / 3600,
                "self_reported_offline_hours": sum(
                    float(row["offline_minutes"] or 0) for row in submitted
                )
                / 60,
                "mean_confidence_1_to_5": mean(
                    [float(row["confidence"]) for row in submitted if row.get("confidence")]
                ),
                "tool_use_counts_json": json.dumps(dict(sorted(tools.items())), sort_keys=True),
                "source_export_sha256": source["sha256"],
            }
        )
    write_csv(args.output / "annotator_summary.csv", annotator_rows)

    model_rows = read_csv(args.model_gold / "full_benchmark_records.csv")
    human_by_tier: dict[int, list[float]] = {}
    for tier in range(1, 5):
        human_by_tier[tier] = [
            float(row["f1"])
            for row in rows
            if row.get("tier") == str(tier) and row.get("f1", "") != ""
        ]
    comparison: list[dict[str, object]] = []
    for tier in range(1, 5):
        comparison.append(
            {
                "population": "human_assigned_items",
                "tier": tier,
                "observations": len(human_by_tier[tier]),
                "mean_f1": mean(human_by_tier[tier]),
                "score_coverage": 1.0,
                "unit": "first submitted non-abstained question answer",
            }
        )
        for method in ("llm", "codeact", "rlm"):
            eligible = [
                row
                for row in model_rows
                if row["tier"] == str(tier)
                and row["method"] == method
                and row["status"] == "succeeded"
            ]
            scored_model = [row for row in eligible if row["score_available"] == "True"]
            comparison.append(
                {
                    "population": f"model_{method}",
                    "tier": tier,
                    "observations": len(scored_model),
                    "mean_f1": mean([float(row["f1"]) for row in scored_model]),
                    "score_coverage": len(scored_model) / len(eligible) if eligible else None,
                    "unit": "successful task-context-repetition run",
                }
            )
    write_csv(args.output / "human_model_tier_comparison.csv", comparison)
    feedback_rows = read_csv(Path("human_eval/REVIEWER_FEEDBACK_AUDIT.csv"))
    write_csv(args.output / "reviewer_feedback_audit.csv", feedback_rows)

    bundle_manifest = json.loads((args.bundle / "manifest.json").read_text())
    manifest = {
        "schema_version": 1,
        "analysis_rule": (
            "First submitted non-abstained baseline answer; semantic set scoring; "
            "post-reference revisions excluded from accuracy. Timing components are "
            "reported separately and are not summed because offline time may overlap wall time."
        ),
        "corrected_bundle": {
            "path": str(args.bundle),
            "bundle_version": bundle_manifest["bundle_version"],
            "questions_sha256": bundle_manifest["questions_sha256"],
            "admin_ground_truth_sha256": bundle_manifest["admin_ground_truth_sha256"],
            "dataset_sha256": bundle_manifest["dataset_sha256"],
        },
        "inputs": inputs,
        "model_comparison_source": {
            "path": str(args.model_gold / "full_benchmark_records.csv"),
            "sha256": file_sha256(args.model_gold / "full_benchmark_records.csv"),
            "note": "Corrected historical scores marked unavailable are excluded; coverage is explicit.",
        },
        "files": {},
    }
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "input_manifest.json":
            manifest["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    (args.output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"reviewers": len(paths), "submitted": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
