#!/usr/bin/env python3
"""Freeze manuscript-facing numbers and checksums for the ICLR submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FREEZE_ID = "rxnhaystack-iclr2027-submission-2026-09-25-v1"
PAID_MODELS = ("gemini-3.7-flash", "gpt-5-mini", "claude-haiku-4.5")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def keyed_rows(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {"/".join(row[key] for key in keys): row for row in rows}


def paid_efficiency_claims(records: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    groups: dict[tuple[int, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        if row["model"] in PAID_MODELS:
            groups[(int(row["tier"]), row["method"], row["context"], row["model"])].append(row)

    by_cell: dict[tuple[int, str, str], list[tuple[float, float]]] = defaultdict(list)
    for (tier, method, context, _model), rows in groups.items():
        terminal = [
            row
            for row in rows
            if row["status"] in {"succeeded", "failed"}
            and row.get("score_correction_status") == "not_affected"
        ]
        expected_trajectories = sum(int(row["question_count"]) for row in terminal)
        successful = [
            row
            for row in terminal
            if row["status"] == "succeeded" and row.get("score_available", "").lower() == "true"
        ]
        successful_trajectories = sum(int(row["question_count"]) for row in successful)
        if not expected_trajectories or not successful_trajectories:
            continue
        score = (
            sum(float(row["f1"]) * int(row["question_count"]) for row in successful)
            / expected_trajectories
        )
        cost = sum(float(row["cost_usd"]) for row in successful) / successful_trajectories
        by_cell[(tier, method, context)].append((score, cost))

    output: dict[str, dict[str, float]] = {}
    for (tier, method, context), values in sorted(by_cell.items()):
        if len(values) != len(PAID_MODELS):
            continue
        output[f"tier{tier}/{method}/x{context}"] = {
            "mean_f1": statistics.fmean(value[0] for value in values),
            "mean_cost_usd_per_successful_trajectory": statistics.fmean(
                value[1] for value in values
            ),
            "n_models": len(values),
        }
    return output


def build_claims(gold_dir: Path) -> dict[str, Any]:
    final_records = read_csv(gold_dir / "final_arm_records.csv")
    full_records = read_csv(gold_dir / "full_benchmark_records.csv")
    scaling = read_csv(gold_dir / "tier_scaling_across_models.csv")
    capability = read_csv(gold_dir / "capability_split/across_models.csv")
    capability_models = read_csv(gold_dir / "capability_split/per_model.csv")
    controls = read_csv(gold_dir / "causal_controls/aggregates.csv")
    prospective = read_csv(gold_dir / "prospective_decomposition/aggregates.csv")
    human = json.loads((gold_dir / "human_validation/summary.json").read_text())
    efficiency = read_csv(gold_dir / "efficiency_appendix/across_models.csv")
    expected_trajectories = sum(int(row["question_count"]) for row in final_records)
    successful_trajectories = sum(
        int(row["question_count"]) for row in final_records if row["status"] == "succeeded"
    )
    return {
        "freeze_id": FREEZE_ID,
        "benchmark": {
            "dataset_reactions": 122456,
            "questions": 100,
            "interfaces": 3,
            "models": len({row["model"] for row in final_records}),
            "terminal_jobs": len(final_records),
            "expected_question_trajectories": expected_trajectories,
            "terminal_question_trajectories": expected_trajectories,
            "successful_question_trajectories": successful_trajectories,
        },
        "core_scaling": keyed_rows(scaling, ("method", "context", "tier")),
        "capability_split_across_models": keyed_rows(capability, ("capability_group",)),
        "capability_split_per_model": keyed_rows(capability_models, ("model", "capability_group")),
        "human_validation": human,
        "causal_controls": keyed_rows(
            controls, ("study", "model_label", "arm", "condition", "context", "tier")
        ),
        "prospective_decomposition": keyed_rows(prospective, ("model_label", "condition")),
        "paid_efficiency_frontier": paid_efficiency_claims(full_records),
        "efficiency_appendix": keyed_rows(efficiency, ("method", "context", "tier")),
    }


def checksums(
    root: Path, patterns: tuple[str, ...], *, exclude: Path | None = None
) -> dict[str, str]:
    paths = {
        path
        for pattern in patterns
        for path in root.rglob(pattern)
        if path.is_file() and (exclude is None or exclude not in path.parents)
    }
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, default=Path("paper_plots/gold/iclr2027"))
    parser.add_argument("--figures-dir", type=Path, default=Path("paper_plots/figures/gold"))
    parser.add_argument("--manuscript-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_plots/gold/iclr2027/submission_freeze"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    claims = build_claims(args.gold_dir)
    claims_path = args.output_dir / "paper_claims.json"
    claims_path.write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n")

    manuscript: dict[str, Any] | None = None
    if args.manuscript_dir is not None:
        manuscript = {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=args.manuscript_dir, text=True
            ).strip(),
            "source_checksums": checksums(
                args.manuscript_dir, ("*.tex", "*.bib", "*.sty", "*.bst")
            ),
            "figure_checksums": checksums(args.manuscript_dir / "figures", ("*.pdf",)),
            "compiled_pdf": (
                {
                    "path": "iclr2027.pdf",
                    "sha256": sha256_file(args.manuscript_dir / "iclr2027.pdf"),
                    "bytes": (args.manuscript_dir / "iclr2027.pdf").stat().st_size,
                }
                if (args.manuscript_dir / "iclr2027.pdf").exists()
                else None
            ),
        }

    manifest = {
        "schema_version": 1,
        "freeze_id": FREEZE_ID,
        "generated_at": datetime.now(UTC).isoformat(),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "claims": {"path": claims_path.name, "sha256": sha256_file(claims_path)},
        "gold_checksums": checksums(
            args.gold_dir, ("*.csv", "*.json", "*.jsonl", "*.md"), exclude=args.output_dir
        ),
        "figure_checksums": checksums(args.figures_dir, ("*.pdf", "*.png", "*.json")),
        "manuscript": manuscript,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "README.md").write_text(
        "# ICLR 2027 submission freeze\n\n"
        f"Freeze ID: `{FREEZE_ID}`.\n\n"
        "`paper_claims.json` freezes manuscript-facing values from the checked gold tables. "
        "`manifest.json` records checksums for every gold table, plot, manuscript source, "
        "manuscript figure, and the compiled PDF. Internal corrected-rescore follow-up is "
        "tracked separately under `../post_submission/`. The exact compiled manuscript is "
        "archived as `iclr2027.pdf`; its checksum and byte count are recorded under "
        "`manuscript.compiled_pdf` in the manifest.\n",
        encoding="utf-8",
    )
    print(
        f"Frozen {claims['benchmark']['expected_question_trajectories']:,} expected "
        f"question trajectories to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
