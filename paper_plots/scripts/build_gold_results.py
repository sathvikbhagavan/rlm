#!/usr/bin/env python3
"""Freeze dashboard results into the canonical ICLR plotting tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import tarfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rxnhaystack.control_room import (
    DEEPSEEK_CODEACT_X1000_CAMPAIGN,
    FULL_CAMPAIGN,
    load_snapshot_directory,
    merge_snapshots,
    scientific_dashboard_view,
)

MODEL_ORDER = (
    "qwen3.5",
    "deepseek-v4-flash",
    "glm-5.2",
    "gemini-3.7-flash",
    "gpt-5-mini",
    "claude-haiku-4.5",
)
MODEL_LABELS = {
    "qwen3.5": "Qwen 3.5",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "glm-5.2": "GLM 5.2",
    "gemini-3.7-flash": "Gemini 3.7 Flash",
    "gpt-5-mini": "GPT-5 mini",
    "claude-haiku-4.5": "Claude Haiku 4.5",
}
MODEL_ALIASES = {
    "RCP-AIaaS/Qwen/Qwen3.5-397B-A17B": "qwen3.5",
    "RCP-AIaaS/deepseek-ai/DeepSeek-V4-Flash-0731": "deepseek-v4-flash",
    "deepseek/deepseek-v4-flash-0731": "deepseek-v4-flash",
    "CSCS-Inference/zai-org/GLM-5.2": "glm-5.2",
    "google/gemini-3.7-flash": "gemini-3.7-flash",
    "openai/gpt-5-mini": "gpt-5-mini",
    "anthropic/claude-haiku-4.5": "claude-haiku-4.5",
}
METHOD_ORDER = ("llm", "codeact", "rlm")
STATUS_ORDER = ("succeeded", "running", "stale", "failed", "pending")
CONTEXT_ORDER = {"100": 0, "500": 1, "1000": 2, "full": 3}
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
    "tier4/task11": 2,
    "tier4/task12": 2,
    "tier4/task12b": 1,
    "tier4/task13": 4,
    "tier4/task14": 2,
    "tier4/task15": 4,
    "tier4/task16": 10,
    "tier4/task17": 5,
    "tier4/task17b": 5,
}
RESOURCE_FIELDS = (
    "peak_combined_memory_mib",
    "peak_docker_memory_mib",
    "peak_host_process_tree_rss_mib",
    "process_wall_time_seconds",
)
METRIC_FIELDS = (
    "calls",
    "cost_chf",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_seconds",
    "tool_time_seconds",
)

DEFAULT_GEMINI_X1000_PACK = Path(
    "paper_plots/gold/source_packs/gemini-codeact-x1000-succeeded-pack.tgz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_context(value: Any) -> str:
    return "full" if str(value) == "full" else str(int(value))


def tier_for_task(task: str) -> int:
    return int(task.removeprefix("tier").split("/", 1)[0])


def successful_attempt(run: dict[str, Any]) -> dict[str, Any] | None:
    successful = [
        attempt for attempt in run.get("attempts", ()) if attempt["status"] == "succeeded"
    ]
    return max(successful, key=lambda item: item["started_at"], default=None)


def result_score(task: str, metrics: dict[str, Any]) -> tuple[str, float | None]:
    score_name = "macro_reaction_f1" if task == "tier4/task15" else "macro_f1"
    value = (metrics.get("results") or {}).get(score_name)
    return score_name, None if value is None else float(value)


def model_slug(model: str) -> str:
    if model in MODEL_ORDER:
        return model
    try:
        return MODEL_ALIASES[model]
    except KeyError as error:
        raise ValueError(f"Unrecognized gold-export model {model!r}") from error


def flatten_run(run: dict[str, Any], *, scope: str) -> dict[str, Any]:
    task = str(run["task"])
    attempt = successful_attempt(run)
    metrics = {} if attempt is None else attempt.get("metrics") or {}
    score_name, f1 = result_score(task, metrics)
    resources = metrics.get("resources") or {}
    slug = model_slug(str(run["model"]))
    row: dict[str, Any] = {
        "scope": scope,
        "run_id": run["run_id"],
        "model": slug,
        "model_label": MODEL_LABELS[slug],
        "method": run["method"],
        "tier": tier_for_task(task),
        "task": task,
        "context": normalize_context(run["corpus_size"]),
        "repetition": int(run["repetition"]),
        "question_count": QUESTION_COUNTS[task],
        "status": run["status"],
        "report_state": run.get("report_state", ""),
        "result_updated_at": run.get("result_updated_at") or "",
        "attempt_count": int(run.get("attempt_count", 0)),
        "sources": ";".join(run.get("sources", ())),
        "failure_categories": json.dumps(run.get("failure_categories", {}), sort_keys=True),
        "score_name": score_name,
        "score_available": f1 is not None,
        "f1": f1,
    }
    row.update({field: metrics.get(field) for field in METRIC_FIELDS})
    row.update({field: resources.get(field) for field in RESOURCE_FIELDS})
    return row


def flatten_packed_run(run: dict[str, Any], *, packed_at: str, pack_name: str) -> dict[str, Any]:
    """Convert one sanitized external result-pack row to the gold schema."""
    tier = int(run["tier"])
    task = f"tier{tier}/task{run['task']}"
    score_name = "macro_reaction_f1" if task == "tier4/task15" else "macro_f1"
    f1 = run.get(f"metrics.results.{score_name}")
    slug = model_slug(str(run["model"]))
    row: dict[str, Any] = {
        "scope": "codeact_x1000",
        "run_id": run["run_id"],
        "model": slug,
        "model_label": MODEL_LABELS[slug],
        "method": run["method"],
        "tier": tier,
        "task": task,
        "context": normalize_context(run["context"]),
        "repetition": int(run["repetition"]),
        "question_count": QUESTION_COUNTS[task],
        "status": run["status"],
        "report_state": "external-succeeded-result-pack",
        "result_updated_at": packed_at,
        "attempt_count": int(run["attempt"]),
        "sources": pack_name,
        "failure_categories": "{}",
        "score_name": score_name,
        "score_available": f1 is not None,
        "f1": None if f1 is None else float(f1),
    }
    row.update({field: run.get(f"metrics.{field}") for field in METRIC_FIELDS})
    row.update({field: run.get(f"metrics.resources.{field}") for field in RESOURCE_FIELDS})
    return row


def load_result_pack(
    path: Path, *, expected_model: str, expected_run_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read and strictly validate a sanitized result pack without extracting it."""
    with tarfile.open(path, "r:gz") as archive:
        manifest_members = [
            member for member in archive.getmembers() if member.name.endswith("/manifest.json")
        ]
        run_members = [
            member for member in archive.getmembers() if member.name.endswith("/runs.json")
        ]
        if len(manifest_members) != 1 or len(run_members) != 1:
            raise ValueError(f"{path} must contain exactly one manifest.json and runs.json")
        manifest_stream = archive.extractfile(manifest_members[0])
        runs_stream = archive.extractfile(run_members[0])
        if manifest_stream is None or runs_stream is None:
            raise ValueError(f"Could not read required members from {path}")
        manifest = json.load(manifest_stream)
        packed_runs = json.load(runs_stream)

    if manifest.get("models") != [expected_model]:
        raise ValueError(f"Unexpected models in {path}: {manifest.get('models')!r}")
    if manifest.get("n_runs") != len(packed_runs):
        raise ValueError(f"Pack manifest cardinality does not match runs.json in {path}")
    actual_run_ids = {str(run["run_id"]) for run in packed_runs}
    if len(actual_run_ids) != len(packed_runs):
        raise ValueError(f"Duplicate run IDs in {path}")
    if actual_run_ids != expected_run_ids:
        missing = sorted(expected_run_ids - actual_run_ids)
        unexpected = sorted(actual_run_ids - expected_run_ids)
        raise ValueError(
            f"Unexpected run set in {path}: missing={missing[:3]!r}, unexpected={unexpected[:3]!r}"
        )
    for run in packed_runs:
        if (
            run.get("model") != expected_model
            or run.get("method") != "codeact"
            or normalize_context(run.get("context")) != "1000"
            or run.get("status") != "succeeded"
        ):
            raise ValueError(f"Unexpected result-pack row {run.get('run_id')!r} in {path}")

    rows = [
        flatten_packed_run(
            run,
            packed_at=str(manifest["packed_at"]),
            pack_name=path.name,
        )
        for run in packed_runs
    ]
    unscored = [row["run_id"] for row in rows if not row["score_available"]]
    if unscored:
        raise ValueError(f"Result pack {path} has runs without plot scores: {unscored[:3]!r}")
    return rows, manifest


def arm_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scope"]), str(row["model"]), str(row["method"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (scope, model, method), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            MODEL_ORDER.index(item[0][1]),
            METHOD_ORDER.index(item[0][2]),
        ),
    ):
        counts = Counter(str(row["status"]) for row in group)
        unscored_successes = sum(
            row["status"] == "succeeded" and not bool(row["score_available"]) for row in group
        )
        unfinished = counts["running"] + counts["stale"] + counts["pending"] + unscored_successes
        is_final = unfinished == 0
        summaries.append(
            {
                "scope": scope,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "method": method,
                "expected_jobs": len(group),
                **{f"{status}_jobs": counts[status] for status in STATUS_ORDER},
                "scored_success_jobs": counts["succeeded"] - unscored_successes,
                "unscored_success_jobs": unscored_successes,
                "is_final": is_final,
                "legend_label": method.upper() if method == "llm" else method.capitalize(),
                "note": "terminal and scored"
                if is_final
                else f"{unfinished} running, stale, pending, or unscored",
            }
        )
    return summaries


def add_arm_finality(rows: list[dict[str, Any]], arms: list[dict[str, Any]]) -> None:
    finality = {
        (str(arm["scope"]), str(arm["model"]), str(arm["method"])): bool(arm["is_final"])
        for arm in arms
    }
    for row in rows:
        row["arm_final"] = finality[(str(row["scope"]), str(row["model"]), str(row["method"]))]


def scaling_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["model"]), str(row["method"]), str(row["context"]), int(row["tier"]))
        ].append(row)
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
            row for row in group if row["status"] == "succeeded" and bool(row["score_available"])
        ]
        expected_weight = sum(int(row["question_count"]) for row in group)
        successful_weight = sum(int(row["question_count"]) for row in successful)
        weighted_score = sum(float(row["f1"]) * int(row["question_count"]) for row in successful)
        f1 = weighted_score / successful_weight if successful_weight else None
        repetition_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in successful:
            repetition_groups[int(row["repetition"])].append(row)
        repetition_scores = []
        for repetition_rows in repetition_groups.values():
            weight = sum(int(row["question_count"]) for row in repetition_rows)
            repetition_scores.append(
                sum(float(row["f1"]) * int(row["question_count"]) for row in repetition_rows)
                / weight
            )
        f1_std = statistics.pstdev(repetition_scores) if repetition_scores else None
        counts = Counter(str(row["status"]) for row in group)
        output.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "method": method,
                "context": context,
                "tier": tier,
                "expected_jobs": len(group),
                **{f"{status}_jobs": counts[status] for status in STATUS_ORDER},
                "expected_trajectories": expected_weight,
                "successful_trajectories": successful_weight,
                "coverage": successful_weight / expected_weight,
                "successful_repetitions": len(repetition_scores),
                "f1": f1,
                "f1_std": f1_std,
                "f1_zero_imputed": weighted_score / expected_weight,
                "f1_best_case": (weighted_score + expected_weight - successful_weight)
                / expected_weight,
                "arm_final": all(bool(row["arm_final"]) for row in group),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty gold table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_readme(path: Path, arms: list[dict[str, Any]], *, as_of: str) -> None:
    lines = [
        "# Gold ICLR 2027 plotting results",
        "",
        f"Frozen from the shared experiment dashboard at `{as_of}`.",
        "",
        "Gold means that the plotting input is frozen, auditable, and provenance-recorded. "
        "It does not mean every experiment arm is finished. `arm_status.csv` and the "
        "`arm_final` columns distinguish terminal arms from provisional ones.",
        "",
        "The CSV files contain sanitized metrics sufficient to regenerate paper plots; "
        "bulky raw trajectories remain in their original experiment artifact stores.",
        "",
        "## Main benchmark arms",
        "",
        "| Model | LLM | CodeAct | RLM |",
        "| --- | ---: | ---: | ---: |",
    ]
    by_arm = {
        (str(row["model"]), str(row["method"])): row
        for row in arms
        if row["scope"] == "full_benchmark"
    }
    for model in MODEL_ORDER:
        cells = []
        for method in METHOD_ORDER:
            arm = by_arm[(model, method)]
            label = "final" if arm["is_final"] else "provisional"
            cells.append(f"{arm['succeeded_jobs']}/{arm['expected_jobs']} {label}")
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `full_benchmark_records.csv`: every one of the 6,300 expected main-benchmark jobs.",
            "- `codeact_x1000_records.csv`: the final DeepSeek and Gemini CodeAct x1000 extensions.",
            "- `final_arm_records.csv`: records belonging to terminal arms.",
            "- `provisional_arm_records.csv`: records belonging to unfinished arms.",
            "- `arm_status.csv`: the finality decision used for legend asterisks.",
            "- `tier_scaling.csv`: the faithful four-tier plotting aggregate.",
            "- `source_manifest.json`: source snapshot and file checksums.",
            "",
            "Regenerate from the repository root:",
            "",
            "```bash",
            "uv run --frozen python paper_plots/scripts/build_gold_results.py",
            "uv run --with-requirements paper_plots/requirements.txt \\",
            "  python paper_plots/scripts/plot_gold_scaling_by_tier.py",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=Path("artifacts/control-room/shared"))
    parser.add_argument("--output", type=Path, default=Path("paper_plots/gold/iclr2027"))
    parser.add_argument(
        "--gemini-x1000-pack",
        type=Path,
        default=DEFAULT_GEMINI_X1000_PACK,
        help="Sanitized Gemini CodeAct x1000 result pack.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots = load_snapshot_directory(args.snapshot_dir)
    merged = scientific_dashboard_view(merge_snapshots(snapshots))
    campaigns = {str(campaign["name"]): campaign for campaign in merged["campaigns"]}
    full = campaigns[FULL_CAMPAIGN]
    x1000 = campaigns[DEEPSEEK_CODEACT_X1000_CAMPAIGN]
    rows = [flatten_run(run, scope="full_benchmark") for run in full["runs"]]
    deepseek_extension_rows = [flatten_run(run, scope="codeact_x1000") for run in x1000["runs"]]
    expected_gemini_ids = {
        str(row["run_id"]).replace("-x500-", "-x1000-")
        for row in rows
        if row["model"] == "gemini-3.7-flash"
        and row["method"] == "codeact"
        and row["context"] == "500"
    }
    gemini_extension_rows, gemini_pack_manifest = load_result_pack(
        args.gemini_x1000_pack,
        expected_model="gemini-3.7-flash",
        expected_run_ids=expected_gemini_ids,
    )
    extension_rows = deepseek_extension_rows + gemini_extension_rows
    all_rows = rows + extension_rows
    arms = arm_summaries(all_rows)
    add_arm_finality(all_rows, arms)
    scaling = scaling_summaries(all_rows)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "full_benchmark_records.csv", rows)
    write_csv(output / "codeact_x1000_records.csv", extension_rows)
    write_csv(output / "final_arm_records.csv", [row for row in all_rows if row["arm_final"]])
    write_csv(
        output / "provisional_arm_records.csv", [row for row in all_rows if not row["arm_final"]]
    )
    write_csv(output / "arm_status.csv", arms)
    write_csv(output / "tier_scaling.csv", scaling)

    as_of = max(str(snapshot["generated_at"]) for snapshot in snapshots)
    write_readme(output / "README.md", arms, as_of=as_of)
    generated_files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "source_manifest.json"
    )
    source_ids = {str(source["id"]) for source in full["sources"] + x1000["sources"]}
    latest_by_source = {str(snapshot["source"]["id"]): snapshot for snapshot in snapshots}
    snapshot_paths = {
        str(load["source"]["id"]): path
        for path in args.snapshot_dir.glob("*.json")
        if (load := json.loads(path.read_text()))["source"]["id"] in source_ids
    }
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of": as_of,
        "repository_commit": git_commit(Path.cwd()),
        "campaigns": {
            FULL_CAMPAIGN: {
                "definition_sha256": full["definition_sha256"],
                "counts": full["counts"],
            },
            DEEPSEEK_CODEACT_X1000_CAMPAIGN: {
                "definition_sha256": x1000["definition_sha256"],
                "counts": x1000["counts"],
            },
        },
        "result_packs": [
            {
                "path": str(args.gemini_x1000_pack),
                "sha256": sha256_file(args.gemini_x1000_pack),
                "bytes": args.gemini_x1000_pack.stat().st_size,
                "packed_at": gemini_pack_manifest["packed_at"],
                "models": gemini_pack_manifest["models"],
                "n_runs": gemini_pack_manifest["n_runs"],
                "rule": gemini_pack_manifest["rule"],
            }
        ],
        "source_snapshots": [
            {
                "source_id": source_id,
                "generated_at": latest_by_source[source_id]["generated_at"],
                "snapshot_id": latest_by_source[source_id]["snapshot_id"],
                "path": str(snapshot_paths[source_id]),
                "sha256": sha256_file(snapshot_paths[source_id]),
            }
            for source_id in sorted(source_ids)
            if source_id in latest_by_source and source_id in snapshot_paths
        ],
        "gold_files": [
            {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in generated_files
        ],
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if len(rows) != 6300 or len(deepseek_extension_rows) != 150 or len(extension_rows) != 300:
        raise ValueError(
            "Gold export cardinality mismatch: "
            f"{len(rows)} main, {len(deepseek_extension_rows)} DeepSeek x1000, "
            f"and {len(gemini_extension_rows)} Gemini x1000"
        )
    print(f"Wrote {len(rows):,} main and {len(extension_rows):,} x1000 records to {output}")


if __name__ == "__main__":
    main()
