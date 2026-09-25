#!/usr/bin/env python3
"""Freeze dashboard results into the canonical ICLR plotting tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from rxnhaystack.score_recovery import apply_corrected_score_recoveries

MODEL_ORDER = (
    "qwen3.5",
    "deepseek-v4-flash",
    "glm-5.2",
    "gemini-3.7-flash",
    "gpt-5-mini",
    "claude-haiku-4.5",
)
PAID_MODELS = frozenset({"gemini-3.7-flash", "gpt-5-mini", "claude-haiku-4.5"})
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
    "qwen3.5-397b": "qwen3.5",
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
EXPECTED_MODELS_BY_METHOD_CONTEXT = {
    ("llm", "100"): frozenset(MODEL_ORDER),
    ("llm", "500"): frozenset(MODEL_ORDER),
    ("codeact", "100"): frozenset(MODEL_ORDER),
    ("codeact", "500"): frozenset(MODEL_ORDER),
    ("codeact", "1000"): frozenset(
        {"qwen3.5", "deepseek-v4-flash", "gemini-3.7-flash", "gpt-5-mini"}
    ),
    ("rlm", "100"): frozenset(MODEL_ORDER),
    ("rlm", "500"): frozenset(MODEL_ORDER),
    ("rlm", "1000"): frozenset({"deepseek-v4-flash", "gemini-3.7-flash", "gpt-5-mini"}),
    ("rlm", "full"): frozenset(MODEL_ORDER),
}
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
DEFAULT_QWEN_X1000_PACK = Path(
    "paper_plots/gold/source_packs/qwen-codeact-x1000-succeeded-pack.tgz"
)
DEFAULT_DEEPSEEK_RLM_X1000_PACK = Path(
    "paper_plots/gold/source_packs/deepseek-rlm-x1000-docker-succeeded-pack.tgz"
)
DEFAULT_SCORE_RECOVERIES = Path("paper_plots/gold/source_packs/deepseek-score-recoveries.json")
DEFAULT_GROUND_TRUTH_CORRECTIONS = Path("paper_plots/gold/ground_truth_corrections.json")
DEFAULT_CORRECTED_SCORE_RECOVERIES = Path("paper_plots/gold/corrected_score_recoveries.json")
GPT_RLM_X1000_DOCKER_CAMPAIGN = "iclr2027-gpt5mini-rlm-x1000-docker-v1"


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


def flatten_packed_run(
    run: dict[str, Any], *, packed_at: str, pack_name: str, scope: str = "codeact_x1000"
) -> dict[str, Any]:
    """Convert one sanitized external result-pack row to the gold schema."""
    tier = int(run["tier"])
    task = f"tier{tier}/task{run['task']}"
    score_name = "macro_reaction_f1" if task == "tier4/task15" else "macro_f1"
    f1 = run.get(f"metrics.results.{score_name}")
    slug = model_slug(str(run["model"]))
    row: dict[str, Any] = {
        "scope": scope,
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
    path: Path,
    *,
    expected_model: str,
    expected_run_ids: set[str],
    expected_method: str = "codeact",
    scope: str = "codeact_x1000",
    canonical_run_id_prefix: str = "",
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
            or run.get("method") != expected_method
            or normalize_context(run.get("context")) != "1000"
            or run.get("status") != "succeeded"
        ):
            raise ValueError(f"Unexpected result-pack row {run.get('run_id')!r} in {path}")

    rows = [
        flatten_packed_run(
            run,
            packed_at=str(manifest["packed_at"]),
            pack_name=path.name,
            scope=scope,
        )
        for run in packed_runs
    ]
    if canonical_run_id_prefix:
        for row in rows:
            row["run_id"] = canonical_run_id_prefix + str(row["run_id"])
    unscored = [row["run_id"] for row in rows if not row["score_available"]]
    if unscored:
        raise ValueError(f"Result pack {path} has runs without plot scores: {unscored[:3]!r}")
    return rows, manifest


def apply_score_recoveries(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Apply narrowly scoped, provenance-recorded recoveries for legacy metrics."""
    payload = json.loads(path.read_text())
    by_id = {str(row["run_id"]): row for row in rows}
    seen: set[str] = set()
    for recovery in payload.get("recoveries", ()):
        run_id = str(recovery["run_id"])
        if run_id in seen or run_id not in by_id:
            raise ValueError(f"Invalid or duplicate score recovery {run_id!r}")
        seen.add(run_id)
        row = by_id[run_id]
        if row["status"] != "succeeded" or row["score_available"]:
            raise ValueError(f"Score recovery does not target an unscored success: {run_id}")
        if row["score_name"] != recovery["score_name"]:
            raise ValueError(f"Score recovery metric mismatch for {run_id}")
        row["f1"] = float(recovery["score_value"])
        row["score_available"] = True
        row["sources"] = f"{row['sources']};score-recovery:{path.name}"
    return payload


def apply_ground_truth_corrections(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Invalidate stale scores without altering immutable model-run evidence."""
    payload = json.loads(path.read_text())
    affected = {str(task) for task in payload["affected_tasks"]}
    for row in rows:
        row["ground_truth_version"] = str(payload["corrected_bundle"])
        row["original_f1"] = row.get("f1")
        if str(row["task"]) not in affected:
            row["score_correction_status"] = "not_affected"
            continue
        if row["status"] == "succeeded" and bool(row["score_available"]):
            row["score_available"] = False
            row["f1"] = None
            row["score_correction_status"] = "historical_score_invalidated"
            row["sources"] = f"{row['sources']};ground-truth:{payload['correction_id']}"
        else:
            row["score_correction_status"] = "affected_without_historical_score"
    return payload


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
        invalidated_successes = sum(
            row["status"] == "succeeded"
            and row.get("score_correction_status") == "historical_score_invalidated"
            for row in group
        )
        unscored_successes = sum(
            row["status"] == "succeeded"
            and not bool(row["score_available"])
            and row.get("score_correction_status") != "historical_score_invalidated"
            for row in group
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
                "scored_success_jobs": (
                    counts["succeeded"] - unscored_successes - invalidated_successes
                ),
                "unscored_success_jobs": unscored_successes,
                "ground_truth_invalidated_jobs": invalidated_successes,
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
        failed = [row for row in group if row["status"] == "failed"]
        resolved = successful + failed
        expected_weight = sum(int(row["question_count"]) for row in group)
        successful_weight = sum(int(row["question_count"]) for row in successful)
        failed_weight = sum(int(row["question_count"]) for row in failed)
        resolved_weight = successful_weight + failed_weight
        weighted_score = sum(float(row["f1"]) * int(row["question_count"]) for row in successful)
        success_only_f1 = weighted_score / successful_weight if successful_weight else None
        f1 = weighted_score / resolved_weight if resolved_weight else None
        repetition_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in resolved:
            repetition_groups[int(row["repetition"])].append(row)
        repetition_scores = []
        for repetition_rows in repetition_groups.values():
            weight = sum(int(row["question_count"]) for row in repetition_rows)
            repetition_scores.append(
                sum(
                    float(row["f1"]) * int(row["question_count"])
                    for row in repetition_rows
                    if row["status"] == "succeeded" and bool(row["score_available"])
                )
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
                "failed_trajectories": failed_weight,
                "resolved_trajectories": resolved_weight,
                "coverage": resolved_weight / expected_weight,
                "successful_coverage": successful_weight / expected_weight,
                "resolved_repetitions": len(repetition_scores),
                "f1": f1,
                "f1_success_only": success_only_f1,
                "f1_std": f1_std,
                "f1_zero_imputed": f1,
                "f1_all_unresolved_zero": weighted_score / expected_weight,
                "f1_best_case": (weighted_score + expected_weight - resolved_weight)
                / expected_weight,
                "arm_final": all(bool(row["arm_final"]) for row in group),
            }
        )
    return output


def cross_model_scaling_summaries(
    scaling_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Average terminal model arms, counting terminal failed trajectories as zero."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in scaling_rows:
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
        eligible = [row for row in group if row["f1"] is not None and bool(row["arm_final"])]
        if not eligible:
            continue
        expected_models = EXPECTED_MODELS_BY_METHOD_CONTEXT[(method, context)]
        observed_models = {str(row["model"]) for row in eligible}
        values = [float(row["f1_zero_imputed"]) for row in eligible]
        model_std = statistics.stdev(values) if len(values) > 1 else None
        excluded_provisional_models = sorted(
            str(row["model"])
            for row in group
            if row["f1"] is not None and not bool(row["arm_final"])
        )
        is_final = observed_models == expected_models
        output.append(
            {
                "method": method,
                "context": context,
                "tier": tier,
                "mean_f1": statistics.fmean(values),
                "model_std": model_std,
                "model_sem": None if model_std is None else model_std / math.sqrt(len(values)),
                "n_models": len(values),
                "target_n_models": len(expected_models),
                "included_models": ";".join(
                    model for model in MODEL_ORDER if model in observed_models
                ),
                "missing_models": ";".join(
                    model for model in MODEL_ORDER if model in expected_models - observed_models
                ),
                "excluded_provisional_models": ";".join(excluded_provisional_models),
                "is_final": is_final,
            }
        )
    return output


def tier_efficiency_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute recorded resource use per successfully answered trajectory."""
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["model"]), str(row["method"]), str(row["context"]), int(row["tier"]))
        ].append(row)

    output: list[dict[str, Any]] = []
    resource_fields = {
        "cost_chf_per_trajectory": "cost_chf",
        "tokens_per_trajectory": "total_tokens",
        "wall_time_seconds_per_trajectory": "process_wall_time_seconds",
    }
    for (model, method, context, tier), group in sorted(
        grouped.items(),
        key=lambda item: (
            MODEL_ORDER.index(item[0][0]),
            item[0][3],
            METHOD_ORDER.index(item[0][1]),
            CONTEXT_ORDER[item[0][2]],
        ),
    ):
        # Ground-truth corrections can invalidate scientific scores without
        # invalidating observed cost, token, timing, or memory measurements.
        successful = [row for row in group if row["status"] == "succeeded"]
        row_out: dict[str, Any] = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "method": method,
            "context": context,
            "tier": tier,
            "successful_jobs": len(successful),
            "failed_jobs": sum(row["status"] == "failed" for row in group),
            "unresolved_jobs": sum(
                row["status"] in {"running", "stale", "pending"} for row in group
            ),
            "arm_final": all(bool(row["arm_final"]) for row in group),
        }
        for output_name, source_name in resource_fields.items():
            measured = [row for row in successful if row[source_name] is not None]
            measured_trajectories = sum(int(row["question_count"]) for row in measured)
            successful_trajectories = sum(int(row["question_count"]) for row in successful)
            row_out[output_name] = (
                sum(float(row[source_name]) for row in measured) / measured_trajectories
                if measured_trajectories
                else None
            )
            row_out[f"{output_name}_coverage"] = (
                measured_trajectories / successful_trajectories if successful_trajectories else 0.0
            )
        output.append(row_out)
    return output


def cross_model_efficiency_summaries(
    model_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Average recorded per-trajectory resource use across terminal model arms."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in model_rows:
        grouped[(str(row["method"]), str(row["context"]), int(row["tier"]))].append(row)
    metrics = ("tokens_per_trajectory", "wall_time_seconds_per_trajectory")
    output: list[dict[str, Any]] = []
    for (method, context, tier), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][2],
            METHOD_ORDER.index(item[0][0]),
            CONTEXT_ORDER[item[0][1]],
        ),
    ):
        eligible = [
            row
            for row in group
            if bool(row["arm_final"]) and all(row[metric] is not None for metric in metrics)
        ]
        if not eligible:
            continue
        expected_models = EXPECTED_MODELS_BY_METHOD_CONTEXT[(method, context)]
        observed_models = {str(row["model"]) for row in eligible}
        row_out: dict[str, Any] = {
            "method": method,
            "context": context,
            "tier": tier,
            "n_models": len(eligible),
            "target_n_models": len(expected_models),
            "included_models": ";".join(model for model in MODEL_ORDER if model in observed_models),
            "missing_models": ";".join(
                model for model in MODEL_ORDER if model in expected_models - observed_models
            ),
            "is_final": observed_models == expected_models,
        }
        for metric in metrics:
            values = [float(row[metric]) for row in eligible]
            model_std = statistics.stdev(values) if len(values) > 1 else None
            row_out[f"mean_{metric}"] = statistics.fmean(values)
            row_out[f"sem_{metric}"] = (
                None if model_std is None else model_std / math.sqrt(len(values))
            )
            row_out[f"minimum_{metric}_coverage"] = min(
                float(row[f"{metric}_coverage"]) for row in eligible
            )

        expected_paid_models = expected_models & PAID_MODELS
        paid_eligible = [
            row
            for row in group
            if bool(row["arm_final"])
            and str(row["model"]) in PAID_MODELS
            and row["cost_chf_per_trajectory"] is not None
        ]
        observed_paid_models = {str(row["model"]) for row in paid_eligible}
        paid_values = [float(row["cost_chf_per_trajectory"]) for row in paid_eligible]
        paid_std = statistics.stdev(paid_values) if len(paid_values) > 1 else None
        row_out.update(
            {
                "mean_cost_chf_per_trajectory": statistics.fmean(paid_values)
                if paid_values
                else None,
                "sem_cost_chf_per_trajectory": None
                if paid_std is None
                else paid_std / math.sqrt(len(paid_values)),
                "minimum_cost_chf_per_trajectory_coverage": min(
                    (float(row["cost_chf_per_trajectory_coverage"]) for row in paid_eligible),
                    default=0.0,
                ),
                "n_paid_models": len(paid_eligible),
                "target_n_paid_models": len(expected_paid_models),
                "included_paid_models": ";".join(
                    model for model in MODEL_ORDER if model in observed_paid_models
                ),
                "missing_paid_models": ";".join(
                    model
                    for model in MODEL_ORDER
                    if model in expected_paid_models - observed_paid_models
                ),
                "paid_cost_is_final": observed_paid_models == expected_paid_models,
            }
        )
        output.append(row_out)
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
        "All plotting aggregates score terminal failed jobs as zero. Running, stale, and "
        "pending jobs are excluded from the current score and keep their arm provisional. "
        "Scores invalidated by a versioned ground-truth correction are also excluded, keep "
        "their original value in `original_f1`, and reduce the reported score coverage; "
        "resource measurements from those successful runs remain valid.",
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
            "- `codeact_x1000_records.csv`: the final Qwen, DeepSeek, Gemini, and GPT-5-mini CodeAct x1000 extensions.",
            "- `rlm_x1000_records.csv`: terminal RLM x1000 extensions available at the freeze time.",
            "- `final_arm_records.csv`: records belonging to terminal arms.",
            "- `provisional_arm_records.csv`: records belonging to unfinished arms.",
            "- `arm_status.csv`: the finality decision used for legend asterisks.",
            "- `tier_scaling.csv`: the faithful four-tier plotting aggregate.",
            "- `tier_scaling_across_models.csv`: unweighted means and standard errors across "
            "terminal model arms; terminal failed trajectories contribute zero.",
            "- `tier_efficiency_by_model.csv`: recorded cost, tokens, and wall time per "
            "successfully answered trajectory for each model. Failed jobs do not enter resource "
            "averages.",
            "- `tier_efficiency_across_models.csv`: unweighted efficiency means and standard "
            "errors across terminal model arms. Cost averages include only paid Gemini, "
            "GPT-5-mini, and Claude models; free SwissAI access is excluded.",
            "- `capability_split/`: checked per-model and across-model full-corpus RLM "
            "summaries for the six operation-specific task groups.",
            "- `efficiency_appendix/`: checked calls, tokens, recorded latency, tool time, "
            "process wall time, and peak-memory summaries for terminal arms.",
            "- `source_manifest.json`: source snapshot and file checksums.",
            "",
            "## Causal controls",
            "",
            "The `causal_controls/` directory freezes the completed GPT-5-mini "
            "matched-cardinality arm, Qwen/Claude chemistry-rule controls and their ordinary "
            "RLM counterparts, and the deterministic executor ceiling. Its record tables and "
            "source manifest preserve the aggregation rules and contributing snapshots. The "
            "directory also stores a status-explicit provisional Qwen matched-cardinality "
            "snapshot, which is excluded from final inference until all 725 cells terminate.",
            "Exact corrected rescores are included and labeled `corrected_exact_rescore`. "
            "Remaining ground-truth-invalidated control scores are excluded exactly as in "
            "the main benchmark tables; the causal-control aggregates report their score "
            "coverage while retaining all measured resource fields.",
            "",
            "## Prospective-route control",
            "",
            "The `prospective_decomposition/` directory freezes the completed Task-16 "
            "control: two models, three target-information conditions, five repetitions, "
            "and three targets per run (30 jobs and 90 trajectories). Aggregate, per-run, and "
            "per-target records are retained.",
            "",
            "Regenerate from the repository root:",
            "",
            "```bash",
            "uv run --frozen python paper_plots/scripts/build_gold_results.py",
            "uv run --with-requirements paper_plots/requirements.txt \\",
            "  python paper_plots/scripts/plot_gold_scaling_by_tier.py",
            "uv run --with-requirements paper_plots/requirements.txt \\",
            "  python paper_plots/scripts/plot_main_results.py",
            "uv run --with-requirements paper_plots/requirements.txt \\",
            "  python paper_plots/scripts/plot_gold_efficiency_by_tier.py",
            "uv run --frozen python paper_plots/scripts/build_capability_split.py",
            "uv run --frozen python paper_plots/scripts/build_efficiency_appendix.py",
            "uv run --with-requirements paper_plots/requirements.txt \\",
            "  python paper_plots/scripts/plot_efficiency_appendix.py",
            "uv run --frozen python paper_plots/scripts/build_causal_controls.py",
            "uv run --with-requirements paper_plots/requirements.txt \\",
            "  python paper_plots/scripts/plot_control_appendix.py",
            "uv run --frozen python paper_plots/scripts/build_prospective_decomposition.py",
            "uv run --with-requirements paper_plots/requirements.txt \\",
            "  python paper_plots/scripts/plot_prospective_appendix.py",
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
    parser.add_argument(
        "--qwen-x1000-pack",
        type=Path,
        default=DEFAULT_QWEN_X1000_PACK,
        help="Sanitized Qwen CodeAct x1000 result pack.",
    )
    parser.add_argument(
        "--deepseek-rlm-x1000-pack",
        type=Path,
        default=DEFAULT_DEEPSEEK_RLM_X1000_PACK,
        help="Sanitized DeepSeek RLM x1000 Docker-result pack.",
    )
    parser.add_argument(
        "--score-recoveries",
        type=Path,
        default=DEFAULT_SCORE_RECOVERIES,
        help="Provenance-recorded score recoveries for legacy successful runs.",
    )
    parser.add_argument(
        "--ground-truth-corrections",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_CORRECTIONS,
        help="Versioned corrections applied after legacy score recoveries.",
    )
    parser.add_argument(
        "--corrected-score-recoveries",
        type=Path,
        default=DEFAULT_CORRECTED_SCORE_RECOVERIES,
        help="Exact rescores recovered after ground-truth correction.",
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
    score_recovery_manifest = apply_score_recoveries(rows, args.score_recoveries)
    deepseek_extension_rows = [
        flattened
        for run in x1000["runs"]
        if (flattened := flatten_run(run, scope="codeact_x1000"))["model"] == "deepseek-v4-flash"
        and flattened["method"] == "codeact"
    ]
    gpt_extension_rows = [
        flattened
        for run in x1000["runs"]
        if (flattened := flatten_run(run, scope="codeact_x1000"))["model"] == "gpt-5-mini"
        and flattened["method"] == "codeact"
    ]
    expected_qwen_ids = {
        str(row["run_id"]).replace("-x500-", "-x1000-")
        for row in rows
        if row["model"] == "qwen3.5" and row["method"] == "codeact" and row["context"] == "500"
    }
    qwen_extension_rows, qwen_pack_manifest = load_result_pack(
        args.qwen_x1000_pack,
        expected_model="qwen3.5-397b",
        expected_run_ids=expected_qwen_ids,
    )
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
    extension_rows = (
        qwen_extension_rows + deepseek_extension_rows + gemini_extension_rows + gpt_extension_rows
    )
    deepseek_rlm_rows = [
        flattened
        for run in x1000["runs"]
        if (flattened := flatten_run(run, scope="rlm_x1000"))["model"] == "deepseek-v4-flash"
        and flattened["method"] == "rlm"
    ]
    expected_docker_pack_ids = {
        str(row["run_id"]).removeprefix("x1000-openrouter-")
        for row in deepseek_rlm_rows
        if row["task"] in {"tier4/task16", "tier4/task17", "tier4/task17b"}
    }
    docker_rlm_rows, deepseek_rlm_pack_manifest = load_result_pack(
        args.deepseek_rlm_x1000_pack,
        expected_model="deepseek-v4-flash",
        expected_run_ids=expected_docker_pack_ids,
        expected_method="rlm",
        scope="rlm_x1000",
        canonical_run_id_prefix="x1000-openrouter-",
    )
    deepseek_rlm_by_id = {str(row["run_id"]): row for row in deepseek_rlm_rows}
    deepseek_rlm_by_id.update({str(row["run_id"]): row for row in docker_rlm_rows})
    deepseek_rlm_rows = list(deepseek_rlm_by_id.values())
    gemini_rlm_rows = [
        flattened
        for run in x1000["runs"]
        if (flattened := flatten_run(run, scope="rlm_x1000"))["model"] == "gemini-3.7-flash"
        and flattened["method"] == "rlm"
    ]
    if len(gemini_rlm_rows) != 150 or any(
        row["status"] not in {"succeeded", "failed"} for row in gemini_rlm_rows
    ):
        raise ValueError("Gemini RLM x1000 arm must contain 150 terminal records")
    gpt_rlm_rows: list[dict[str, Any]] = []
    gpt_docker_campaign = campaigns.get(GPT_RLM_X1000_DOCKER_CAMPAIGN)
    if gpt_docker_campaign is not None:
        gpt_non_docker_rows = [
            flattened
            for run in x1000["runs"]
            if (flattened := flatten_run(run, scope="rlm_x1000"))["model"] == "gpt-5-mini"
            and flattened["method"] == "rlm"
        ]
        gpt_docker_rows = [
            flatten_run(run, scope="rlm_x1000") for run in gpt_docker_campaign["runs"]
        ]
        gpt_rlm_rows = gpt_non_docker_rows + gpt_docker_rows
        if len(gpt_non_docker_rows) != 135 or len(gpt_docker_rows) != 15:
            raise ValueError(
                "GPT-5-mini RLM x1000 cardinality mismatch: "
                f"{len(gpt_non_docker_rows)} non-Docker and {len(gpt_docker_rows)} Docker"
            )
        unresolved_gpt = [
            row["run_id"] for row in gpt_rlm_rows if row["status"] not in {"succeeded", "failed"}
        ]
        if unresolved_gpt:
            raise ValueError(f"GPT-5-mini RLM x1000 arm is not terminal: {unresolved_gpt[:3]!r}")
    rlm_x1000_rows = deepseek_rlm_rows + gemini_rlm_rows + gpt_rlm_rows
    all_rows = rows + extension_rows + rlm_x1000_rows
    ground_truth_correction_manifest = apply_ground_truth_corrections(
        all_rows, args.ground_truth_corrections
    )
    corrected_recovery_manifest, corrected_recovery_count = apply_corrected_score_recoveries(
        all_rows, args.corrected_score_recoveries
    )
    arms = arm_summaries(all_rows)
    add_arm_finality(all_rows, arms)
    scaling = scaling_summaries(all_rows)
    cross_model_scaling = cross_model_scaling_summaries(scaling)
    efficiency = tier_efficiency_summaries(all_rows)
    cross_model_efficiency = cross_model_efficiency_summaries(efficiency)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "full_benchmark_records.csv", rows)
    write_csv(output / "codeact_x1000_records.csv", extension_rows)
    write_csv(output / "rlm_x1000_records.csv", rlm_x1000_rows)
    write_csv(output / "final_arm_records.csv", [row for row in all_rows if row["arm_final"]])
    write_csv(
        output / "provisional_arm_records.csv", [row for row in all_rows if not row["arm_final"]]
    )
    write_csv(output / "arm_status.csv", arms)
    write_csv(output / "tier_scaling.csv", scaling)
    write_csv(output / "tier_scaling_across_models.csv", cross_model_scaling)
    write_csv(output / "tier_efficiency_by_model.csv", efficiency)
    write_csv(output / "tier_efficiency_across_models.csv", cross_model_efficiency)

    as_of = max(str(snapshot["generated_at"]) for snapshot in snapshots)
    write_readme(output / "README.md", arms, as_of=as_of)
    generated_files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "source_manifest.json"
    )
    campaign_sources = full["sources"] + x1000["sources"]
    if gpt_docker_campaign is not None:
        campaign_sources += gpt_docker_campaign["sources"]
    source_ids = {str(source["id"]) for source in campaign_sources}
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
            **(
                {
                    GPT_RLM_X1000_DOCKER_CAMPAIGN: {
                        "definition_sha256": gpt_docker_campaign["definition_sha256"],
                        "counts": gpt_docker_campaign["counts"],
                    }
                }
                if gpt_docker_campaign is not None
                else {}
            ),
        },
        "result_packs": [
            {
                "path": str(args.qwen_x1000_pack),
                "sha256": sha256_file(args.qwen_x1000_pack),
                "bytes": args.qwen_x1000_pack.stat().st_size,
                "packed_at": qwen_pack_manifest["packed_at"],
                "models": qwen_pack_manifest["models"],
                "n_runs": qwen_pack_manifest["n_runs"],
                "rule": qwen_pack_manifest["rule"],
            },
            {
                "path": str(args.gemini_x1000_pack),
                "sha256": sha256_file(args.gemini_x1000_pack),
                "bytes": args.gemini_x1000_pack.stat().st_size,
                "packed_at": gemini_pack_manifest["packed_at"],
                "models": gemini_pack_manifest["models"],
                "n_runs": gemini_pack_manifest["n_runs"],
                "rule": gemini_pack_manifest["rule"],
            },
            {
                "path": str(args.deepseek_rlm_x1000_pack),
                "sha256": sha256_file(args.deepseek_rlm_x1000_pack),
                "bytes": args.deepseek_rlm_x1000_pack.stat().st_size,
                "packed_at": deepseek_rlm_pack_manifest["packed_at"],
                "models": deepseek_rlm_pack_manifest["models"],
                "n_runs": deepseek_rlm_pack_manifest["n_runs"],
                "rule": deepseek_rlm_pack_manifest["rule"],
            },
        ],
        "score_recoveries": {
            "path": str(args.score_recoveries),
            "sha256": sha256_file(args.score_recoveries),
            "count": len(score_recovery_manifest["recoveries"]),
        },
        "ground_truth_corrections": {
            "path": str(args.ground_truth_corrections),
            "sha256": sha256_file(args.ground_truth_corrections),
            "correction_id": ground_truth_correction_manifest["correction_id"],
            "corrected_bundle": ground_truth_correction_manifest["corrected_bundle"],
            "affected_tasks": ground_truth_correction_manifest["affected_tasks"],
        },
        "corrected_score_recoveries": {
            "path": str(args.corrected_score_recoveries),
            "sha256": sha256_file(args.corrected_score_recoveries),
            "recovery_id": corrected_recovery_manifest["recovery_id"],
            "available": len(corrected_recovery_manifest["recoveries"]),
            "applied": corrected_recovery_count,
        },
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
    if (
        len(rows) != 6300
        or len(qwen_extension_rows) != 150
        or len(deepseek_extension_rows) != 150
        or len(gpt_extension_rows) != 150
        or len(extension_rows) != 600
        or len(deepseek_rlm_rows) != 150
        or len(gemini_rlm_rows) != 150
        or len(gpt_rlm_rows) not in {0, 150}
    ):
        raise ValueError(
            "Gold export cardinality mismatch: "
            f"{len(rows)} main, {len(qwen_extension_rows)} Qwen x1000, "
            f"{len(deepseek_extension_rows)} DeepSeek x1000, "
            f"{len(gemini_extension_rows)} Gemini x1000, "
            f"{len(gpt_extension_rows)} GPT-5-mini x1000, and "
            f"{len(deepseek_rlm_rows)} DeepSeek RLM x1000, "
            f"{len(gemini_rlm_rows)} Gemini RLM x1000, plus "
            f"{len(gpt_rlm_rows)} GPT-5-mini RLM x1000"
        )
    print(
        f"Wrote {len(rows):,} main, {len(extension_rows):,} CodeAct x1000, and "
        f"{len(rlm_x1000_rows):,} RLM x1000 records to {output}"
    )


if __name__ == "__main__":
    main()
