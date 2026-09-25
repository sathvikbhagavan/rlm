#!/usr/bin/env python3
"""Recover and exactly rescore historical outputs after ground-truth corrections."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from rxnhaystack.control_room import (
    load_snapshot_directory,
    merge_snapshots,
    scientific_dashboard_view,
)
from rxnhaystack.score_recovery import (
    ANSI_ESCAPE_RE,
    TASK_QUESTION_IDS,
    context_indices,
    context_size_from_label,
    extract_tier3_predictions,
    historical_min_selected_ground_truth,
    load_ground_truth,
    parse_indices,
    parse_sample_metrics,
    positive_cardinality_from_condition,
    precision_recall_f1,
    reconstruct_task_contexts,
    score_tier3_run,
    sha256_bytes,
)

DEFAULT_RECORDS = (
    Path("paper_plots/gold/iclr2027/full_benchmark_records.csv"),
    Path("paper_plots/gold/iclr2027/codeact_x1000_records.csv"),
    Path("paper_plots/gold/iclr2027/rlm_x1000_records.csv"),
    Path("paper_plots/gold/iclr2027/causal_controls/records.csv"),
    Path("paper_plots/gold/iclr2027/causal_controls/matched_qwen_provisional.csv"),
)
DEFAULT_HISTORICAL_GT = Path("human_eval/generated/canonical-v4/admin/ground_truth.jsonl")
DEFAULT_CORRECTED_GT = Path("human_eval/generated/canonical-v7/admin/ground_truth.jsonl")
DEFAULT_HISTORICAL_TASK15 = Path(
    "paper_plots/gold/source_packs/task15_historical_chains_pre_ffbbe41.json"
)
DEFAULT_DATASET = Path(
    "/home/amin/datasets/rxnhaystack/reactionSmilesFigShareUSPTO2023_cleaned.txt"
)
AFFECTED_TASKS = frozenset((*TASK_QUESTION_IDS, "tier4/task15"))
TASK15_RING_ORDER = ("quinoline", "indole", "benzothiazole", "benzimidazole")
TASK15_METRIC_RE = re.compile(r"Metrics \[sample=(\d+)\].*?reaction_f1=([0-9.]+)", re.DOTALL)
TASK15_PREDICTION_RE = re.compile(r"Predicted \[([^\]]+)\]:\s*\((.*?)\)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a generated ledger only after its complete content is durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_targets(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("score_correction_status") not in {
                    "historical_score_invalidated",
                    "corrected_exact_rescore",
                }:
                    continue
                run_id = str(row["run_id"])
                method = row.get("method")
                if method is None:
                    method = "executor" if row.get("arm") == "executor" else "rlm"
                normalized = {
                    **row,
                    "method": method,
                    "source_tables": [str(path)],
                }
                if run_id in targets:
                    targets[run_id]["source_tables"].append(str(path))
                    if not targets[run_id].get("condition") and row.get("condition"):
                        targets[run_id]["condition"] = row["condition"]
                else:
                    targets[run_id] = normalized
    return targets


def discover_wandb_urls(cache_dir: Path) -> dict[str, str]:
    snapshots = load_snapshot_directory(cache_dir)
    merged = scientific_dashboard_view(
        merge_snapshots(snapshots, stale_after_seconds=10 * 365 * 24 * 3600)
    )
    output: dict[str, str] = {}
    for campaign in merged["campaigns"]:
        for run in campaign["runs"]:
            successful = [
                attempt
                for attempt in run.get("attempts", ())
                if attempt.get("status") == "succeeded"
            ]
            for attempt in sorted(successful, key=lambda item: item.get("started_at", "")):
                url = (attempt.get("metrics") or {}).get("wandb_url")
                if url:
                    output[str(run["run_id"])] = str(url)
    return output


def wandb_output_endpoint(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) != 4 or parts[2] != "runs":
        raise ValueError(f"Unexpected W&B run URL: {url}")
    entity, project, _runs, run_id = parts
    return f"https://api.wandb.ai/files/{entity}/{project}/{run_id}/output.log"


def safe_cache_name(run_id: str) -> str:
    return hashlib.sha256(run_id.encode()).hexdigest() + ".log.gz"


def fetch_log(
    *,
    run_id: str,
    wandb_url: str,
    api_key: str,
    cache_dir: Path,
) -> tuple[str, bytes | None, str]:
    path = cache_dir / safe_cache_name(run_id)
    if path.exists():
        return run_id, gzip.decompress(path.read_bytes()), "wandb-cache"
    response = requests.get(
        wandb_output_endpoint(wandb_url),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    )
    if response.status_code != 200:
        return run_id, None, f"wandb-http-{response.status_code}"
    content = response.content
    path.write_bytes(gzip.compress(content, compresslevel=6))
    return run_id, content, "wandb-output"


def fetch_logs(
    *,
    targets: dict[str, dict[str, Any]],
    urls: dict[str, str],
    api_key: str,
    cache_dir: Path,
    workers: int,
) -> tuple[dict[str, bytes], dict[str, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    inputs = [
        (run_id, urls[run_id])
        for run_id, target in targets.items()
        if target["method"] != "executor" and run_id in urls
    ]

    def retrieve(item: tuple[str, str]) -> tuple[str, bytes | None, str]:
        return fetch_log(
            run_id=item[0],
            wandb_url=item[1],
            api_key=api_key,
            cache_dir=cache_dir,
        )

    logs: dict[str, bytes] = {}
    status: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for run_id, content, source in pool.map(retrieve, inputs):
            status[run_id] = source
            if content is not None:
                logs[run_id] = content
    return logs, status


def original_score(target: dict[str, Any]) -> float:
    value = target.get("original_f1")
    if value in (None, ""):
        raise ValueError(f"No historical score for {target['run_id']}")
    return float(value)


def task15_predictions(log_text: str) -> dict[str, tuple[int, ...]]:
    cleaned = ANSI_ESCAPE_RE.sub("", log_text)
    return {
        match.group(1): parse_indices(match.group(2))
        for match in TASK15_PREDICTION_RE.finditer(cleaned)
        if match.group(1) in TASK15_RING_ORDER
    }


def task15_logged_f1(log_text: str) -> dict[int, float]:
    cleaned = ANSI_ESCAPE_RE.sub("", log_text)
    return {
        int(match.group(1)): float(match.group(2)) for match in TASK15_METRIC_RE.finditer(cleaned)
    }


def selected_task15_support(
    *, old_chains: tuple[tuple[int, ...], ...], context_size: int, dataset_size: int
) -> set[int]:
    full_support = {index for chain in old_chains for index in chain}
    top_k = dataset_size if context_size < 0 else min(context_size, dataset_size)
    forced_count = min(
        len(full_support),
        top_k // 2,
        max(3, int((len(full_support) / dataset_size) * top_k)),
    )
    target_chain_count = min(max(1, (forced_count + 2) // 3), len(old_chains))
    selected: list[tuple[int, ...]] = []
    support: set[int] = set()
    for chain in old_chains:
        if len(selected) >= target_chain_count:
            break
        candidate = support | set(chain)
        candidate_forced = min(
            len(candidate),
            top_k // 2,
            max(3, int((len(candidate) / dataset_size) * top_k)),
        )
        if len(candidate) > candidate_forced:
            continue
        selected.append(chain)
        support = candidate
    if not selected and old_chains:
        support = set(old_chains[0])
    return support


def task15_reaction_f1(
    prediction: tuple[int, ...], accepted_chains: tuple[tuple[int, ...], ...]
) -> float:
    predicted = set(prediction)
    return max(
        (precision_recall_f1(predicted, set(chain))[2] for chain in accepted_chains),
        default=0.0,
    )


class Task15Contexts:
    def __init__(self, *, dataset_path: Path, historical_chains_path: Path):
        self.dataset_path = dataset_path
        self.dataset = [line.strip() for line in dataset_path.open() if line.strip()]
        payload = json.loads(historical_chains_path.read_text(encoding="utf-8"))
        self.old_chains = {
            ring: tuple(tuple(int(index) for index in chain) for chain in item["chains"])
            for ring, item in payload.items()
        }
        self.cache: dict[
            tuple[int, str], tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]
        ] = {}

    def accepted_chains(
        self, *, context_size: int, ring_system: str, sample_index: int
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        key = (context_size, ring_system)
        if key in self.cache:
            return self.cache[key]
        old = self.old_chains[ring_system]
        support = selected_task15_support(
            old_chains=old,
            context_size=context_size,
            dataset_size=len(self.dataset),
        )
        selected_indices = context_indices(
            dataset_size=len(self.dataset),
            context_size=context_size,
            correct_indices=support,
            rng=__import__("random").Random(42 + sample_index),
            min_selected_ground_truth=max(3, len(support)),
        )
        if context_size < 0:
            historical = old
            from task15_ring_chain_ground_truth import hardcoded_chains_for_question

            corrected = tuple(hardcoded_chains_for_question(ring_system))
        else:
            context_lines = [f"{index} {self.dataset[index]}" for index in sorted(selected_indices)]
            historical = self._mine_context_chains(
                context_lines=context_lines,
                ring_system=ring_system,
                replacement_chains=old,
            )
            corrected = self._mine_context_chains(
                context_lines=context_lines,
                ring_system=ring_system,
                replacement_chains=None,
            )
        self.cache[key] = historical, corrected
        return self.cache[key]

    @staticmethod
    def _mine_context_chains(
        *,
        context_lines: list[str],
        ring_system: str,
        replacement_chains: tuple[tuple[int, ...], ...] | None,
    ) -> tuple[tuple[int, ...], ...]:
        import task15_ring_chain_ground_truth as ground_truth_module
        from task15_ring_chain_graph import ground_truth_ring_path_in_context

        original = ground_truth_module.hardcoded_chains_for_question
        if replacement_chains is not None:
            ground_truth_module.hardcoded_chains_for_question = lambda _ring: replacement_chains
        try:
            ground_truth, _filters = ground_truth_ring_path_in_context(context_lines, ring_system)
        finally:
            ground_truth_module.hardcoded_chains_for_question = original
        if ground_truth is None:
            raise ValueError(f"No Task-15 {ring_system} chain in reconstructed context")
        return tuple(
            tuple(chain)
            for chain in (
                ground_truth.accepted_reaction_indices or (ground_truth.reaction_indices,)
            )
        )


def recover_task15(
    *,
    target: dict[str, Any],
    log_text: str,
    contexts: Task15Contexts,
) -> tuple[float | None, list[dict[str, Any]], dict[str, tuple[int, ...]]]:
    predictions = task15_predictions(log_text)
    old_logged = task15_logged_f1(log_text)
    if set(predictions) != set(TASK15_RING_ORDER) or set(old_logged) != set(range(4)):
        return None, [], predictions
    context_size = context_size_from_label(str(target["context"]))
    rows: list[dict[str, Any]] = []
    for sample_index, ring_system in enumerate(TASK15_RING_ORDER):
        prediction = predictions[ring_system]
        if ring_system not in {"quinoline", "indole"}:
            rows.append(
                {
                    "question_id": f"rxh-t4-task15-{ring_system}",
                    "old_f1": old_logged[sample_index],
                    "corrected_f1": old_logged[sample_index],
                    "recovery": "ground-truth-unchanged",
                }
            )
            continue
        historical, corrected = contexts.accepted_chains(
            context_size=context_size,
            ring_system=ring_system,
            sample_index=sample_index,
        )
        old_f1 = task15_reaction_f1(prediction, historical)
        corrected_f1 = task15_reaction_f1(prediction, corrected)
        if abs(old_f1 - old_logged[sample_index]) > 5e-4:
            return None, rows, predictions
        rows.append(
            {
                "question_id": f"rxh-t4-task15-{ring_system}",
                "old_f1": old_f1,
                "corrected_f1": corrected_f1,
                "recovery": "exact-task15-chain-rescore",
                "historical_accepted_chain_count": len(historical),
                "corrected_accepted_chain_count": len(corrected),
                "predicted_count": len(prediction),
            }
        )
    return sum(row["corrected_f1"] for row in rows) / 4, rows, predictions


def recover_executor(
    *, target: dict[str, Any], dataset_lines: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    """Rerun the deterministic oracle against the corrected task definitions."""

    from experiments.iclr2027 import oracle_executor

    task = str(target["task"])
    tier, task_name = task.split("/", 1)
    task_id = task_name.removeprefix("task")
    context_size = context_size_from_label(str(target["context"]))
    previous_context_size = oracle_executor.CONTEXT_SIZE
    oracle_executor.CONTEXT_SIZE = context_size
    try:
        results = oracle_executor.EXECUTORS[(tier, task_id)](dataset_lines)
    finally:
        oracle_executor.CONTEXT_SIZE = previous_context_size
    if not results or not all(bool(result["exact_match"]) for result in results):
        raise RuntimeError(f"Corrected deterministic executor failed for {target['run_id']}")
    question_ids = TASK_QUESTION_IDS[task]
    if len(question_ids) != len(results):
        raise ValueError(f"Executor result cardinality mismatch for {target['run_id']}")
    rows = [
        {
            "question_id": question_id,
            "old_f1": 1.0,
            "corrected_f1": 1.0,
            "recovery": "corrected-deterministic-executor-rerun",
            "context_sha256": result["context_sha256"],
            "predicted_count": result["predicted_count"],
            "corrected_ground_truth_count": result["ground_truth_count"],
            "exact_match": result["exact_match"],
        }
        for question_id, result in zip(question_ids, results, strict=True)
    ]
    return 1.0, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/score-recovery"))
    parser.add_argument(
        "--output", type=Path, default=Path("paper_plots/gold/corrected_score_recoveries.json")
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("paper_plots/gold/source_packs/corrected_score_recovery_predictions.jsonl"),
    )
    parser.add_argument("--historical-ground-truth", type=Path, default=DEFAULT_HISTORICAL_GT)
    parser.add_argument("--corrected-ground-truth", type=Path, default=DEFAULT_CORRECTED_GT)
    parser.add_argument("--historical-task15", type=Path, default=DEFAULT_HISTORICAL_TASK15)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-task15", action="store_true")
    parser.add_argument("--allow-recovery-regression", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY is required to retrieve preserved W&B output logs")
    targets = load_targets(DEFAULT_RECORDS)
    urls = discover_wandb_urls(args.snapshot_dir)
    logs, fetch_status = fetch_logs(
        targets=targets,
        urls=urls,
        api_key=api_key,
        cache_dir=args.cache_dir / "wandb-output",
        workers=args.workers,
    )
    historical = load_ground_truth(args.historical_ground_truth)
    corrected = load_ground_truth(args.corrected_ground_truth)
    dataset_size = sum(1 for line in args.dataset.open() if line.strip())
    dataset_sha256 = sha256_file(args.dataset)
    corrected_ground_truth_sha256 = sha256_file(args.corrected_ground_truth)

    cached_executor_recoveries: dict[str, dict[str, Any]] = {}
    previous_recovery_count = 0
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        previous_inputs = previous.get("inputs") or {}
        if (previous.get("dataset") or {}).get("sha256") == dataset_sha256 and (
            previous_inputs.get("corrected_ground_truth") or {}
        ).get("sha256") == corrected_ground_truth_sha256:
            previous_recovery_count = len(previous.get("recoveries", ()))
            cached_executor_recoveries = {
                str(recovery["run_id"]): recovery
                for recovery in previous.get("recoveries", ())
                if recovery.get("prediction_source") == "corrected-deterministic-executor-rerun"
            }

    repo_root = Path(__file__).resolve().parents[2]
    repo_root_path = str(repo_root)
    if repo_root_path not in sys.path:
        sys.path.insert(0, repo_root_path)
    tier4_path = str(repo_root / "tier4")
    if tier4_path not in sys.path:
        sys.path.insert(0, tier4_path)
    task15_contexts = None
    if not args.skip_task15:
        task15_contexts = Task15Contexts(
            dataset_path=args.dataset,
            historical_chains_path=args.historical_task15,
        )
    executor_dataset_lines: list[str] | None = None

    recoveries: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    context_cache: dict[tuple[str, int, int | None, int], dict[str, frozenset[int]]] = {}
    for run_id, target in sorted(targets.items()):
        task = str(target["task"])
        method = str(target["method"])
        if method == "executor":
            cached_executor = cached_executor_recoveries.get(run_id)
            if cached_executor is not None:
                if (
                    str(cached_executor["task"]) != task
                    or abs(float(cached_executor["historical_score"]) - original_score(target))
                    > 5e-4
                ):
                    raise ValueError(f"Stale cached executor recovery for {run_id}")
                recoveries.append(cached_executor)
                continue
            if executor_dataset_lines is None:
                raw_lines = [line.strip() for line in args.dataset.open() if line.strip()]
                executor_dataset_lines = [f"{index} {line}" for index, line in enumerate(raw_lines)]
            corrected_score, question_scores = recover_executor(
                target=target, dataset_lines=executor_dataset_lines
            )
            historical_score = original_score(target)
            if abs(historical_score - 1.0) > 5e-4:
                raise ValueError(f"Unexpected historical executor score for {run_id}")
            recoveries.append(
                {
                    "run_id": run_id,
                    "task": task,
                    "score_name": "macro_f1",
                    "score_value": corrected_score,
                    "historical_score": historical_score,
                    "recomputed_historical_score": historical_score,
                    "question_scores": question_scores,
                    "prediction_source": "corrected-deterministic-executor-rerun",
                    "source_sha256": "",
                    "wandb_url": "",
                }
            )
            continue
        content = logs.get(run_id)
        if content is None:
            unresolved.append(
                {
                    "run_id": run_id,
                    "task": task,
                    "reason": fetch_status.get(run_id, "no-wandb-url"),
                    "wandb_url": urls.get(run_id, ""),
                }
            )
            continue
        log_text = content.decode("utf-8", errors="replace")
        log_sha256 = sha256_bytes(content)
        recovery_diagnostics: dict[str, Any] = {}
        if task == "tier4/task15":
            if task15_contexts is None:
                unresolved.append({"run_id": run_id, "task": task, "reason": "task15-skipped"})
                continue
            corrected_score, question_scores, predictions = recover_task15(
                target=target,
                log_text=log_text,
                contexts=task15_contexts,
            )
            recovery_diagnostics = {
                "parsed_prediction_count": len(predictions),
                "resolved_question_count": len(question_scores),
            }
            for ring_system, prediction in predictions.items():
                prediction_rows.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "question_id": f"rxh-t4-task15-{ring_system}",
                        "predicted_indices": list(prediction),
                        "source": fetch_status[run_id],
                        "source_sha256": log_sha256,
                        "extraction_method": "task15-structured-console-field",
                    }
                )
        else:
            metrics = parse_sample_metrics(log_text)
            extracted = (
                extract_tier3_predictions(log_text, task=task, method=method)
                if method in {"codeact", "rlm"}
                else {}
            )
            context_size = context_size_from_label(str(target["context"]))
            cardinality = positive_cardinality_from_condition(str(target.get("condition", "")))
            minimum_positives = historical_min_selected_ground_truth(task=task, method=method)
            cache_key = (task, context_size, cardinality, minimum_positives)
            if cache_key not in context_cache:
                context_cache[cache_key] = reconstruct_task_contexts(
                    task=task,
                    context_size=context_size,
                    historical_ground_truth=historical,
                    dataset_size=dataset_size,
                    min_selected_ground_truth=minimum_positives,
                    positive_cardinality=cardinality,
                )
            corrected_score, question_scores = score_tier3_run(
                task=task,
                context_by_question=context_cache[cache_key],
                historical_ground_truth=historical,
                corrected_ground_truth=corrected,
                metrics=metrics,
                extracted=extracted,
            )
            recovery_diagnostics = {
                "parsed_metric_count": len(metrics),
                "extracted_prediction_count": len(extracted),
                "resolved_question_count": len(question_scores),
            }
            for question_id, prediction in extracted.items():
                prediction_rows.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "question_id": question_id,
                        "predicted_indices": list(prediction.indices),
                        "source": fetch_status[run_id],
                        "source_sha256": log_sha256,
                        "extraction_method": prediction.extraction_method,
                    }
                )
        if corrected_score is None:
            unresolved.append(
                {
                    "run_id": run_id,
                    "task": task,
                    "reason": "prediction-underdetermined-or-validation-failed",
                    "wandb_url": urls.get(run_id, ""),
                    "source_sha256": log_sha256,
                    **recovery_diagnostics,
                }
            )
            continue
        historical_score = original_score(target)
        validation_score = sum(float(row["old_f1"]) for row in question_scores) / len(
            question_scores
        )
        if abs(validation_score - historical_score) > 5e-4:
            unresolved.append(
                {
                    "run_id": run_id,
                    "task": task,
                    "reason": "historical-macro-score-validation-failed",
                    "historical_score": historical_score,
                    "recomputed_historical_score": validation_score,
                    "source_sha256": log_sha256,
                }
            )
            continue
        recoveries.append(
            {
                "run_id": run_id,
                "task": task,
                "score_name": "macro_reaction_f1" if task == "tier4/task15" else "macro_f1",
                "score_value": corrected_score,
                "historical_score": historical_score,
                "recomputed_historical_score": validation_score,
                "question_scores": question_scores,
                "prediction_source": fetch_status[run_id],
                "source_sha256": log_sha256,
                "wandb_url": urls.get(run_id, ""),
            }
        )

    if len(recoveries) < previous_recovery_count and not args.allow_recovery_regression:
        raise RuntimeError(
            "Refusing to replace a recovery ledger with fewer exact rescores: "
            f"{len(recoveries)} < {previous_recovery_count}"
        )

    prediction_content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(prediction_rows, key=lambda item: (item["run_id"], item["question_id"]))
    )
    atomic_write_text(args.predictions_output, prediction_content)
    payload = {
        "schema_version": 1,
        "recovery_id": "rxnhaystack-corrected-score-recovery-2026-09-25-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "historical_bundle": "rxnhaystack-human-1.3.0",
        "corrected_bundle": "rxnhaystack-human-1.6.0",
        "dataset": {
            "path": str(args.dataset),
            "sha256": dataset_sha256,
            "records": dataset_size,
        },
        "inputs": {
            "historical_ground_truth": {
                "path": str(args.historical_ground_truth),
                "sha256": sha256_file(args.historical_ground_truth),
            },
            "corrected_ground_truth": {
                "path": str(args.corrected_ground_truth),
                "sha256": corrected_ground_truth_sha256,
            },
            "historical_task15": {
                "path": str(args.historical_task15),
                "sha256": sha256_file(args.historical_task15),
            },
            "prediction_extract": {
                "path": str(args.predictions_output),
                "sha256": sha256_file(args.predictions_output),
            },
        },
        "counts": {
            "unique_affected_runs": len(targets),
            "recovered": len(recoveries),
            "unresolved": len(unresolved),
            "fetch_status": dict(Counter(fetch_status.values())),
            "recovered_by_task": dict(Counter(row["task"] for row in recoveries)),
            "unresolved_by_reason": dict(Counter(row["reason"] for row in unresolved)),
        },
        "recoveries": recoveries,
        "unresolved": unresolved,
    }
    atomic_write_text(args.output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
