"""Deterministic upper bound for the five oracle-predicate task configurations."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from rlm.codeact_helpers import build_context_pipeline, load_lines
from rxnhaystack.metrics import RunMetrics, write_run_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = os.environ.get(
    "RXNHAYSTACK_CLEANED_DATASET",
    str(Path.home() / "datasets/rxnhaystack/reactionSmilesFigShareUSPTO2023_cleaned.txt"),
)
SEED = int(os.environ.get("RXNHAYSTACK_SEED", "42"))
CONTEXT_SIZE = int(os.environ.get("RXNHAYSTACK_CONTEXT_SIZE", "100"))


def add_tier_path(tier: str) -> None:
    path = str(PROJECT_ROOT / tier)
    if path not in sys.path:
        sys.path.insert(0, path)


def context_hash(context: str) -> str:
    return hashlib.sha256(context.encode()).hexdigest()


def predicate_source_hash(tier: str) -> str:
    return hashlib.sha256((PROJECT_ROOT / tier / "oracle_predicates.py").read_bytes()).hexdigest()


def line_index(line: str) -> int:
    return int(line.split(" ", 1)[0])


def exact_set_result(
    *,
    key: str,
    context: str,
    predicted: set[Any],
    expected: set[Any],
) -> dict[str, Any]:
    return {
        "question": key,
        "context_sha256": context_hash(context),
        "predicted_count": len(predicted),
        "ground_truth_count": len(expected),
        "exact_match": predicted == expected,
    }


def execute_task6(lines: list[str]) -> list[dict[str, Any]]:
    add_tier_path("tier3")
    from oracle_predicates import task6_reaction_matches
    from rlm_task6 import AMIDE_COUPLING_LABELS, MIN_SELECTED_GROUND_TRUTH
    from task6_hardcoded_ground_truth import TASK6_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION

    pipeline = build_context_pipeline(
        name="random",
        lines=lines,
        rng=random.Random(SEED),
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )
    results = []
    for key in AMIDE_COUPLING_LABELS:
        expected_full = set(TASK6_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION[key])
        context = pipeline.build_context(
            context_size=CONTEXT_SIZE,
            correct_indices=expected_full,
            query=key,
        )
        context_lines = context.splitlines()
        expected = expected_full & {line_index(line) for line in context_lines}
        predicted = {
            line_index(line) for line in context_lines if task6_reaction_matches(line, key)
        }
        results.append(
            exact_set_result(key=key, context=context, predicted=predicted, expected=expected)
        )
    return results


def execute_task10(lines: list[str]) -> list[dict[str, Any]]:
    add_tier_path("tier3")
    from oracle_predicates import task10_reaction_matches
    from rlm_task10 import MIN_SELECTED_GROUND_TRUTH
    from task10_hardcoded_ground_truth import TASK10_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION
    from task10_prompt_config import REACTION_KEYS

    pipeline = build_context_pipeline(
        name="random",
        lines=lines,
        rng=random.Random(SEED),
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )
    results = []
    for key in REACTION_KEYS:
        expected_full = set(TASK10_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION[key])
        context = pipeline.build_context(
            context_size=CONTEXT_SIZE,
            correct_indices=expected_full,
            query=key,
        )
        context_lines = context.splitlines()
        expected = expected_full & {line_index(line) for line in context_lines}
        predicted = {
            line_index(line) for line in context_lines if task10_reaction_matches(line, key)
        }
        results.append(
            exact_set_result(key=key, context=context, predicted=predicted, expected=expected)
        )
    return results


def execute_task23(lines: list[str]) -> list[dict[str, Any]]:
    add_tier_path("tier3")
    from oracle_predicates import task23_reaction_matches
    from rlm_task23 import MIN_SELECTED_GROUND_TRUTH, REACTION_KEY
    from task23_hardcoded_ground_truth import TASK23_HARDCODED_GROUND_TRUTH_INDICES

    expected_full = set(TASK23_HARDCODED_GROUND_TRUTH_INDICES)
    pipeline = build_context_pipeline(
        name="random",
        lines=lines,
        rng=random.Random(SEED),
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )
    context = pipeline.build_context(
        context_size=CONTEXT_SIZE,
        correct_indices=expected_full,
        query=REACTION_KEY,
    )
    context_lines = context.splitlines()
    expected = expected_full & {line_index(line) for line in context_lines}
    predicted = {line_index(line) for line in context_lines if task23_reaction_matches(line)}
    return [
        exact_set_result(
            key=REACTION_KEY,
            context=context,
            predicted=predicted,
            expected=expected,
        )
    ]


def execute_task13(lines: list[str]) -> list[dict[str, Any]]:
    add_tier_path("tier4")
    from task13_fg_chain_graph import ground_truth_fg_path_in_context
    from task13_fg_chain_ground_truth import (
        FIXED_QUESTIONS,
        TASK13_MIN_SELECTED_GROUND_TRUTH,
        chains_for_context_sampling,
    )

    results = []
    for offset, question in enumerate(FIXED_QUESTIONS):
        sampling = chains_for_context_sampling(question, CONTEXT_SIZE)
        pipeline = build_context_pipeline(
            name="random",
            lines=lines,
            rng=random.Random(SEED + offset),
            min_selected_ground_truth=TASK13_MIN_SELECTED_GROUND_TRUTH,
        )
        context = pipeline.build_context(
            context_size=CONTEXT_SIZE,
            correct_indices=set(sampling.support_indices),
            query=f"fg_chain_{question.source_fg}_{question.target_fg}",
        )
        first = ground_truth_fg_path_in_context(
            context.splitlines(), question.source_fg, question.target_fg
        )[0]
        second = ground_truth_fg_path_in_context(
            context.splitlines(), question.source_fg, question.target_fg
        )[0]
        if first is None or second is None:
            raise RuntimeError(f"No deterministic Task-13 result for {question.key}")
        expected = set(first.accepted_reaction_indices or (first.reaction_indices,))
        predicted = set(second.accepted_reaction_indices or (second.reaction_indices,))
        results.append(
            exact_set_result(
                key=f"{question.source_fg}->{question.target_fg}",
                context=context,
                predicted=predicted,
                expected=expected,
            )
        )
    return results


def execute_task14(lines: list[str]) -> list[dict[str, Any]]:
    add_tier_path("tier4")
    from task14_protecting_group_graph import ground_truth_pairs_in_context
    from task14_protecting_group_ground_truth import (
        FIXED_QUESTIONS,
        TASK14_MIN_SELECTED_GROUND_TRUTH,
        pairs_for_context_sampling,
    )

    results = []
    for offset, spec in enumerate(FIXED_QUESTIONS):
        sampling = pairs_for_context_sampling(spec, CONTEXT_SIZE)
        pipeline = build_context_pipeline(
            name="random",
            lines=lines,
            rng=random.Random(SEED + offset),
            min_selected_ground_truth=TASK14_MIN_SELECTED_GROUND_TRUTH,
        )
        context = pipeline.build_context(
            context_size=CONTEXT_SIZE,
            correct_indices=set(sampling.support_indices),
            query=f"pg_pairs_{spec.label}",
        )
        first = ground_truth_pairs_in_context(context.splitlines(), spec.label)
        second = ground_truth_pairs_in_context(context.splitlines(), spec.label)
        expected = {(pair.install_index, pair.remove_index) for pair in first}
        predicted = {(pair.install_index, pair.remove_index) for pair in second}
        results.append(
            exact_set_result(
                key=spec.label,
                context=context,
                predicted=predicted,
                expected=expected,
            )
        )
    return results


EXECUTORS = {
    ("tier3", "6"): execute_task6,
    ("tier3", "10"): execute_task10,
    ("tier3", "23"): execute_task23,
    ("tier4", "13"): execute_task13,
    ("tier4", "14"): execute_task14,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("tier3", "tier4"), required=True)
    parser.add_argument("--task", choices=("6", "10", "23", "13", "14"), required=True)
    args = parser.parse_args()
    key = (args.tier, args.task)
    if key not in EXECUTORS:
        parser.error(f"Unsupported oracle executor task: {args.tier}/task{args.task}")

    started = time.monotonic()
    results = EXECUTORS[key](load_lines(DATASET_PATH))
    if not results or not all(result["exact_match"] for result in results):
        raise RuntimeError(f"Deterministic oracle mismatch for {args.tier}/task{args.task}")
    write_run_metrics(
        RunMetrics(
            calls=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_seconds=time.monotonic() - started,
            tool_time_seconds=time.monotonic() - started,
            cost_chf=0.0,
            cost_usd=0.0,
            results={
                "deterministic_oracle": True,
                "tier": args.tier,
                "task": args.task,
                "context_size": CONTEXT_SIZE,
                "seed": SEED,
                "predicate_condition": "oracle-executor-v1",
                "predicate_source_sha256": predicate_source_hash(args.tier),
                "sample_count": len(results),
                "macro_f1": 1.0,
                "exact_match_accuracy": 1.0,
                "samples": results,
            },
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
