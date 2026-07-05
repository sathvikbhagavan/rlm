"""Verify task17 context sampling at context sizes 100 and 500."""

from __future__ import annotations

import random

from rlm.codeact_helpers import build_context_pipeline, load_lines
from task17_ground_truth import (
    CHAIN_LENGTH,
    FIXED_QUESTIONS,
    TASK17_FORCED_CHAIN_COUNT,
    TASK17_MIN_SELECTED_GROUND_TRUTH,
    TASK17_TOTAL_REACTIONS,
    chains_for_context_sampling,
    full_support_indices_for_question,
    ground_truth_chains_in_context,
    hardcoded_chains_for_question,
    random_pool_excluded_indices,
    tier3_forced_reaction_count,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
SEED = 42
CONTEXT_SIZES = (100, 500, -1)
SAMPLES_PER_QUESTION = 5


def reaction_ids_in_context(context_lines: list[str]) -> set[int]:
    return {
        int(line.split(" ", 1)[0])
        for line in context_lines
        if line.strip() and " " in line and line.split(" ", 1)[0].isdigit()
    }


def audit_context_size(lines: list[str], context_size: int) -> list[str]:
    failures: list[str] = []

    print(f"\n{'=' * 72}")
    print(f"CONTEXT_SIZE={context_size}")
    print(f"{'=' * 72}")

    for qi, spec in enumerate(FIXED_QUESTIONS):
        full_support = full_support_indices_for_question(spec)
        all_chains = hardcoded_chains_for_question(spec.question_id)
        sampling_plan = chains_for_context_sampling(spec, context_size)
        expected_forced = tier3_forced_reaction_count(
            len(sampling_plan.support_indices), context_size
        )

        print(
            f"\n[{spec.question_id}] "
            f"full_support={len(full_support)} "
            f"full_chains={len(all_chains)} "
            f"plan_chains={sampling_plan.selected_chain_count} "
            f"plan_support={len(sampling_plan.support_indices)} "
            f"plan_forced_budget={sampling_plan.forced_count} "
            f"pipeline_forced_expected={expected_forced}"
        )

        if context_size < 0:
            if sampling_plan.selected_chain_count != len(all_chains):
                failures.append(
                    f"{spec.question_id}@{context_size}: expected all {len(all_chains)} chains, "
                    f"got {sampling_plan.selected_chain_count}"
                )
            if len(sampling_plan.support_indices) != len(full_support):
                failures.append(
                    f"{spec.question_id}@{context_size}: expected full support {len(full_support)}, "
                    f"got {len(sampling_plan.support_indices)}"
                )
        elif sampling_plan.selected_chain_count < TASK17_FORCED_CHAIN_COUNT and len(all_chains) >= TASK17_FORCED_CHAIN_COUNT:
            failures.append(
                f"{spec.question_id}@{context_size}: expected {TASK17_FORCED_CHAIN_COUNT} chains, "
                f"got {sampling_plan.selected_chain_count}"
            )
        elif len(sampling_plan.support_indices) < TASK17_MIN_SELECTED_GROUND_TRUTH and len(all_chains) >= TASK17_FORCED_CHAIN_COUNT:
            failures.append(
                f"{spec.question_id}@{context_size}: support {len(sampling_plan.support_indices)} "
                f"< min forced reactions {TASK17_MIN_SELECTED_GROUND_TRUTH}"
            )
        if context_size >= 0 and len(sampling_plan.support_indices) > sampling_plan.forced_count:
            failures.append(
                f"{spec.question_id}@{context_size}: support {len(sampling_plan.support_indices)} "
                f"> plan forced budget {sampling_plan.forced_count}"
            )
        if context_size >= 0 and len(sampling_plan.support_indices) > expected_forced:
            failures.append(
                f"{spec.question_id}@{context_size}: support {len(sampling_plan.support_indices)} "
                f"> pipeline forced budget {expected_forced}"
            )

        bad_ids = [idx for idx in sampling_plan.support_indices if idx < 0 or idx >= len(lines)]
        if bad_ids:
            failures.append(f"{spec.question_id}@{context_size}: invalid reaction ids {bad_ids[:5]}")

        for attempt in range(SAMPLES_PER_QUESTION):
            seed = SEED + qi * 1000 + attempt
            support_indices = set(sampling_plan.support_indices)
            sampling_excluded = random_pool_excluded_indices(
                spec, support_indices, context_size=context_size
            )
            other_gt = full_support - support_indices
            pipeline = build_context_pipeline(
                name="random",
                lines=lines,
                rng=random.Random(seed),
                min_selected_ground_truth=TASK17_MIN_SELECTED_GROUND_TRUTH,
            )
            context_text = pipeline.build_context(
                context_size=context_size,
                correct_indices=support_indices,
                query=f"task17_{spec.question_id}",
                excluded_indices=sampling_excluded,
            )
            context_lines = [line for line in context_text.splitlines() if line.strip()]
            context_ids = reaction_ids_in_context(context_lines)
            context_coverage = len(context_lines) / len(lines) if lines else 0.0
            support_in_context = sampling_plan.support_indices & context_ids
            gt_chains = ground_truth_chains_in_context(context_lines, spec)
            selected_in_context = [
                chain for chain in sampling_plan.selected_chains if all(r in context_ids for r in chain)
            ]

            if len(context_lines) != (len(lines) if context_size < 0 else min(context_size, len(lines))):
                failures.append(
                    f"{spec.question_id}@{context_size}/try{attempt}: "
                    f"context_len={len(context_lines)} expected={len(lines) if context_size < 0 else min(context_size, len(lines))}"
                )
            if len(support_in_context) != len(sampling_plan.support_indices):
                failures.append(
                    f"{spec.question_id}@{context_size}/try{attempt}: "
                    f"forced support missing {len(support_in_context)}/{len(sampling_plan.support_indices)}"
                )
            if not gt_chains:
                failures.append(
                    f"{spec.question_id}@{context_size}/try{attempt}: zero gt chains in context"
                )
            if not selected_in_context:
                failures.append(
                    f"{spec.question_id}@{context_size}/try{attempt}: "
                    "no selected-forced chain fully in context"
                )
            leaked_other_gt = other_gt & context_ids
            if leaked_other_gt:
                failures.append(
                    f"{spec.question_id}@{context_size}/try{attempt}: "
                    f"other GT leaked into random pool: {sorted(leaked_other_gt)[:5]}"
                )
            if context_size < 0:
                if len(gt_chains) != len(all_chains):
                    failures.append(
                        f"{spec.question_id}@{context_size}/try{attempt}: "
                        f"gt_chains={len(gt_chains)} != full_dataset={len(all_chains)}"
                    )
                if context_coverage < 1.0:
                    failures.append(
                        f"{spec.question_id}@{context_size}/try{attempt}: "
                        f"coverage={context_coverage:.4f} expected 1.0"
                    )
            elif len(gt_chains) != len(selected_in_context):
                failures.append(
                    f"{spec.question_id}@{context_size}/try{attempt}: "
                    f"gt_chains={len(gt_chains)} != selected_in_context={len(selected_in_context)}"
                )
            for chain in gt_chains:
                if len(chain) != CHAIN_LENGTH:
                    failures.append(f"{spec.question_id}@{context_size}/try{attempt}: bad chain len {chain}")

            print(
                f"  try={attempt} seed={seed} ctx={len(context_lines)} "
                f"forced_in_ctx={len(support_in_context)}/{len(sampling_plan.support_indices)} "
                f"other_gt_leaked={len(leaked_other_gt)} "
                f"selected_chains_in_ctx={len(selected_in_context)}/{sampling_plan.selected_chain_count} "
                f"gt_chains_in_ctx={len(gt_chains)}"
            )

    return failures


def print_tier3_table() -> None:
    print(f"\nTier3 forced-count table (dataset={TASK17_TOTAL_REACTIONS})")
    header = f"{'question':<40} {'support':>7} " + " ".join(f"cs={cs:>3}" for cs in CONTEXT_SIZES)
    print(header)
    for spec in FIXED_QUESTIONS:
        support = len(full_support_indices_for_question(spec))
        counts = [tier3_forced_reaction_count(support, cs) for cs in CONTEXT_SIZES]
        chain_counts = [
            chains_for_context_sampling(spec, cs).selected_chain_count for cs in CONTEXT_SIZES
        ]
        print(
            f"{spec.question_id:<40} {support:>7} "
            + " ".join(f"{c:>7}" for c in counts)
            + "  chains:"
            + " ".join(f"{c:>3}" for c in chain_counts)
        )


def main() -> None:
    lines = load_lines(DATASET_PATH)
    print(f"dataset_lines={len(lines)}")

    spec = FIXED_QUESTIONS[0]
    support = list(full_support_indices_for_question(spec))[:3]
    for rid in support:
        if lines[rid].split(" ", 1)[0] != str(rid):
            print(f"WARNING: reaction id {rid} != line prefix at lines[{rid}]")
            break
    else:
        print(f"index convention OK (checked {support})")

    print_tier3_table()

    all_failures: list[str] = []
    for context_size in CONTEXT_SIZES:
        all_failures.extend(audit_context_size(lines, context_size))

    if all_failures:
        print(f"\nFAILED ({len(all_failures)} issues):")
        for msg in all_failures:
            print(f"  - {msg}")
        raise SystemExit(1)

    print(f"\nOK: all checks passed for context sizes {CONTEXT_SIZES}")


if __name__ == "__main__":
    main()
