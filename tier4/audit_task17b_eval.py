"""Verify task17b ground truth chemistry and eval sampling."""

from __future__ import annotations

import random

from rlm.codeact_helpers import build_context_pipeline, load_lines
from task17b_ground_truth import (
    FIXED_QUESTIONS,
    TASK17B_FORCED_CHAIN_COUNT,
    TASK17B_TOTAL_REACTIONS,
    chains_for_context_sampling,
    full_support_indices_for_question,
    ground_truth_chains_in_context,
    hardcoded_chains_for_question,
    min_selected_ground_truth_for_spec,
    random_pool_excluded_indices,
    tier3_forced_reaction_count,
)
from task17b_smirks_sequential_graph import (
    build_line_mol_cache,
    classify_question_steps,
    parse_records_from_lines,
    verify_chain,
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


def audit_all_chains(lines, records, lines_by_index, mol_cache) -> list[str]:
    failures: list[str] = []
    print("\n=== Chemistry: verify_chain on ALL hardcoded chains ===")
    for spec in FIXED_QUESTIONS:
        step_hits, _ = classify_question_steps(lines, spec, mol_cache=mol_cache)
        step_hit_sets = [set(h.keys()) for h in step_hits]
        chains = hardcoded_chains_for_question(spec.question_id)
        bad = 0
        for chain in chains:
            ok, reason = verify_chain(
                chain,
                spec.question_id,
                records,
                lines_by_index,
                step_hits=step_hit_sets,
                chain_length=spec.chain_length,
            )
            if not ok:
                failures.append(f"{spec.question_id}: chain {chain} -> {reason}")
                bad += 1
        status = "OK" if bad == 0 else "FAIL"
        print(f"  {status} {spec.question_id}: {len(chains)} chains, {bad} failures")
    return failures


def min_union_support(chains: tuple[tuple[int, ...], ...], count: int) -> int:
    """Smallest union of reaction indices across `count` chains (overlap allowed)."""
    from itertools import combinations

    if len(chains) < count:
        return len({idx for c in chains for idx in c})
    best = float("inf")
    for combo in combinations(chains, count):
        union = {idx for c in combo for idx in c}
        best = min(best, len(union))
    return int(best)


def audit_context_size(lines: list[str], context_size: int) -> list[str]:
    failures: list[str] = []

    print(f"\n{'=' * 72}")
    print(f"CONTEXT_SIZE={context_size}")
    print(f"{'=' * 72}")

    for qi, spec in enumerate(FIXED_QUESTIONS):
        full_support = full_support_indices_for_question(spec)
        all_chains = hardcoded_chains_for_question(spec.question_id)
        sampling_plan = chains_for_context_sampling(spec, context_size)
        min_selected = min_selected_ground_truth_for_spec(spec)
        expected_forced = tier3_forced_reaction_count(
            len(sampling_plan.support_indices),
            context_size,
            spec=spec,
        )

        print(
            f"\n[{spec.question_id}] L={spec.chain_length} "
            f"full_support={len(full_support)} full_chains={len(all_chains)} "
            f"plan_chains={sampling_plan.selected_chain_count} "
            f"plan_support={len(sampling_plan.support_indices)} "
            f"plan_forced_budget={sampling_plan.forced_count} "
            f"pipeline_forced_expected={expected_forced}"
        )

        if context_size < 0:
            if sampling_plan.selected_chain_count != len(all_chains):
                failures.append(
                    f"{spec.question_id}@{context_size}: expected all {len(all_chains)} chains"
                )
            if len(sampling_plan.support_indices) != len(full_support):
                failures.append(
                    f"{spec.question_id}@{context_size}: expected full support {len(full_support)}"
                )
        else:
            target_chains = min(TASK17B_FORCED_CHAIN_COUNT, len(all_chains))
            if sampling_plan.selected_chain_count < target_chains:
                failures.append(
                    f"{spec.question_id}@{context_size}: expected {target_chains} chains, "
                    f"got {sampling_plan.selected_chain_count}"
                )
            achievable = min_union_support(all_chains, target_chains)
            if len(sampling_plan.support_indices) > achievable:
                failures.append(
                    f"{spec.question_id}@{context_size}: support {len(sampling_plan.support_indices)} "
                    f"> achievable minimum {achievable}"
                )
            if len(sampling_plan.support_indices) > sampling_plan.forced_count:
                failures.append(
                    f"{spec.question_id}@{context_size}: support > plan forced budget"
                )
            if len(sampling_plan.support_indices) > expected_forced:
                failures.append(
                    f"{spec.question_id}@{context_size}: support > pipeline forced budget"
                )

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
                min_selected_ground_truth=min_selected,
            )
            context_text = pipeline.build_context(
                context_size=context_size,
                correct_indices=support_indices,
                query=f"task17b_{spec.question_id}",
                excluded_indices=sampling_excluded,
            )
            context_lines = [line for line in context_text.splitlines() if line.strip()]
            context_ids = reaction_ids_in_context(context_lines)
            support_in_context = support_indices & context_ids
            gt_chains = ground_truth_chains_in_context(context_lines, spec)
            leaked = other_gt & context_ids

            if context_size >= 0 and len(context_lines) != context_size:
                failures.append(
                    f"{spec.question_id}@{context_size}/seed{seed}: "
                    f"context len {len(context_lines)} != {context_size}"
                )
            if support_in_context != support_indices:
                failures.append(
                    f"{spec.question_id}@{context_size}/seed{seed}: "
                    f"missing forced support {support_indices - support_in_context}"
                )
            if context_size >= 0 and leaked:
                failures.append(
                    f"{spec.question_id}@{context_size}/seed{seed}: "
                    f"other-GT leakage {len(leaked)} indices"
                )
            if not gt_chains and context_size >= 0:
                failures.append(
                    f"{spec.question_id}@{context_size}/seed{seed}: no scorable GT in context"
                )
            if context_size < 0 and len(gt_chains) != len(all_chains):
                failures.append(
                    f"{spec.question_id}@-1/seed{seed}: "
                    f"in-context GT {len(gt_chains)} != full {len(all_chains)}"
                )

            print(
                f"  seed={seed}: ctx={len(context_lines)} "
                f"forced_in={len(support_in_context)}/{len(support_indices)} "
                f"gt_chains={len(gt_chains)} leakage={len(leaked)}"
            )

    return failures


def audit_in_context_gt_chemistry(
    lines, records, lines_by_index, mol_cache, context_size: int
) -> list[str]:
    failures: list[str] = []
    print(f"\n=== In-context GT chemistry check (context_size={context_size}) ===")
    for qi, spec in enumerate(FIXED_QUESTIONS):
        step_hits, _ = classify_question_steps(lines, spec, mol_cache=mol_cache)
        step_hit_sets = [set(h.keys()) for h in step_hits]
        sampling_plan = chains_for_context_sampling(spec, context_size)
        support_indices = set(sampling_plan.support_indices)
        sampling_excluded = random_pool_excluded_indices(
            spec, support_indices, context_size=context_size
        )
        min_selected = min_selected_ground_truth_for_spec(spec)
        pipeline = build_context_pipeline(
            name="random",
            lines=lines,
            rng=random.Random(SEED + qi),
            min_selected_ground_truth=min_selected,
        )
        context_text = pipeline.build_context(
            context_size=context_size,
            correct_indices=support_indices,
            query=f"task17b_{spec.question_id}",
            excluded_indices=sampling_excluded,
        )
        context_lines = [line for line in context_text.splitlines() if line.strip()]
        gt_chains = ground_truth_chains_in_context(context_lines, spec)
        for chain in gt_chains:
            ok, reason = verify_chain(
                chain,
                spec.question_id,
                records,
                lines_by_index,
                step_hits=step_hit_sets,
                chain_length=spec.chain_length,
            )
            if not ok:
                failures.append(f"{spec.question_id}: in-context GT {chain} -> {reason}")
        print(f"  OK {spec.question_id}: {len(gt_chains)} in-context chains verified")
    return failures


def main() -> None:
    lines = load_lines(DATASET_PATH)
    records = parse_records_from_lines(lines)
    lines_by_index = {int(line.split(" ", 1)[0]): line for line in lines}
    print(f"Loaded {len(lines)} lines (expected ~{TASK17B_TOTAL_REACTIONS})")
    print("Building mol cache...")
    mol_cache = build_line_mol_cache(lines)
    print("Done.")

    failures: list[str] = []
    failures.extend(audit_all_chains(lines, records, lines_by_index, mol_cache))
    for context_size in CONTEXT_SIZES:
        failures.extend(audit_context_size(lines, context_size))
    failures.extend(audit_in_context_gt_chemistry(lines, records, lines_by_index, mol_cache, 100))
    failures.extend(audit_in_context_gt_chemistry(lines, records, lines_by_index, mol_cache, -1))

    print(f"\n{'=' * 72}")
    if failures:
        print(f"AUDIT FAILED ({len(failures)} issues):")
        for f in failures[:30]:
            print(f"  - {f}")
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
        raise SystemExit(1)
    print("AUDIT PASSED: all 5 questions — chemistry, sampling, and in-context GT verified.")


if __name__ == "__main__":
    main()
