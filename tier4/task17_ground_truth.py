"""Hardcoded ground truth for tier4 task17 (SMIRKS sequential chains).

Full-dataset 2-reaction chains were mined from reactionSmilesFigShareUSPTO2023_cleaned.txt
using Rxn-INSIGHT SMIRKS templates (one template per step), canonical-SMILES product-to-reactant
linking, and per-question persistence checks.

All accepted chains live in task17_hardcoded_chains.json. At evaluation time,
in-context GT is the subset of those chains whose reaction indices all appear in the
sampled context.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from task17_smirks_sequential_graph import (
    BUILTIN_QUESTIONS,
    CHAIN_LENGTH,
    DATASET_TOTAL_REACTIONS,
    QuestionSpec,
)

TASK17_TOTAL_REACTIONS = DATASET_TOTAL_REACTIONS
TASK17_FORCED_CHAIN_COUNT = 2
TASK17_MIN_SELECTED_GROUND_TRUTH = CHAIN_LENGTH * TASK17_FORCED_CHAIN_COUNT
TASK17_GROUND_TRUTH_DEFINITION = (
    "ordered distinct 2-reaction chains [r_0, r_1] where step 1 matches one fixed SMIRKS "
    "template, step 2 matches another fixed SMIRKS template, consecutive reactions link by "
    "exact canonical SMILES equality between a product of r_0 and a reactant of r_1, and a "
    "question-specific persistence predicate holds on the canonical spine; only reactions "
    "present in context"
)

TIER4_DIR = Path(__file__).resolve().parent
CHAINS_JSON_PATH = TIER4_DIR / "task17_hardcoded_chains.json"

FIXED_QUESTIONS: list[QuestionSpec] = list(BUILTIN_QUESTIONS)

HARDCODED_GT_CHAIN_COUNTS: dict[str, int] = {
    "boc_buchwald": 30,
    "nitro_reduction_amide": 190,
    "suzuki_carbonyl_reduction": 14,
    "wittig_hydrogenation": 3,
    "alcohol_oxidation_reductive_amination": 133,
}

HARDCODED_GT_EXAMPLE: dict[str, tuple[int, ...]] = {
    "boc_buchwald": (4789, 4790),
    "nitro_reduction_amide": (5107, 5108),
    "suzuki_carbonyl_reduction": (19676, 24317),
    "wittig_hydrogenation": (18439, 18440),
    "alcohol_oxidation_reductive_amination": (3804, 3805),
}


@dataclass(frozen=True)
class Task17ContextSampling:
    selected_chains: tuple[tuple[int, ...], ...]
    support_indices: frozenset[int]
    forced_count: int
    selected_chain_count: int


def tier3_forced_reaction_count(
    answer_count: int,
    context_size: int,
    *,
    dataset_size: int = TASK17_TOTAL_REACTIONS,
    min_selected_ground_truth: int = TASK17_MIN_SELECTED_GROUND_TRUTH,
) -> int:
    if context_size < 0:
        top_k = dataset_size
    else:
        top_k = min(context_size, dataset_size)
    if top_k == 0 or answer_count == 0:
        return 0

    ratio_scaled_floor = int((answer_count / dataset_size) * top_k)
    half_cap = top_k // 2
    return min(
        answer_count,
        half_cap,
        max(min_selected_ground_truth, ratio_scaled_floor),
    )


@lru_cache(maxsize=1)
def load_mined_payload() -> dict[str, dict[str, object]]:
    with CHAINS_JSON_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hardcoded_chains_for_question(question_id: str) -> tuple[tuple[int, ...], ...]:
    payload = load_mined_payload()[question_id]
    return tuple(tuple(chain) for chain in payload["chains"])


def full_support_indices_for_question(spec: QuestionSpec) -> set[int]:
    return {idx for chain in hardcoded_chains_for_question(spec.question_id) for idx in chain}


def example_chain_for_question(spec: QuestionSpec) -> tuple[int, ...]:
    example = HARDCODED_GT_EXAMPLE.get(spec.question_id)
    if example is None:
        raise KeyError(f"No hardcoded example chain for {spec.question_id}")
    if len(example) != CHAIN_LENGTH:
        raise ValueError(
            f"Expected chain length {CHAIN_LENGTH}, got {len(example)} for {spec.question_id}"
        )
    return example


def full_dataset_chain_count(spec: QuestionSpec) -> int:
    return HARDCODED_GT_CHAIN_COUNTS.get(
        spec.question_id,
        len(hardcoded_chains_for_question(spec.question_id)),
    )


def chains_for_context_sampling(
    spec: QuestionSpec,
    context_size: int,
) -> Task17ContextSampling:
    all_chains = hardcoded_chains_for_question(spec.question_id)
    full_support = full_support_indices_for_question(spec)
    if context_size < 0:
        return Task17ContextSampling(
            selected_chains=all_chains,
            support_indices=frozenset(full_support),
            forced_count=len(full_support),
            selected_chain_count=len(all_chains),
        )

    forced_count = tier3_forced_reaction_count(len(full_support), context_size)
    target_chain_count = min(
        max(1, math.ceil(forced_count / CHAIN_LENGTH)),
        len(all_chains),
        TASK17_FORCED_CHAIN_COUNT,
    )

    selected_chains: list[tuple[int, ...]] = []
    support_acc: set[int] = set()

    def try_add_chain(chain: tuple[int, ...], *, require_disjoint: bool) -> bool:
        if chain in selected_chains:
            return False
        if require_disjoint and set(chain) & support_acc:
            return False
        candidate_support = support_acc | set(chain)
        pipeline_forced = tier3_forced_reaction_count(len(candidate_support), context_size)
        if len(candidate_support) > pipeline_forced:
            return False
        selected_chains.append(chain)
        support_acc.update(chain)
        return True

    for chain in all_chains:
        if len(selected_chains) >= target_chain_count:
            break
        try_add_chain(chain, require_disjoint=True)

    if len(selected_chains) < target_chain_count:
        for chain in all_chains:
            if len(selected_chains) >= target_chain_count:
                break
            try_add_chain(chain, require_disjoint=False)

    if not selected_chains and all_chains:
        first = all_chains[0]
        selected_chains = [first]
        support_acc = set(first)

    selected = tuple(selected_chains)
    return Task17ContextSampling(
        selected_chains=selected,
        support_indices=frozenset(support_acc),
        forced_count=forced_count,
        selected_chain_count=len(selected),
    )


def random_pool_excluded_indices(
    spec: QuestionSpec,
    support_indices: set[int] | frozenset[int],
    *,
    context_size: int,
) -> frozenset[int]:
    """GT reaction indices from chains other than the forced support set."""
    if context_size < 0:
        return frozenset()
    full_support = full_support_indices_for_question(spec)
    return frozenset(full_support - set(support_indices))


def format_task17_chains(chains: list[tuple[int, ...]], limit: int = 20) -> str:
    shown = [",".join(str(x) for x in chain) for chain in chains[:limit]]
    if len(chains) > limit:
        shown.append(f"... (+{len(chains) - limit} more)")
    return " | ".join(shown)


def print_task17_startup_banner(*, max_chains_per_question: int) -> None:
    for spec in FIXED_QUESTIONS:
        count = full_dataset_chain_count(spec)
        support = len(full_support_indices_for_question(spec))
        example = example_chain_for_question(spec)
        print(
            f"Ground truth [{spec.question_id}] label={spec.label} "
            f"full_dataset_chains={count} support_indices={support} "
            f"example_chain={list(example)} max_chains_per_question={max_chains_per_question}"
        )


def print_task17_sample_context(
    *,
    sample_index: int,
    spec: QuestionSpec,
    gt_chains: list[tuple[int, ...]],
    sampling: Task17ContextSampling,
    full_support_indices: set[int],
    support_indices: set[int],
    support_in_context: int,
    context_size: int,
    context_line_count: int,
    context_coverage: float,
) -> None:
    print(
        f"\nQuestion {sample_index + 1}/{len(FIXED_QUESTIONS)}: "
        f"{spec.label} ({spec.question_id}), gt_chains={len(gt_chains)}"
    )
    print(
        f"[CONTEXT] requested_size={context_size} actual_size={context_line_count} "
        f"selected_chains={sampling.selected_chain_count}/{full_dataset_chain_count(spec)} "
        f"forced_count={sampling.forced_count} "
        f"support_in_context={support_in_context}/{len(support_indices)} "
        f"full_support={len(full_support_indices)} "
        f"in_context_chains={len(gt_chains)}/{sampling.selected_chain_count} "
        f"coverage={context_coverage:.4f}"
    )


def print_task17_sample_metrics(
    *,
    sample_index: int,
    spec: QuestionSpec,
    response: str,
    predicted: set[tuple[int, ...]],
    gt_set: set[tuple[int, ...]],
    precision: float,
    recall: float,
    f1: float,
    exact_set_match: bool,
) -> None:
    print(f"Response [sample={sample_index}]: {response[:500]}{'…' if len(response) > 500 else ''}")
    print(f"Predicted [{spec.question_id}]: {len(predicted)} chains")
    print(f"Ground truth [{spec.question_id}]: {len(gt_set)} chains")
    print(
        f"Metrics [sample={sample_index}] -> exact_set_match={int(exact_set_match)} "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
    )


def build_task17_wandb_sample_log(
    *,
    sample_index: int,
    spec: QuestionSpec,
    gt_chains: list[tuple[int, ...]],
    sampling: Task17ContextSampling,
    full_support_indices: set[int],
    support_indices: set[int],
    support_in_context: int,
    predicted: set[tuple[int, ...]],
    precision: float,
    recall: float,
    f1: float,
    exact_set_match: float,
    response: str,
    context_size: int,
    context_coverage: float,
    context_line_count: int,
    completion_prompt_char_count: int,
    final_input_tokens: int,
    final_output_tokens: int,
    final_total_tokens: int,
    iterations: int,
    sample_cost_usd: float | None = None,
) -> dict[str, object]:
    gt_set = {tuple(chain) for chain in gt_chains}
    payload: dict[str, object] = {
        "sample_iteration": sample_index,
        f"sample/{sample_index}/question_id": spec.question_id,
        f"sample/{sample_index}/label": spec.label,
        f"sample/{sample_index}/ground_truth_count": len(gt_chains),
        f"sample/{sample_index}/ground_truth_full_count": full_dataset_chain_count(spec),
        f"sample/{sample_index}/ground_truth_chains": format_task17_chains(gt_chains),
        f"sample/{sample_index}/pred_chain_count": len(predicted),
        f"sample/{sample_index}/pred_chains": format_task17_chains(sorted(predicted)),
        f"sample/{sample_index}/selected_chain_count": sampling.selected_chain_count,
        f"sample/{sample_index}/forced_reaction_count": sampling.forced_count,
        f"sample/{sample_index}/support_indices_in_context": support_in_context,
        f"sample/{sample_index}/support_indices_selected_count": len(support_indices),
        f"sample/{sample_index}/support_indices_full_count": len(full_support_indices),
        f"sample/{sample_index}/response_char_count": len(response),
        f"sample/{sample_index}/precision": precision,
        f"sample/{sample_index}/recall": recall,
        f"sample/{sample_index}/f1": f1,
        f"sample/{sample_index}/exact_set_match": exact_set_match,
        f"sample/{sample_index}/completion_prompt_char_count": completion_prompt_char_count,
        f"sample/{sample_index}/context_size": context_size,
        f"sample/{sample_index}/context_coverage": context_coverage,
        f"sample/{sample_index}/retrieved_line_count": context_line_count,
        f"sample/{sample_index}/final_total_input_tokens": final_input_tokens,
        f"sample/{sample_index}/final_total_output_tokens": final_output_tokens,
        f"sample/{sample_index}/final_total_tokens": final_total_tokens,
        f"sample/{sample_index}/iterations": iterations,
        f"sample/{sample_index}/gt_chain_keys": format_task17_chains(sorted(gt_set)),
    }
    if sample_cost_usd is not None:
        payload[f"sample/{sample_index}/final_total_cost_usd"] = sample_cost_usd
    return payload


def print_task17_run_summary(
    *,
    total: int,
    exact_match_count: int,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"SMIRKS sequential-chain questions evaluated: {total}")
    print(f"Exact set match: {exact_match_count}/{total}")
    print(f"Exact set match accuracy: {exact_match_count / total if total else 0.0:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


def update_task17_run_summary(
    run,
    *,
    total: int,
    exact_match_count: int,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
    total_input_tokens: int,
    total_output_tokens: int,
    samples_with_cost: int,
    total_cost_usd: float,
) -> None:
    run.summary["questions_evaluated"] = total
    run.summary["exact_set_match_correct"] = exact_match_count
    run.summary["exact_set_match_accuracy"] = exact_match_count / total if total else 0.0
    run.summary["macro_precision"] = macro_precision
    run.summary["macro_recall"] = macro_recall
    run.summary["macro_f1"] = macro_f1
    run.summary["avg_total_input_tokens_per_sample"] = (
        total_input_tokens / total if total else 0.0
    )
    run.summary["avg_total_output_tokens_per_sample"] = (
        total_output_tokens / total if total else 0.0
    )
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost


def ground_truth_chains_in_context(
    context_lines: list[str],
    spec: QuestionSpec,
    *,
    max_chains_per_question: int = 0,
) -> list[tuple[int, ...]]:
    from task17_smirks_sequential_graph import (
        ground_truth_chains_in_context as filter_chains_to_context,
    )

    all_chains = list(hardcoded_chains_for_question(spec.question_id))
    in_context = filter_chains_to_context(context_lines, all_chains)
    if max_chains_per_question > 0:
        return in_context[:max_chains_per_question]
    return in_context


# Backward-compatible aliases
def load_mined_chains() -> dict[str, dict[str, object]]:
    return load_mined_payload()


def ground_truth_chains(question_id: str) -> list[tuple[int, ...]]:
    return list(hardcoded_chains_for_question(question_id))


def question_spec(question_id: str) -> QuestionSpec:
    from task17_smirks_sequential_graph import QUESTION_BY_ID

    return QUESTION_BY_ID[question_id]


def all_question_specs() -> tuple[QuestionSpec, ...]:
    return BUILTIN_QUESTIONS
