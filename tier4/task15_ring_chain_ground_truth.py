"""Hardcoded ring-construction chain ground truth for tier4 task15."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from task15_ring_chain_graph import (
    BUILTIN_RING_SYSTEMS,
    DATASET_TOTAL_REACTIONS,
    MAX_HEAVY_ATOMS,
    MAX_MOLECULE_FREQ_REFERENCE,
    MIN_HEAVY_ATOMS,
    MIN_LOCAL_MOLECULE_FREQ,
    PATH_LENGTH,
    RingSystemSpec,
)

TASK15_TOTAL_REACTIONS = DATASET_TOTAL_REACTIONS
TASK15_MIN_SELECTED_GROUND_TRUTH = PATH_LENGTH
TASK15_GROUND_TRUTH_DEFINITION = (
    f"reaction chain of exactly {PATH_LENGTH} reactions building a fused ring system from an acyclic "
    "precursor via exact canonical-SMILES product-to-reactant links; ring detection by RDKit SMARTS; "
    f"full-dataset hub filter (>{MAX_MOLECULE_FREQ_REFERENCE} at {DATASET_TOTAL_REACTIONS} reactions); "
    f"context-local hub filter scaled from that reference with floor {MIN_LOCAL_MOLECULE_FREQ}; "
    f"heavy-atom window [{MIN_HEAVY_ATOMS}, {MAX_HEAVY_ATOMS}]; only reactions present in context"
)

DEFAULT_RING_QUERIES: tuple[str, ...] = (
    "quinoline",
    "indole",
    "benzothiazole",
    "benzimidazole",
)

HARDCODED_CHAINS_JSON = Path(__file__).with_name("task15_ring_hardcoded_chains.json")

HARDCODED_GT_CHAIN_COUNTS: dict[str, int] = {
    'quinoline': 199,
    'indole': 184,
    'benzothiazole': 44,
    'benzimidazole': 142,
}

HARDCODED_GT_EXAMPLE: dict[str, tuple[int, ...]] = {
    'quinoline': (1015, 1016, 1017),
    'indole': (2583, 2584, 2585),
    'benzothiazole': (16766, 16711, 111531),
    'benzimidazole': (7688, 7689, 7690),
}


@dataclass(frozen=True)
class RingQuestion:
    ring_system: str

    @property
    def key(self) -> str:
        return self.ring_system


def question_key(ring_system: str) -> str:
    return ring_system


FIXED_QUESTIONS: list[RingQuestion] = [
    RingQuestion(ring_system) for ring_system in DEFAULT_RING_QUERIES
]


@dataclass(frozen=True)
class Task15ContextSampling:
    selected_chains: tuple[tuple[int, ...], ...]
    support_indices: frozenset[int]
    forced_count: int
    selected_chain_count: int


def tier3_forced_reaction_count(
    answer_count: int,
    context_size: int,
    *,
    dataset_size: int = TASK15_TOTAL_REACTIONS,
    min_selected_ground_truth: int = TASK15_MIN_SELECTED_GROUND_TRUTH,
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
def _load_mined_payload() -> dict[str, dict[str, object]]:
    with HARDCODED_CHAINS_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hardcoded_chains_for_question(ring_system: str) -> tuple[tuple[int, ...], ...]:
    payload = _load_mined_payload()[question_key(ring_system)]
    return tuple(tuple(chain) for chain in payload["chains"])


def ring_spec_for_question(question: RingQuestion) -> RingSystemSpec:
    for spec in BUILTIN_RING_SYSTEMS:
        if spec.label == question.ring_system:
            return spec
    raise KeyError(f"Unknown ring system: {question.ring_system}")


def full_support_indices_for_question(question: RingQuestion) -> set[int]:
    return {
        idx
        for chain in hardcoded_chains_for_question(question.ring_system)
        for idx in chain
    }


def chains_for_context_sampling(
    question: RingQuestion,
    context_size: int,
) -> Task15ContextSampling:
    all_chains = hardcoded_chains_for_question(question.ring_system)
    full_support = full_support_indices_for_question(question)
    forced_count = tier3_forced_reaction_count(len(full_support), context_size)
    target_chain_count = min(
        max(1, math.ceil(forced_count / PATH_LENGTH)),
        len(all_chains),
    )

    selected_chains: list[tuple[int, ...]] = []
    support_acc: set[int] = set()
    for chain in all_chains:
        if len(selected_chains) >= target_chain_count:
            break
        candidate_support = support_acc | set(chain)
        pipeline_forced = tier3_forced_reaction_count(len(candidate_support), context_size)
        if len(candidate_support) > pipeline_forced:
            continue
        selected_chains.append(chain)
        support_acc = candidate_support

    if not selected_chains and all_chains:
        selected_chains = [all_chains[0]]
        support_acc = set(all_chains[0])

    selected = tuple(selected_chains)
    return Task15ContextSampling(
        selected_chains=selected,
        support_indices=frozenset(support_acc),
        forced_count=forced_count,
        selected_chain_count=len(selected),
    )


def full_dataset_chain_count(question: RingQuestion) -> int:
    return HARDCODED_GT_CHAIN_COUNTS.get(
        question.key,
        len(hardcoded_chains_for_question(question.ring_system)),
    )


def format_task15_chains(chains: list[tuple[int, ...]], limit: int = 20) -> str:
    shown = ["->".join(str(idx) for idx in chain) for chain in chains[:limit]]
    if len(chains) > limit:
        shown.append(f"... (+{len(chains) - limit} more)")
    return " | ".join(shown)


def print_task15_startup_banner() -> None:
    for question in FIXED_QUESTIONS:
        example = HARDCODED_GT_EXAMPLE.get(question.key)
        count = full_dataset_chain_count(question)
        if example is None:
            print(
                f"Ground truth [3-reaction {question.ring_system}] "
                f"chains={count} (not yet mined)"
            )
            continue
        print(
            f"Ground truth [3-reaction {question.ring_system}] "
            f"chains={count} example={list(example)}"
        )


def print_task15_sample_context(
    *,
    sample_index: int,
    question: RingQuestion,
    gt,
    sampling: Task15ContextSampling,
    full_support_indices: set[int],
    support_indices: set[int],
    support_in_context: int,
    context_size: int,
    context_line_count: int,
    context_coverage: float,
    filters,
) -> None:
    gt_chains = sorted(gt.accepted_reaction_indices or (gt.reaction_indices,))
    print(
        f"\nQuestion {sample_index + 1}: ring_system={question.ring_system} "
        f"gt_chains={len(gt_chains)}"
    )
    print(
        f"[CONTEXT] requested_size={context_size} actual_size={context_line_count} "
        f"selected_chains={sampling.selected_chain_count}/{full_dataset_chain_count(question)} "
        f"forced_count={sampling.forced_count} "
        f"support_in_context={support_in_context}/{len(support_indices)} "
        f"full_support={len(full_support_indices)} "
        f"in_context_chains={len(gt_chains)}/{sampling.selected_chain_count} "
        f"coverage={context_coverage:.4f} "
        f"molecule_freq_cap={filters.molecule_freq_cap}"
    )


def print_task15_sample_metrics(
    *,
    sample_index: int,
    question: RingQuestion,
    response: str,
    pred_rxns: tuple[int, ...],
    scores: dict[str, float | str],
) -> None:
    print(f"Response [sample={sample_index}]: {response[:500]}{'…' if len(response) > 500 else ''}")
    print(f"Predicted [{question.ring_system}]: {pred_rxns}")
    print(
        f"Metrics [sample={sample_index}] -> correct={scores['is_correct']:.0f} "
        f"valid={scores['valid_path']:.0f} "
        f"objective_len={scores['objective_length_match']:.0f} "
        f"index_match={scores['index_match']:.0f} "
        f"reaction_f1={scores['reaction_f1']:.4f} "
        f"reason={scores['validity_reason']}"
    )


def build_task15_wandb_sample_log(
    *,
    sample_index: int,
    question: RingQuestion,
    gt,
    sampling: Task15ContextSampling,
    full_support_indices: set[int],
    support_indices: set[int],
    support_in_context: int,
    filters,
    pred_rxns: tuple[int, ...],
    scores: dict[str, float | str],
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
    gt_chains = sorted(gt.accepted_reaction_indices or (gt.reaction_indices,))
    payload: dict[str, object] = {
        "sample_idx": sample_index,
        f"sample/{sample_index}/path_length": PATH_LENGTH,
        f"sample/{sample_index}/ring_system": question.ring_system,
        f"sample/{sample_index}/ground_truth_count": len(gt_chains),
        f"sample/{sample_index}/ground_truth_full_count": full_dataset_chain_count(question),
        f"sample/{sample_index}/ground_truth_chains": format_task15_chains(list(gt_chains)),
        f"sample/{sample_index}/molecule_freq_cap": filters.molecule_freq_cap,
        f"sample/{sample_index}/frequent_molecule_count": len(filters.frequent_molecules),
        f"sample/{sample_index}/pred_reaction_indices": ",".join(str(x) for x in pred_rxns),
        f"sample/{sample_index}/selected_chain_count": sampling.selected_chain_count,
        f"sample/{sample_index}/forced_reaction_count": sampling.forced_count,
        f"sample/{sample_index}/support_indices_in_context": support_in_context,
        f"sample/{sample_index}/support_indices_selected_count": len(support_indices),
        f"sample/{sample_index}/support_indices_full_count": len(full_support_indices),
        f"sample/{sample_index}/response_char_count": len(response),
        f"sample/{sample_index}/validity_reason": scores["validity_reason"],
        f"sample/{sample_index}/is_correct": scores["is_correct"],
        f"sample/{sample_index}/valid_path": scores["valid_path"],
        f"sample/{sample_index}/index_match": scores["index_match"],
        f"sample/{sample_index}/objective_length_match": scores["objective_length_match"],
        f"sample/{sample_index}/reaction_precision": scores["reaction_precision"],
        f"sample/{sample_index}/reaction_recall": scores["reaction_recall"],
        f"sample/{sample_index}/reaction_f1": scores["reaction_f1"],
        f"sample/{sample_index}/normalized_lcs": scores["normalized_lcs"],
        f"sample/{sample_index}/completion_prompt_char_count": completion_prompt_char_count,
        f"sample/{sample_index}/context_size": context_size,
        f"sample/{sample_index}/context_coverage": context_coverage,
        f"sample/{sample_index}/retrieved_line_count": context_line_count,
        f"sample/{sample_index}/final_total_input_tokens": final_input_tokens,
        f"sample/{sample_index}/final_total_output_tokens": final_output_tokens,
        f"sample/{sample_index}/final_total_tokens": final_total_tokens,
        f"sample/{sample_index}/iterations": iterations,
    }
    if sample_cost_usd is not None:
        payload[f"sample/{sample_index}/final_total_cost_usd"] = sample_cost_usd
    return payload


def print_task15_run_summary(
    *,
    total: int,
    index_match_count: int,
    macro_accuracy: float,
    macro_valid_path: float,
    macro_objective_length: float,
    macro_reaction_f1: float,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Ring-system queries evaluated: {total}")
    print(f"Index match: {index_match_count}/{total}")
    print(f"Macro accuracy: {macro_accuracy:.4f}")
    print(f"Macro valid path: {macro_valid_path:.4f}")
    print(f"Macro objective-length match: {macro_objective_length:.4f}")
    print(f"Macro reaction F1: {macro_reaction_f1:.4f}")


def update_task15_run_summary(
    run,
    *,
    total: int,
    index_match_count: int,
    macro_accuracy: float,
    macro_valid_path: float,
    macro_objective_length: float,
    macro_reaction_f1: float,
    total_input_tokens: int,
    total_output_tokens: int,
    samples_with_cost: int,
    total_cost_usd: float,
) -> None:
    run.summary["queries_evaluated"] = total
    run.summary["index_match_correct"] = index_match_count
    run.summary["macro_accuracy"] = macro_accuracy
    run.summary["macro_valid_path"] = macro_valid_path
    run.summary["macro_objective_length_match"] = macro_objective_length
    run.summary["macro_reaction_f1"] = macro_reaction_f1
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
