"""Hardcoded functional-group chain ground truth for tier4 task13.

Full-dataset 7-reaction chains were mined from reactionSmilesFigShareUSPTO2023_cleaned.txt
with molecule_freq_cap=200 at 122456 reactions,
heavy-atom window [3, 90], and RDKit SMARTS FG detection.

All accepted chains live in task13_fg_hardcoded_chains.json. At evaluation time,
in-context GT is the subset of those chains present in the sampled context that also
pass context-local hub and heavy-atom filters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from task13_fg_chain_graph import (
    DATASET_TOTAL_REACTIONS,
    MAX_HEAVY_ATOMS,
    MAX_MOLECULE_FREQ_REFERENCE,
    MIN_HEAVY_ATOMS,
    MIN_LOCAL_MOLECULE_FREQ,
    PATH_LENGTH,
)

TASK13_TOTAL_REACTIONS = DATASET_TOTAL_REACTIONS
TASK13_MIN_SELECTED_GROUND_TRUTH = PATH_LENGTH
TASK13_GROUND_TRUTH_DEFINITION = (
    "all reaction chains of exactly 7 reactions converting source functional group to target "
    "functional group via exact canonical-SMILES product-to-reactant links between consecutive "
    "reactions, with FG detection by RDKit SMARTS; full-dataset hub filter "
    f"(>{MAX_MOLECULE_FREQ_REFERENCE} at {DATASET_TOTAL_REACTIONS} reactions); context-local hub "
    f"filter scaled from that reference with floor {MIN_LOCAL_MOLECULE_FREQ}; heavy-atom window "
    f"[{MIN_HEAVY_ATOMS}, {MAX_HEAVY_ATOMS}]; only reactions present in context"
)

FIXED_FG_QUERIES: list[tuple[str, str]] = [
    # ("primary_alcohol", "tertiary_amide"),
    ("primary_alcohol", "carboxylic_acid"),
    ("alkyl_halide", "tertiary_amine"),
    ("ester", "tertiary_amide"),
    ("nitrile", "primary_amide"),
]

HARDCODED_CHAINS_JSON = Path(__file__).with_name("task13_fg_hardcoded_chains.json")

HARDCODED_GT_CHAIN_COUNTS: dict[tuple[str, str], int] = {
    ('primary_alcohol', 'tertiary_amide'): 6571,
    ('primary_alcohol', 'carboxylic_acid'): 550,
    ('alkyl_halide', 'tertiary_amine'): 2069,
    ('ester', 'tertiary_amide'): 5378,
    ('nitrile', 'primary_amide'): 1201,
}

HARDCODED_GT_EXAMPLE: dict[tuple[str, str], tuple[int, ...]] = {
    ('primary_alcohol', 'tertiary_amide'): (1392, 1393, 1394, 1395, 1396, 1397, 1398),
    ('primary_alcohol', 'carboxylic_acid'): (1388, 1389, 1390, 1391, 1392, 1393, 1394),
    ('alkyl_halide', 'tertiary_amine'): (1393, 1394, 1395, 1396, 1397, 1398, 1399),
    ('ester', 'tertiary_amide'): (2852, 2954, 2868, 2869, 2872, 2873, 2874),
    ('nitrile', 'primary_amide'): (5051, 5092, 5111, 5112, 5113, 5114, 5115),
}


@dataclass(frozen=True)
class FgQuestion:
    source_fg: str
    target_fg: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.source_fg, self.target_fg)


FIXED_QUESTIONS: list[FgQuestion] = [
    FgQuestion(source_fg, target_fg) for source_fg, target_fg in FIXED_FG_QUERIES
]


def fg_pair_key(source_fg: str, target_fg: str) -> str:
    return f"{source_fg}->{target_fg}"


@lru_cache(maxsize=1)
def _load_mined_payload() -> dict[str, dict[str, object]]:
    with HARDCODED_CHAINS_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hardcoded_chains_for_pair(source_fg: str, target_fg: str) -> tuple[tuple[int, ...], ...]:
    payload = _load_mined_payload()[fg_pair_key(source_fg, target_fg)]
    return tuple(tuple(chain) for chain in payload["chains"])


def full_support_indices_for_question(question: FgQuestion) -> set[int]:
    return {
        idx
        for chain in hardcoded_chains_for_pair(question.source_fg, question.target_fg)
        for idx in chain
    }


@dataclass(frozen=True)
class Task13ContextSampling:
    selected_chains: tuple[tuple[int, ...], ...]
    support_indices: frozenset[int]
    forced_count: int
    selected_chain_count: int


def tier3_forced_reaction_count(
    answer_count: int,
    context_size: int,
    *,
    dataset_size: int = TASK13_TOTAL_REACTIONS,
    min_selected_ground_truth: int = TASK13_MIN_SELECTED_GROUND_TRUTH,
) -> int:
    """Mirror RandomContextPipeline forced-count logic for the support pool."""
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


def chains_for_context_sampling(
    question: FgQuestion,
    context_size: int,
) -> Task13ContextSampling:
    """Select complete chains and support indices for tier3-style context sampling."""
    all_chains = hardcoded_chains_for_pair(question.source_fg, question.target_fg)
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

    if not selected_chains:
        selected_chains = [all_chains[0]]
        support_acc = set(all_chains[0])

    selected = tuple(selected_chains)
    support_indices = frozenset(support_acc)
    return Task13ContextSampling(
        selected_chains=selected,
        support_indices=support_indices,
        forced_count=forced_count,
        selected_chain_count=len(selected),
    )


def example_chain_for_question(question: FgQuestion) -> tuple[int, ...]:
    chain = HARDCODED_GT_EXAMPLE.get(question.key)
    if chain is None:
        raise KeyError(f"No hardcoded example chain for {question.key}")
    if len(chain) != PATH_LENGTH:
        raise ValueError(
            f"Expected chain length {PATH_LENGTH}, got {len(chain)} for {question.key}"
        )
    return chain


def support_indices_for_question(question: FgQuestion) -> set[int]:
    return full_support_indices_for_question(question)


def full_dataset_chain_count(question: FgQuestion) -> int:
    return HARDCODED_GT_CHAIN_COUNTS[question.key]


def print_task13_startup_banner() -> None:
    for question in FIXED_QUESTIONS:
        example = example_chain_for_question(question)
        print(
            f"Ground truth [7-reaction {question.source_fg}->{question.target_fg}] "
            f"full_dataset_example={list(example)} "
            f"support_indices={len(support_indices_for_question(question))}"
        )


def print_task13_sample_context(
    *,
    sample_index: int,
    question: FgQuestion,
    gt,
    sampling: Task13ContextSampling,
    full_support_indices: set[int],
    support_indices: set[int],
    support_in_context: int,
    context_size: int,
    context_line_count: int,
    context_coverage: float,
    filters,
) -> None:
    print(
        f"\nQuestion {sample_index + 1}/{len(FIXED_QUESTIONS)}: "
        f"{question.source_fg}->{question.target_fg}, "
        f"gt_len={len(gt.reaction_indices)}, "
        f"gt_chains={len(gt.accepted_reaction_indices)}"
    )
    print(
        f"[CONTEXT] requested_size={context_size} actual_size={context_line_count} "
        f"selected_chains={sampling.selected_chain_count}/{full_dataset_chain_count(question)} "
        f"forced_count={sampling.forced_count} "
        f"support_in_context={support_in_context}/{len(support_indices)} "
        f"full_support={len(full_support_indices)} "
        f"in_context_chains={len(gt.accepted_reaction_indices)}/{sampling.selected_chain_count} "
        f"molecule_freq_cap={filters.molecule_freq_cap} "
        f"frequent_molecules={len(filters.frequent_molecules)} "
        f"coverage={context_coverage:.4f}"
    )


def format_task13_chains(chains: list[tuple[int, ...]], limit: int = 20) -> str:
    shown = [",".join(str(x) for x in chain) for chain in chains[:limit]]
    if len(chains) > limit:
        shown.append(f"... (+{len(chains) - limit} more)")
    return " | ".join(shown)


def print_task13_sample_metrics(
    *,
    sample_index: int,
    question: FgQuestion,
    response: str,
    parsed_chains: list[tuple[int, ...]],
    gt_chains: list[tuple[int, ...]],
    scores: dict[str, object],
) -> None:
    print(f"Response [sample={sample_index}]: {response[:500]}{'…' if len(response) > 500 else ''}")
    print(
        f"Predicted [{question.source_fg}->{question.target_fg}]: "
        f"{scores['valid_chain_count']}/{scores['parsed_chain_count']} valid chains"
    )
    print(f"Ground truth [{question.source_fg}->{question.target_fg}]: {len(gt_chains)} chains")
    print(
        f"Metrics [sample={sample_index}] -> precision={scores['precision']:.4f} "
        f"recall={scores['recall']:.4f} f1={scores['f1']:.4f} "
        f"exact_match={bool(scores['is_exact_match'])}"
    )


def build_task13_wandb_sample_log(
    *,
    sample_index: int,
    question: FgQuestion,
    gt,
    sampling: Task13ContextSampling,
    full_support_indices: set[int],
    support_indices: set[int],
    support_in_context: int,
    filters,
    parsed_chains: list[tuple[int, ...]],
    gt_chains: list[tuple[int, ...]],
    scores: dict[str, object],
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
    payload: dict[str, object] = {
        "sample_idx": sample_index,
        f"sample/{sample_index}/path_length": PATH_LENGTH,
        f"sample/{sample_index}/source_fg": question.source_fg,
        f"sample/{sample_index}/target_fg": question.target_fg,
        f"sample/{sample_index}/ground_truth_count": len(gt_chains),
        f"sample/{sample_index}/ground_truth_full_count": full_dataset_chain_count(question),
        f"sample/{sample_index}/ground_truth_chains": format_task13_chains(gt_chains),
        f"sample/{sample_index}/molecule_freq_cap": filters.molecule_freq_cap,
        f"sample/{sample_index}/frequent_molecule_count": len(filters.frequent_molecules),
        f"sample/{sample_index}/response_parsed_count": scores["parsed_chain_count"],
        f"sample/{sample_index}/response_valid_count": scores["valid_chain_count"],
        f"sample/{sample_index}/response_parsed_chains": format_task13_chains(parsed_chains),
        f"sample/{sample_index}/selected_chain_count": sampling.selected_chain_count,
        f"sample/{sample_index}/forced_reaction_count": sampling.forced_count,
        f"sample/{sample_index}/support_indices_in_context": support_in_context,
        f"sample/{sample_index}/support_indices_selected_count": len(support_indices),
        f"sample/{sample_index}/support_indices_full_count": len(full_support_indices),
        f"sample/{sample_index}/response_char_count": len(response),
        f"sample/{sample_index}/validity_reason": scores["validity_reason"],
        f"sample/{sample_index}/precision": scores["precision"],
        f"sample/{sample_index}/recall": scores["recall"],
        f"sample/{sample_index}/f1": scores["f1"],
        f"sample/{sample_index}/is_exact_match": scores["is_exact_match"],
        f"sample/{sample_index}/invalid_chain_count": scores["invalid_chain_count"],
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


def print_task13_run_summary(
    *,
    total: int,
    exact_match_count: int,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Queries evaluated: {total}")
    print(f"Exact match: {exact_match_count}/{total}")
    print(f"Exact match accuracy: {exact_match_count / total if total else 0.0:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


def update_task13_run_summary(
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
    run.summary["queries_evaluated"] = total
    run.summary["exact_match_correct"] = exact_match_count
    run.summary["exact_match_accuracy"] = exact_match_count / total if total else 0.0
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
