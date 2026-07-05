"""Hardcoded synthetic-chain ground truth for tier4 task11.

Chains were selected from reactionSmilesFigShareUSPTO2023_cleaned.txt using
exact canonical SMILES product-to-reactant linking (no global molecule-frequency
filter). Starts were chosen so each question has 10 valid chains with a compact
set of supporting reaction indices.

Context sampling follows the tier3 ratio rule on the full support pool, then
selects the ceiling number of complete lexicographic chains whose reaction
union is passed to the random context pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TASK11_TOTAL_REACTIONS = 122456
TASK11_MIN_SELECTED_GROUND_TRUTH = 7
TASK11_GROUND_TRUTH_DEFINITION = (
    "ordered distinct reaction indices [r_0, ..., r_{L-1}] with r_0 = start, "
    "each consecutive pair linked by exact canonical SMILES equality between "
    "at least one product of r_k and one reactant of r_{k+1}, considering only "
    "reactions present in the provided context"
)

# One question each for chain lengths 3 and 4.
FIXED_QUESTIONS: list[tuple[int, int]] = [
    (11605, 3),
    (23326, 4),
]

HARDCODED_GT_CHAINS: dict[tuple[int, int], list[tuple[int, ...]]] = {
    (11605, 3): [
        (11605, 10983, 13481),
        (11605, 11220, 12017),
        (11605, 11575, 13530),
        (11605, 12206, 11859),
        (11605, 12290, 10911),
        (11605, 12543, 12758),
        (11605, 13099, 13075),
        (11605, 13371, 11606),
        (11605, 13829, 12045),
        (11605, 13976, 12658),
    ],
    (23326, 4): [
        (23326, 60156, 60157, 60158),
        (23326, 83532, 83503, 83504),
        (23326, 83532, 83515, 83516),
        (23326, 83532, 83518, 83519),
        (23326, 83532, 83524, 83525),
        (23326, 83532, 83529, 83530),
        (23326, 83532, 83533, 83534),
        (23326, 98117, 98118, 98119),
        (23326, 98449, 98118, 98119),
        (23326, 117034, 117035, 117036),
    ],
}


def chain_indices_for_question(start_index: int, chain_length: int) -> set[int]:
    key = (start_index, chain_length)
    chains = HARDCODED_GT_CHAINS.get(key)
    if chains is None:
        raise KeyError(f"No hardcoded chains for start={start_index}, length={chain_length}")
    return {idx for chain in chains for idx in chain}


@dataclass(frozen=True)
class Task11ContextSampling:
    selected_chains: tuple[tuple[int, ...], ...]
    support_indices: frozenset[int]
    forced_count: int
    selected_chain_count: int


def tier3_forced_reaction_count(
    answer_count: int,
    context_size: int,
    *,
    dataset_size: int = TASK11_TOTAL_REACTIONS,
    min_selected_ground_truth: int = TASK11_MIN_SELECTED_GROUND_TRUTH,
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
    start_index: int,
    chain_length: int,
    context_size: int,
) -> Task11ContextSampling:
    """Select complete chains and support indices for tier3-style context sampling."""
    key = (start_index, chain_length)
    all_chains = HARDCODED_GT_CHAINS[key]
    full_support = chain_indices_for_question(start_index, chain_length)
    forced_count = tier3_forced_reaction_count(len(full_support), context_size)
    selected_chain_count = min(
        max(1, math.ceil(forced_count / chain_length)),
        len(all_chains),
    )
    selected_chains = tuple(all_chains[:selected_chain_count])
    support_indices = frozenset(idx for chain in selected_chains for idx in chain)
    return Task11ContextSampling(
        selected_chains=selected_chains,
        support_indices=support_indices,
        forced_count=forced_count,
        selected_chain_count=selected_chain_count,
    )
