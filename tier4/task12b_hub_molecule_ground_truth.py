"""Hardcoded hub-molecule ground truth for tier4 task12b.

Hub molecules were selected from reactionSmilesFigShareUSPTO2023_cleaned.txt using
exact canonical SMILES product-to-reactant linking with index-asc DAG ordering
(no global molecule-frequency filter). Eight non-overlapping hub molecules with
exactly three downstream consumer reactions each were clustered into one question.

Context sampling follows the tier3 ratio rule on the full support pool, then
selects the ceiling number of complete lexicographic hubs (1 producer + 3
consumers each). Protected filler sampling keeps random reactions from sharing
molecules with the selected support subgraph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TASK12B_TOTAL_REACTIONS = 122456
TASK12B_DAG_MODE = "index_asc"
TASK12B_MIN_DOWNSTREAM = 3
TASK12B_MIN_SELECTED_GROUND_TRUTH = 7
TASK12B_REACTIONS_PER_HUB = 4
TASK12B_GROUND_TRUTH_DEFINITION = (
    "canonical SMILES strings for molecules that appear as a product in exactly one "
    "reaction in the provided context and as a reactant in at least three distinct "
    "downstream consumer reactions with strictly higher reaction index (index_asc DAG); "
    "only molecules from reactions present in the provided context"
)

HARDCODED_GT_HUB_MOLECULES: tuple[str, ...] = (
    "CC(C)(C)OC(=O)N[C@H]1CN[C@H]2CC[C@@H]1C2",
    "CC(C)(C)OC(=O)Nc1cnc(Cl)c(F)c1",
    "CCc1cc(N2C(=S)N(c3cnc(C#N)c(C(F)(F)F)c3)C(=O)C23CCC3)ccc1OCCBr",
    "COC(=O)c1cc(O)c(Cl)c([N+](=O)[O-])c1",
    "Cc1ccnc2[nH]c(C(=O)Cl)cc12",
    "Cc1cnc(Nc2cn[nH]c2)nc1-c1ccc(C(=O)NCC#N)cc1",
    "N#Cc1cncc2[nH]c(C(=O)O)cc12",
    "O=C(O)c1cc2c(Cl)c(Cl)ncc2[nH]1",
)

# Lexicographic hub order; each hub has one producer and three downstream consumers.
HUB_SUPPORT: tuple[tuple[str, frozenset[int]], ...] = (
    (
        "CC(C)(C)OC(=O)N[C@H]1CN[C@H]2CC[C@@H]1C2",
        frozenset({426, 466, 470, 490}),
    ),
    (
        "CC(C)(C)OC(=O)Nc1cnc(Cl)c(F)c1",
        frozenset({525, 714, 735, 81730}),
    ),
    (
        "CCc1cc(N2C(=S)N(c3cnc(C#N)c(C(F)(F)F)c3)C(=O)C23CCC3)ccc1OCCBr",
        frozenset({928, 929, 953, 954}),
    ),
    (
        "COC(=O)c1cc(O)c(Cl)c([N+](=O)[O-])c1",
        frozenset({240, 242, 18946, 18949}),
    ),
    (
        "Cc1ccnc2[nH]c(C(=O)Cl)cc12",
        frozenset({501, 502, 508, 509}),
    ),
    (
        "Cc1cnc(Nc2cn[nH]c2)nc1-c1ccc(C(=O)NCC#N)cc1",
        frozenset({1158, 1159, 1196, 121625}),
    ),
    (
        "N#Cc1cncc2[nH]c(C(=O)O)cc12",
        frozenset({585, 586, 587, 588}),
    ),
    (
        "O=C(O)c1cc2c(Cl)c(Cl)ncc2[nH]1",
        frozenset({717, 718, 741, 742}),
    ),
)

SUPPORT_INDICES: frozenset[int] = frozenset(
    idx for _, hub_indices in HUB_SUPPORT for idx in hub_indices
)


def support_indices() -> set[int]:
    return set(SUPPORT_INDICES)


@dataclass(frozen=True)
class Task12bContextSampling:
    selected_hub_smiles: tuple[str, ...]
    support_indices: frozenset[int]
    forced_count: int
    selected_hub_count: int


def tier3_forced_reaction_count(
    answer_count: int,
    context_size: int,
    *,
    dataset_size: int = TASK12B_TOTAL_REACTIONS,
    min_selected_ground_truth: int = TASK12B_MIN_SELECTED_GROUND_TRUTH,
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


def hubs_for_context_sampling(context_size: int) -> Task12bContextSampling:
    """Select complete hubs and support indices for tier3-style context sampling."""
    full_support = support_indices()
    forced_count = tier3_forced_reaction_count(len(full_support), context_size)
    selected_hub_count = min(
        max(1, math.ceil(forced_count / TASK12B_REACTIONS_PER_HUB)),
        len(HUB_SUPPORT),
    )
    selected_hubs = HUB_SUPPORT[:selected_hub_count]
    selected_hub_smiles = tuple(smi for smi, _ in selected_hubs)
    support = frozenset(idx for _, hub_indices in selected_hubs for idx in hub_indices)
    return Task12bContextSampling(
        selected_hub_smiles=selected_hub_smiles,
        support_indices=support,
        forced_count=forced_count,
        selected_hub_count=selected_hub_count,
    )
