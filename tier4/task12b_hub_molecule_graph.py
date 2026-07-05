"""Context-local hub-molecule graph helpers for tier4 task12b."""

from __future__ import annotations

import random
from collections import defaultdict

from rdkit import Chem

from rlm.codeact_helpers import parse_reaction_sides


def canonicalize_components(smiles_str: str) -> list[str]:
    canonical: list[str] = []
    for smi in smiles_str.split("."):
        smi = smi.strip()
        if not smi:
            continue
        try:
            csmi = Chem.CanonSmiles(smi)
            if csmi:
                canonical.append(csmi)
        except Exception:
            continue
    return canonical


def canonicalize_smiles(smiles: str) -> str | None:
    try:
        return Chem.CanonSmiles(smiles.strip())
    except Exception:
        return None


def build_producer_consumer_maps(
    lines: list[str],
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    producers: dict[str, set[int]] = defaultdict(set)
    consumers: dict[str, set[int]] = defaultdict(set)

    for line in lines:
        idx_str, _ = line.split(" ", 1)
        idx = int(idx_str)
        try:
            reactant_side, product_side = parse_reaction_sides(line)
        except (ValueError, IndexError):
            continue

        reactants = canonicalize_components(reactant_side)
        products = canonicalize_components(product_side)
        if not reactants or not products:
            continue

        for smi in reactants:
            consumers[smi].add(idx)
        for smi in products:
            producers[smi].add(idx)

    return dict(producers), dict(consumers)


def molecules_in_lines(lines: list[str]) -> set[str]:
    molecules: set[str] = set()
    for line in lines:
        try:
            reactant_side, product_side = parse_reaction_sides(line)
        except (ValueError, IndexError):
            continue
        molecules.update(canonicalize_components(reactant_side))
        molecules.update(canonicalize_components(product_side))
    return molecules


def hub_molecules_in_context(
    lines: list[str],
    *,
    min_downstream: int = 3,
    dag_mode: str = "index_asc",
) -> list[str]:
    """Return canonical SMILES for hub molecules present in `lines`."""
    producers, consumers = build_producer_consumer_maps(lines)
    hits: list[str] = []

    for smi, producer_set in producers.items():
        if len(producer_set) != 1:
            continue
        producer_idx = next(iter(producer_set))
        downstream = {idx for idx in consumers.get(smi, set()) if idx != producer_idx}
        if dag_mode == "index_asc":
            downstream = {idx for idx in downstream if idx > producer_idx}
        elif dag_mode == "index_desc":
            downstream = {idx for idx in downstream if idx < producer_idx}
        if len(downstream) >= min_downstream:
            hits.append(smi)

    return sorted(hits)


def ground_truth_hub_molecules_in_context(
    lines: list[str],
    *,
    min_downstream: int = 3,
    dag_mode: str = "index_asc",
) -> list[str]:
    """Return hub molecules valid within the reactions provided in `lines`."""
    context_indices = {
        int(line.split(" ", 1)[0])
        for line in lines
        if line.strip() and " " in line and line.split(" ", 1)[0].isdigit()
    }
    if not context_indices:
        return []

    return hub_molecules_in_context(
        lines,
        min_downstream=min_downstream,
        dag_mode=dag_mode,
    )


def build_protected_hub_context(
    lines: list[str],
    support_indices: set[int],
    *,
    context_size: int,
    rng: random.Random,
) -> str:
    """Build context with forced support reactions and no molecule overlap in fillers."""
    line_by_idx = {int(line.split(" ", 1)[0]): line for line in lines}
    support_lines = [line_by_idx[idx] for idx in sorted(support_indices) if idx in line_by_idx]
    protected_molecules = molecules_in_lines(support_lines)

    forced = sorted(support_indices)
    forced_set = set(forced)
    excluded: set[int] = set()
    for line in lines:
        idx = int(line.split(" ", 1)[0])
        if idx in forced_set:
            continue
        try:
            reactant_side, product_side = parse_reaction_sides(line)
        except (ValueError, IndexError):
            continue
        reaction_molecules = set(canonicalize_components(reactant_side)) | set(
            canonicalize_components(product_side)
        )
        if reaction_molecules & protected_molecules:
            excluded.add(idx)

    if context_size < 0:
        selected = sorted(forced_set | ({i for i in range(len(lines))} - forced_set - excluded))
        return "\n".join(line_by_idx[idx] for idx in selected if idx in line_by_idx)

    top_k = min(context_size, len(lines))
    remainder_pool = [
        idx for idx in range(len(lines)) if idx not in forced_set and idx not in excluded
    ]
    random_take = min(top_k - len(forced), len(remainder_pool))
    random_indices = rng.sample(remainder_pool, k=random_take)
    sampled = forced + random_indices
    rng.shuffle(sampled)
    return "\n".join(line_by_idx[idx] for idx in sampled)
