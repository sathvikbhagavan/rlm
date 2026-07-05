"""Context-local synthetic-chain graph helpers for tier4 task11."""

from __future__ import annotations

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


def build_successor_graph(lines: list[str]) -> dict[int, set[int]]:
    """Build product-to-reactant edges among reactions present in `lines`."""
    products_by_idx: dict[int, list[str]] = {}
    consumers: dict[str, set[int]] = defaultdict(set)
    valid_indices: set[int] = set()

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

        products_by_idx[idx] = products
        valid_indices.add(idx)
        for smi in reactants:
            consumers[smi].add(idx)

    successors: dict[int, set[int]] = defaultdict(set)
    for idx, products in products_by_idx.items():
        for smi in products:
            for consumer_idx in consumers.get(smi, set()):
                if consumer_idx != idx and consumer_idx in valid_indices:
                    successors[idx].add(consumer_idx)

    return dict(successors)


def find_synthetic_chains(
    successors: dict[int, set[int]],
    start_index: int,
    chain_length: int,
    *,
    cap: int = 10_000,
) -> list[tuple[int, ...]]:
    """Enumerate all chains of exactly `chain_length` starting at `start_index`."""
    results: list[tuple[int, ...]] = []

    def _dfs(path: list[int]) -> None:
        if len(results) >= cap:
            return
        if len(path) == chain_length:
            results.append(tuple(path))
            return
        for nxt in sorted(successors.get(path[-1], set())):
            if nxt not in path:
                path.append(nxt)
                _dfs(path)
                path.pop()
                if len(results) >= cap:
                    return

    if start_index not in successors and chain_length > 1:
        if chain_length == 1:
            return [(start_index,)]
        return []

    _dfs([start_index])
    return results


def ground_truth_chains_in_context(
    lines: list[str],
    start_index: int,
    chain_length: int,
) -> list[tuple[int, ...]]:
    """Return chains valid within the reactions provided in `lines`."""
    context_indices = {
        int(line.split(" ", 1)[0])
        for line in lines
        if line.strip() and " " in line and line.split(" ", 1)[0].isdigit()
    }
    if start_index not in context_indices:
        return []

    successors = build_successor_graph(lines)
    chains = find_synthetic_chains(successors, start_index, chain_length)
    return [chain for chain in chains if all(idx in context_indices for idx in chain)]
