"""Context-local longest-chain graph helpers for tier4 task12."""

from __future__ import annotations

from collections import defaultdict, deque

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


def build_dag_graph(
    lines: list[str],
    *,
    dag_mode: str = "index_asc",
) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, list[str]]]:
    """Build a product-to-reactant DAG among reactions present in `lines`."""
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
    predecessors: dict[int, set[int]] = defaultdict(set)
    for idx, products in products_by_idx.items():
        for smi in products:
            for consumer_idx in consumers.get(smi, set()):
                if consumer_idx == idx or consumer_idx not in valid_indices:
                    continue
                if dag_mode == "index_asc" and idx >= consumer_idx:
                    continue
                if dag_mode == "index_desc" and idx <= consumer_idx:
                    continue
                successors[idx].add(consumer_idx)
                predecessors[consumer_idx].add(idx)

    for idx in products_by_idx:
        successors.setdefault(idx, set())
        predecessors.setdefault(idx, set())

    return dict(successors), dict(predecessors), products_by_idx


def _ancestor_closure(predecessors: dict[int, set[int]], targets: set[int]) -> set[int]:
    visited = set(targets)
    queue: deque[int] = deque(targets)
    while queue:
        current = queue.popleft()
        for predecessor in predecessors.get(current, set()):
            if predecessor not in visited:
                visited.add(predecessor)
                queue.append(predecessor)
    return visited


def _topo_order_if_dag(nodes: set[int], successors: dict[int, set[int]]) -> list[int] | None:
    indegree = {node: 0 for node in nodes}
    for node in nodes:
        for successor in successors.get(node, set()):
            if successor in nodes:
                indegree[successor] += 1

    queue: deque[int] = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for successor in sorted(successors.get(node, set())):
            if successor not in indegree:
                continue
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)

    if len(order) != len(nodes):
        return None
    return order


def _is_better_chain(candidate: tuple[int, ...], current: tuple[int, ...]) -> bool:
    if len(candidate) > len(current):
        return True
    if len(candidate) == len(current) and candidate < current:
        return True
    return False


def longest_chain_for_target(
    lines: list[str],
    target_product_smiles: str,
    *,
    dag_mode: str = "index_asc",
) -> tuple[int, ...]:
    """Return the longest DAG chain producing `target_product_smiles` within `lines`."""
    try:
        target_canon = Chem.CanonSmiles(target_product_smiles)
    except Exception:
        return tuple()

    successors, predecessors, products_by_idx = build_dag_graph(lines, dag_mode=dag_mode)
    target_reactions = {
        idx for idx, products in products_by_idx.items() if target_canon in products
    }
    if not target_reactions:
        return tuple()

    ancestors = _ancestor_closure(predecessors, target_reactions)
    order = _topo_order_if_dag(ancestors, successors)
    if order is None:
        return tuple()

    best_chain_by_node: dict[int, tuple[int, ...]] = {}
    for node in order:
        predecessors_in_ancestors = [
            predecessor
            for predecessor in predecessors.get(node, set())
            if predecessor in ancestors
        ]
        if not predecessors_in_ancestors:
            best_chain_by_node[node] = (node,)
            continue

        best_chain = tuple()
        for predecessor in predecessors_in_ancestors:
            predecessor_chain = best_chain_by_node.get(predecessor)
            if predecessor_chain is None:
                continue
            candidate = predecessor_chain + (node,)
            if _is_better_chain(candidate, best_chain):
                best_chain = candidate
        if best_chain:
            best_chain_by_node[node] = best_chain

    best_chain = tuple()
    for target in target_reactions:
        if target not in ancestors:
            continue
        candidate = best_chain_by_node.get(target, (target,))
        if _is_better_chain(candidate, best_chain):
            best_chain = candidate

    return best_chain


def ground_truth_longest_chain_in_context(
    lines: list[str],
    target_product_smiles: str,
    *,
    dag_mode: str = "index_asc",
) -> tuple[int, ...]:
    """Return the longest valid chain for `target_product_smiles` in `lines`."""
    context_indices = {
        int(line.split(" ", 1)[0])
        for line in lines
        if line.strip() and " " in line and line.split(" ", 1)[0].isdigit()
    }
    if not context_indices:
        return tuple()

    chain = longest_chain_for_target(
        lines,
        target_product_smiles,
        dag_mode=dag_mode,
    )
    if not chain or not all(idx in context_indices for idx in chain):
        return tuple()
    return chain
