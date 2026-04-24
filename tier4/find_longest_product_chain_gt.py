"""
Ground-truth script: find the longest reaction chain that produces a target product.

A chain is an ordered list of distinct reaction indices [r0, r1, ..., rk] such that
for each consecutive pair (ri, ri+1), at least one product of ri matches at least one
reactant of ri+1 by canonical SMILES equality.

This script builds a DAG product->reactant reaction graph, then:
1) finds reactions that directly produce the target product;
2) restricts to ancestor reactions that can reach those targets;
3) computes the longest chain ending at any target reaction:
   - O(V + E) exact DP on the ancestor-induced DAG.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque

from rdkit import Chem

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"


def parse_reaction_sides(indexed_line: str) -> tuple[str, str]:
    _, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    return parts[0].strip(), parts[-1].strip()


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


def build_product_reactant_graph(
    lines: list[str],
    max_molecule_freq: int,
    max_successors_per_reaction: int,
    dag_mode: str,
) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, list[str]]]:
    products_by_idx: dict[int, list[str]] = {}
    reactants_by_idx: dict[int, list[str]] = {}

    producers: dict[str, set[int]] = defaultdict(set)
    consumers: dict[str, set[int]] = defaultdict(set)

    for line in lines:
        idx_str, _ = line.split(" ", 1)
        idx = int(idx_str)
        try:
            reactant_side, product_side = parse_reaction_sides(line)
        except (ValueError, IndexError):
            continue

        r_canon = canonicalize_components(reactant_side)
        p_canon = canonicalize_components(product_side)
        if not r_canon or not p_canon:
            continue

        reactants_by_idx[idx] = r_canon
        products_by_idx[idx] = p_canon

        for smi in p_canon:
            producers[smi].add(idx)
        for smi in r_canon:
            consumers[smi].add(idx)

    frequent_molecules: set[str] = set()
    for smi, idxs in producers.items():
        if len(idxs) > max_molecule_freq:
            frequent_molecules.add(smi)
    for smi, idxs in consumers.items():
        if len(idxs) > max_molecule_freq:
            frequent_molecules.add(smi)

    successors: dict[int, set[int]] = defaultdict(set)
    predecessors: dict[int, set[int]] = defaultdict(set)
    for idx, prods in products_by_idx.items():
        for smi in prods:
            if smi in frequent_molecules:
                continue
            for consumer_idx in consumers.get(smi, set()):
                if consumer_idx == idx:
                    continue
                # Enforce DAG orientation deterministically by reaction index.
                if dag_mode == "index_asc" and idx >= consumer_idx:
                    continue
                if dag_mode == "index_desc" and idx <= consumer_idx:
                    continue
                successors[idx].add(consumer_idx)

        if len(successors[idx]) > max_successors_per_reaction:
            successors[idx] = set(sorted(successors[idx])[:max_successors_per_reaction])

        for nxt in successors[idx]:
            predecessors[nxt].add(idx)

    # Ensure all reaction indices with parsed products exist in dicts
    for idx in products_by_idx:
        successors.setdefault(idx, set())
        predecessors.setdefault(idx, set())

    return dict(successors), dict(predecessors), products_by_idx


def ancestor_closure(predecessors: dict[int, set[int]], targets: set[int]) -> set[int]:
    visited: set[int] = set(targets)
    q: deque[int] = deque(targets)
    while q:
        cur = q.popleft()
        for p in predecessors.get(cur, set()):
            if p not in visited:
                visited.add(p)
                q.append(p)
    return visited


def topo_order_if_dag(nodes: set[int], successors: dict[int, set[int]]) -> list[int] | None:
    indeg: dict[int, int] = {n: 0 for n in nodes}
    for u in nodes:
        for v in successors.get(u, set()):
            if v in nodes:
                indeg[v] += 1

    q: deque[int] = deque(sorted(n for n, d in indeg.items() if d == 0))
    order: list[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(successors.get(u, set())):
            if v not in indeg:
                continue
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(nodes):
        return None
    return order


def longest_path_dag(
    nodes: set[int],
    successors: dict[int, set[int]],
    predecessors: dict[int, set[int]],
    target_nodes: set[int],
) -> list[int]:
    order = topo_order_if_dag(nodes, successors)
    if order is None:
        raise ValueError("Graph is not DAG.")

    neg_inf = -10**12
    dist: dict[int, int] = {n: neg_inf for n in nodes}
    parent: dict[int, int | None] = {n: None for n in nodes}

    for n in order:
        preds = [p for p in predecessors.get(n, set()) if p in nodes]
        if not preds:
            dist[n] = 1
            parent[n] = None
            continue
        best_pred = max(preds, key=lambda p: dist[p])
        if dist[best_pred] > neg_inf:
            dist[n] = dist[best_pred] + 1
            parent[n] = best_pred

    best_target = max(target_nodes, key=lambda t: dist.get(t, neg_inf))
    if dist.get(best_target, neg_inf) <= neg_inf:
        return []

    chain_rev = []
    cur: int | None = best_target
    while cur is not None:
        chain_rev.append(cur)
        cur = parent[cur]
    chain_rev.reverse()
    return chain_rev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find longest reaction chain producing a target product."
    )
    parser.add_argument(
        "--target-product",
        type=str,
        required=True,
        help="Target product SMILES (single molecule).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DATASET_PATH,
        help=f"Path to USPTO reaction dataset (default: {DATASET_PATH}).",
    )
    parser.add_argument(
        "--max-molecule-freq",
        type=int,
        default=200,
        help="Ignore molecules appearing in >N reactions as reactant or product.",
    )
    parser.add_argument(
        "--max-successors-per-reaction",
        type=int,
        default=10_000,
        help="Cap outgoing edges per reaction after filtering frequent molecules.",
    )
    parser.add_argument(
        "--dag-mode",
        choices=["index_asc", "index_desc"],
        default="index_asc",
        help=(
            "How to orient edges to force a DAG. "
            "'index_asc' keeps i->j only when i<j; "
            "'index_desc' keeps i->j only when i>j."
        ),
    )
    parser.add_argument(
        "--show-reactions",
        action="store_true",
        help="Also print full reaction strings for chain indices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        target_canon = Chem.CanonSmiles(args.target_product)
    except Exception as exc:
        raise ValueError(f"Invalid --target-product SMILES: {args.target_product}") from exc

    print(f"Loading dataset from: {args.dataset_path}")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]
    print(f"Loaded {len(lines)} reactions")

    print("Building product->reactant graph ...")
    successors, predecessors, products_by_idx = build_product_reactant_graph(
        lines=lines,
        max_molecule_freq=args.max_molecule_freq,
        max_successors_per_reaction=args.max_successors_per_reaction,
        dag_mode=args.dag_mode,
    )
    edge_count = sum(len(v) for v in successors.values())
    print(f"Graph nodes={len(successors)} edges={edge_count}")

    target_reactions = {
        idx for idx, prods in products_by_idx.items() if target_canon in set(prods)
    }
    if not target_reactions:
        print(f"No reactions directly produce target product: {target_canon}")
        return

    print(
        f"Target product={target_canon} is produced by {len(target_reactions)} "
        f"reaction(s): {sorted(target_reactions)}"
    )

    ancestors = ancestor_closure(predecessors, target_reactions)
    print(f"Ancestor closure size (including targets): {len(ancestors)}")

    topo = topo_order_if_dag(ancestors, successors)
    if topo is None:
        raise RuntimeError(
            "Ancestor-induced graph is cyclic despite DAG construction mode. "
            "Please report this; it indicates a bug."
        )
    print(
        "Ancestor-induced subgraph is DAG (forced by graph construction) "
        "-> using exact O(V+E) DP."
    )
    chain = longest_path_dag(
        nodes=ancestors,
        successors=successors,
        predecessors=predecessors,
        target_nodes=target_reactions,
    )
    solver_mode = "dag_dp"

    if not chain:
        print("No valid chain found.")
        return

    print("\n=== Longest chain (ground truth) ===")
    print(f"solver_mode: {solver_mode}")
    print(f"chain_length (reactions): {len(chain)}")
    print(f"reaction_indices: {','.join(str(x) for x in chain)}")

    if args.show_reactions:
        print("\nReaction lines:")
        for idx in chain:
            print(lines[idx])


if __name__ == "__main__":
    main()

