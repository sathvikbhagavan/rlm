"""
Find 5 starting reaction indices with small, exact ground truth for length-2 and length-3 chains.
Builds the graph without a successor cap so counts are exact.
"""

import random
from collections import defaultdict
from rdkit import Chem

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
MAX_MOLECULE_FREQ = 200
MAX_SUCCESSORS_UNCAPPED = 10_000  # effectively no cap
MIN_L2_CHAINS = 10                # minimum length-2 chains
MAX_L2_CHAINS = 20                # maximum length-2 chains (keep output tractable)
MAX_L3_CHAINS = 30                # only pick starts with this many length-3 chains or fewer
NUM_CANDIDATES = 5
SEED = 42


def parse_reaction_sides(indexed_line: str) -> tuple[str, str]:
    _, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    return parts[0].strip(), parts[-1].strip()


def canonicalize_components(smiles_str: str) -> list[str]:
    canonical = []
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


def build_graph_uncapped(lines: list[str]) -> dict[int, set[int]]:
    products_by_idx: dict[int, list[str]] = {}
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
        products_by_idx[idx] = p_canon
        for smi in p_canon:
            producers[smi].add(idx)
        for smi in r_canon:
            consumers[smi].add(idx)

    frequent: set[str] = set()
    for smi, idxs in producers.items():
        if len(idxs) > MAX_MOLECULE_FREQ:
            frequent.add(smi)
    for smi, idxs in consumers.items():
        if len(idxs) > MAX_MOLECULE_FREQ:
            frequent.add(smi)

    successors: dict[int, set[int]] = defaultdict(set)
    for idx, prods in products_by_idx.items():
        for smi in prods:
            if smi in frequent:
                continue
            for consumer_idx in consumers.get(smi, set()):
                if consumer_idx != idx:
                    successors[idx].add(consumer_idx)

    return dict(successors)


def find_chains(successors: dict[int, set[int]], start: int, length: int) -> list[tuple[int, ...]]:
    results: list[tuple[int, ...]] = []

    def _dfs(path: list[int]) -> None:
        if len(path) == length:
            results.append(tuple(path))
            return
        for nxt in sorted(successors.get(path[-1], set())):
            if nxt not in path:
                path.append(nxt)
                _dfs(path)
                path.pop()

    _dfs([start])
    return results


def main() -> None:
    rng = random.Random(SEED)

    print("Loading dataset ...")
    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]
    print(f"Loaded {len(lines)} reactions")

    print("Building uncapped product→reactant graph ...")
    successors = build_graph_uncapped(lines)
    print(f"Nodes with ≥1 successor: {sum(1 for s in successors.values() if s)}")

    # Shuffle candidates and scan
    all_starts = list(successors.keys())
    rng.shuffle(all_starts)

    results = []
    print(f"\nSearching for starts with l2∈[{MIN_L2_CHAINS},{MAX_L2_CHAINS}] and l3∈[0,{MAX_L3_CHAINS}] ...")
    for start in all_starts:
        if len(successors.get(start, set())) == 0:
            continue

        l2 = find_chains(successors, start, 2)
        if not (MIN_L2_CHAINS <= len(l2) <= MAX_L2_CHAINS):
            continue

        l3 = find_chains(successors, start, 3)
        if len(l3) > MAX_L3_CHAINS:
            continue

        # Verify no intermediate node was uncapped-trimmed
        # (redundant here since graph is uncapped, but good to log)
        max_successor_count = max(
            len(successors.get(b, set())) for (_, b) in l2
        ) if l2 else 0

        results.append({
            "start": start,
            "l2_chains": l2,
            "l3_chains": l3,
            "max_intermediate_successors": max_successor_count,
        })
        print(f"  Found: start={start}  l2={len(l2)}  l3={len(l3)}  max_B_successors={max_successor_count}")

        if len(results) >= NUM_CANDIDATES:
            break

    print(f"\n{'=' * 65}")
    print(f"{'Index':<10} {'L2 chains':<12} {'L3 chains':<12} L2 ground truth")
    print("-" * 65)
    for r in results:
        l2_str = " | ".join(",".join(str(x) for x in c) for c in r["l2_chains"])
        print(f"  {r['start']:<8} {len(r['l2_chains']):<12} {len(r['l3_chains']):<12} {l2_str}")

    print(f"\nL3 ground truth per start:")
    for r in results:
        if r["l3_chains"]:
            l3_str = " | ".join(",".join(str(x) for x in c) for c in r["l3_chains"])
            print(f"  start={r['start']}: {l3_str}")
        else:
            print(f"  start={r['start']}: (no length-3 chains)")


if __name__ == "__main__":
    main()
