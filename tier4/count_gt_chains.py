"""
Count ground truth synthetic chains in the dataset for lengths 2, 3, 4, 5.
Reuses the graph-building logic from rlm_task11.py without running any LLM.

For length 2: exact count (= graph edges).
For lengths 3-5: DFS per start node, capped at MAX_CHAINS_PER_START to stay tractable.

Usage:
    python count_gt_chains.py
"""

from collections import defaultdict
from rdkit import Chem

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
MAX_MOLECULE_FREQ = 200
MAX_SUCCESSORS_PER_REACTION = 50
# Per-start cap for lengths ≥ 3 (mirrors rlm_task11.py MAX_CHAINS_PER_START)
MAX_CHAINS_PER_START = 100
CHAIN_LENGTHS = [2, 3, 4, 5]


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


def build_product_reactant_graph(lines: list[str]) -> dict[int, set[int]]:
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

    print(f"  Frequent (filtered) molecules : {len(frequent)}")

    successors: dict[int, set[int]] = defaultdict(set)
    for idx, prods in products_by_idx.items():
        for smi in prods:
            if smi in frequent:
                continue
            for consumer_idx in consumers.get(smi, set()):
                if consumer_idx != idx:
                    successors[idx].add(consumer_idx)
        if len(successors[idx]) > MAX_SUCCESSORS_PER_REACTION:
            successors[idx] = set(sorted(successors[idx])[:MAX_SUCCESSORS_PER_REACTION])

    return dict(successors)


def count_chains_from(
    successors: dict[int, set[int]],
    start: int,
    chain_length: int,
    cap: int,
) -> int:
    """DFS count of chains of exactly `chain_length` starting from `start`, capped at `cap`."""
    count = 0

    def _dfs(path: list[int]) -> None:
        nonlocal count
        if count >= cap:
            return
        if len(path) == chain_length:
            count += 1
            return
        for nxt in sorted(successors.get(path[-1], set())):
            if nxt not in path:
                path.append(nxt)
                _dfs(path)
                path.pop()
                if count >= cap:
                    return

    _dfs([start])
    return count


def percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {k: 0 for k in ("p25", "p50", "p75", "p99", "max")}
    s = sorted(values)
    n = len(s)
    return {
        "p25": s[n // 4],
        "p50": s[n // 2],
        "p75": s[3 * n // 4],
        "p99": s[int(n * 0.99)],
        "max": s[-1],
    }


def main() -> None:
    print(f"Loading dataset from {DATASET_PATH} ...")
    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]
    print(f"Loaded {len(lines)} reactions")

    print("Building product→reactant graph ...")
    successors = build_product_reactant_graph(lines)

    all_starts = sorted(successors.keys())
    print(f"  Reactions with ≥1 successor    : {len(all_starts)}")

    print(f"\n{'=' * 65}")
    print(f"{'Chain length':<14} {'Starts with ≥1 chain':>22} {'Total chains (capped)':>22}")
    print(f"{'-' * 65}")

    for length in CHAIN_LENGTHS:
        cap = MAX_CHAINS_PER_START if length >= 3 else 10**9  # no cap for length-2
        chain_counts: list[int] = []
        starts_with_chain = 0
        total_chains = 0

        for start in all_starts:
            n = count_chains_from(successors, start, length, cap)
            if n > 0:
                starts_with_chain += 1
                total_chains += n
                chain_counts.append(n)

        p = percentiles(chain_counts)
        cap_note = f" (capped at {cap}/start)" if length >= 3 else ""
        print(f"  {length:<12} {starts_with_chain:>22,} {total_chains:>22,}{cap_note}")
        if chain_counts:
            print(
                f"  {'':12}   chains/start: "
                f"p25={p['p25']}  median={p['p50']}  p75={p['p75']}  "
                f"p99={p['p99']}  max={p['max']}"
            )
        print()

    print(f"MAX_MOLECULE_FREQ={MAX_MOLECULE_FREQ}, MAX_SUCCESSORS_PER_REACTION={MAX_SUCCESSORS_PER_REACTION}")


if __name__ == "__main__":
    main()
