"""
For length-2 chains, find which molecule (shared between product of A and reactant of B)
is responsible for the most chain-start reactions.

Usage:
    python top_linking_molecules.py
"""

from collections import defaultdict
from rdkit import Chem

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
MAX_MOLECULE_FREQ = 200
MAX_SUCCESSORS_PER_REACTION = 50
TOP_N = 20


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


def main() -> None:
    print(f"Loading dataset ...")
    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]
    print(f"Loaded {len(lines)} reactions")

    producers: dict[str, set[int]] = defaultdict(set)  # molecule -> reactions that produce it
    consumers: dict[str, set[int]] = defaultdict(set)  # molecule -> reactions that consume it

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
    print(f"Frequent (filtered) molecules: {len(frequent)}")

    # For each non-frequent molecule, count how many unique start reactions it enables
    # and how many total (A, B) pairs it creates
    molecule_starts: dict[str, set[int]] = {}
    molecule_pairs: dict[str, int] = {}

    for smi in producers:
        if smi in frequent:
            continue
        prod_set = producers[smi]
        cons_set = consumers.get(smi, set())
        # A is a valid start if it produces this molecule AND some other reaction consumes it
        starts = {a for a in prod_set if cons_set - {a}}
        if not starts:
            continue
        # Count pairs respecting the per-start successor cap
        pairs = sum(
            min(len(cons_set - {a}), MAX_SUCCESSORS_PER_REACTION)
            for a in prod_set
        )
        molecule_starts[smi] = starts
        molecule_pairs[smi] = pairs

    # Sort by number of start reactions
    ranked = sorted(molecule_starts.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"\nTop {TOP_N} linking molecules by number of start reactions (length-2 chains):")
    print(f"\n{'Rank':<6} {'Starts':>8} {'Pairs':>10}  SMILES")
    print("-" * 80)
    for rank, (smi, starts) in enumerate(ranked[:TOP_N], 1):
        pairs = molecule_pairs[smi]
        print(f"  {rank:<4} {len(starts):>8,} {pairs:>10,}  {smi}")


if __name__ == "__main__":
    main()
