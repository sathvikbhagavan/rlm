"""Scan length-5 chain counts per task16 target structural class."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rlm.codeact_helpers import load_lines, parse_reaction_sides
from task11_synthetic_chain_graph import canonicalize_components

RDLogger.DisableLog("rdApp.*")

CHAIN_LENGTH = 5
DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MAX_CHAINS_PER_TARGET = 500
MAX_CANDIDATES_PER_CLASS = 2000

TARGET_CLASSES: dict[str, dict[str, object]] = {
    "T1_tertiary_amide_aryl": {
        "smarts": "[#6:1][CX3](=O)[NX3]([#6])([#6])",
        "min_heavy": 20,
        "max_heavy": 50,
        "min_producers": 2,
    },
    "T2_biaryl_pyridine": {
        "smarts": "[#6]c1cccnc1",
        "min_heavy": 18,
        "max_heavy": 50,
        "min_producers": 2,
        "extra_smarts": "[#6]-[#6]",  # at least one C-C bond to another ring system
    },
    "T3_aryl_alkyl_ether": {
        "smarts": "[#6][OX2][CX4;!$(C=O)]",
        "min_heavy": 18,
        "max_heavy": 50,
        "min_producers": 2,
    },
    "T4_substituted_heterocycle": {
        "smarts": "[#6]~[#6]~[#6]~[#6]~[#6]",  # placeholder, use disjunction below
        "min_heavy": 20,
        "max_heavy": 50,
        "min_producers": 2,
        "ring_smarts": (
            "c1cc([NX3])nn1",  # pyrazole-like
            "c1cscn1",  # thiazole
            "c1cncnc1",  # pyrimidine
            "c1[nH]cnc1",  # imidazole
        ),
    },
    "T5_drug_like": {
        "smarts": "[NX3][CX3](=O)[#6]",
        "min_heavy": 25,
        "max_heavy": 45,
        "min_producers": 2,
        "extra_smarts": "c1cccnc1",  # heteroaryl present
    },
}


@dataclass(frozen=True)
class TargetStats:
    smiles: str
    producers: int
    chains: int
    example_chain: tuple[int, ...] | None


def build_graph(lines: list[str]) -> tuple[dict[int, set[int]], dict[str, set[int]], dict[str, int]]:
    predecessors: dict[int, set[int]] = defaultdict(set)
    product_to_rxns: dict[str, set[int]] = defaultdict(set)
    heavy_atoms: dict[str, int] = {}

    products_by_idx: dict[int, list[str]] = {}
    consumers: dict[str, set[int]] = defaultdict(set)

    for line in lines:
        idx = int(line.split(" ", 1)[0])
        try:
            reactant_side, product_side = parse_reaction_sides(line)
        except Exception:
            continue
        reactants = canonicalize_components(reactant_side)
        products = canonicalize_components(product_side)
        if not reactants or not products:
            continue
        products_by_idx[idx] = products
        for smi in reactants:
            consumers[smi].add(idx)
        for smi in products:
            product_to_rxns[smi].add(idx)
            if smi not in heavy_atoms:
                mol = Chem.MolFromSmiles(smi)
                heavy_atoms[smi] = mol.GetNumHeavyAtoms() if mol else 0

    for idx, products in products_by_idx.items():
        for smi in products:
            for consumer_idx in consumers.get(smi, set()):
                if consumer_idx > idx:
                    predecessors[consumer_idx].add(idx)

    return dict(predecessors), dict(product_to_rxns), heavy_atoms


def backward_chains(
    predecessors: dict[int, set[int]],
    terminal: int,
    *,
    chain_length: int = CHAIN_LENGTH,
    cap: int = MAX_CHAINS_PER_TARGET,
) -> list[tuple[int, ...]]:
    results: list[tuple[int, ...]] = []

    def dfs(chain: list[int]) -> None:
        if len(results) >= cap:
            return
        if len(chain) == chain_length:
            results.append(tuple(chain))
            return
        head = chain[0]
        for pred in sorted(predecessors.get(head, set()), reverse=True):
            if pred < head and pred not in chain:
                dfs([pred, *chain])

    dfs([terminal])
    return results


def matches_class(
    smiles: str,
    heavy_atoms: dict[str, int],
    spec: dict[str, object],
    patterns: dict[str, Chem.Mol],
) -> bool:
    heavy = heavy_atoms.get(smiles, 0)
    if heavy < int(spec["min_heavy"]) or heavy > int(spec["max_heavy"]):
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    class_key = spec.get("_key", "")
    if class_key == "T4_substituted_heterocycle":
        ring_smarts = spec.get("ring_smarts", ())
        if not any(mol.HasSubstructMatch(patterns[f"T4_ring_{i}"]) for i in range(len(ring_smarts))):  # type: ignore[arg-type]
            return False
        # at least two ring substituents: crude proxy = >=3 ring atoms matched + heavy >= min
        return heavy >= int(spec["min_heavy"])

    if not mol.HasSubstructMatch(patterns[class_key]):
        return False
    extra = spec.get("extra_smarts")
    if extra and not mol.HasSubstructMatch(patterns[f"{class_key}__extra"]):
        return False
    if class_key == "T2_biaryl_pyridine":
        # pyridine connected to another carbon (biaryl/heteroaryl-aryl)
        return mol.HasSubstructMatch(patterns["T2_biaryl_pyridine__biaryl"])
    return True


def compile_patterns() -> dict[str, Chem.Mol]:
    patterns: dict[str, Chem.Mol] = {}
    for key, spec in TARGET_CLASSES.items():
        spec = dict(spec)
        spec["_key"] = key
        TARGET_CLASSES[key] = spec
        smarts = str(spec["smarts"])
        if key != "T4_substituted_heterocycle":
            patterns[key] = Chem.MolFromSmarts(smarts)
        extra = spec.get("extra_smarts")
        if extra:
            patterns[f"{key}__extra"] = Chem.MolFromSmarts(str(extra))
        if key == "T2_biaryl_pyridine":
            patterns["T2_biaryl_pyridine__biaryl"] = Chem.MolFromSmarts("c1cccnc1-[#6]")
        ring_smarts = spec.get("ring_smarts")
        if ring_smarts:
            for i, rs in enumerate(ring_smarts):  # type: ignore[arg-type]
                patterns[f"T4_ring_{i}"] = Chem.MolFromSmarts(rs)
    return patterns


def scan_class(
    class_key: str,
    spec: dict[str, object],
    predecessors: dict[int, set[int]],
    product_to_rxns: dict[str, set[int]],
    heavy_atoms: dict[str, int],
    patterns: dict[str, Chem.Mol],
) -> tuple[int, list[TargetStats]]:
    spec = dict(spec)
    spec["_key"] = class_key
    min_producers = int(spec["min_producers"])

    candidates = [
        smi
        for smi in product_to_rxns
        if matches_class(smi, heavy_atoms, spec, patterns)
    ]
    candidates.sort(key=lambda s: (-len(product_to_rxns[s]), -heavy_atoms.get(s, 0)))

    stats: list[TargetStats] = []
    for smiles in candidates[:MAX_CANDIDATES_PER_CLASS]:
        producers = product_to_rxns[smiles]
        if len(producers) < min_producers:
            continue
        chains: set[tuple[int, ...]] = set()
        for terminal in sorted(producers):
            for chain in backward_chains(predecessors, terminal):
                chains.add(chain)
            if len(chains) >= MAX_CHAINS_PER_TARGET:
                break
        if chains:
            example = min(chains)
            stats.append(
                TargetStats(
                    smiles=smiles,
                    producers=len(producers),
                    chains=len(chains),
                    example_chain=example,
                )
            )

    stats.sort(key=lambda row: (-row.chains, -row.producers, -heavy_atoms.get(row.smiles, 0)))
    return len(candidates), stats


def main() -> None:
    lines = load_lines(DATASET_PATH)
    print(f"Loaded {len(lines)} reactions")
    predecessors, product_to_rxns, heavy_atoms = build_graph(lines)
    patterns = compile_patterns()

    print(
        f"\nScanning length-{CHAIN_LENGTH} chains "
        f"(max {MAX_CANDIDATES_PER_CLASS} candidates/class, "
        f"cap {MAX_CHAINS_PER_TARGET} chains/target)\n"
    )
    print(
        "class\tmatching_products\twith_chains\ttop_chains\ttop_producers\texample_smiles\texample_chain"
    )

    for class_key, spec in TARGET_CLASSES.items():
        match_count, stats = scan_class(
            class_key, spec, predecessors, product_to_rxns, heavy_atoms, patterns
        )
        with_chains = len(stats)
        if stats:
            top = stats[0]
            print(
                f"{class_key}\t{match_count}\t{with_chains}\t{top.chains}\t{top.producers}\t"
                f"{top.smiles}\t{list(top.example_chain)}"
            )
            for row in stats[1:5]:
                print(
                    f"  \t\t\t{row.chains}\t{row.producers}\t{row.smiles}\t{list(row.example_chain)}"
                )
            chain_counts = [s.chains for s in stats]
            print(
                f"  summary: targets_with_chains={with_chains} "
                f"median_chains={sorted(chain_counts)[len(chain_counts)//2]} "
                f"max_chains={max(chain_counts)} "
                f"total_chains_capped={sum(min(c, MAX_CHAINS_PER_TARGET) for c in chain_counts)}"
            )
        else:
            print(f"{class_key}\t{match_count}\t0\t-\t-\t-\t-")


if __name__ == "__main__":
    main()
