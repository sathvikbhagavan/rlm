"""Search for new orthogonal task16 targets (simpler / more chains first)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, RDLogger
from rlm.codeact_helpers import load_lines

from task16_truncated_synthesis_graph import (
    FULL_CHAIN_LENGTH,
    build_predecessor_graph,
    is_clean_synthesis_chain,
    mine_full_chains_for_target,
    parse_records_from_lines,
    prefixes_from_full_chains,
    terminal_indices_from_full_chains,
)

RDLogger.DisableLog("rdApp.*")

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
CHAINS_JSON = Path(__file__).with_name("task16_truncated_hardcoded_chains.json")

EXISTING_TARGETS = {
    "CN(C)C(=O)C(=O)c1c[nH]c2cccc(OCc3ccccc3)c12",
    "C=CC(=O)Nc1cccc(-c2cc(-c3ccncc3)cc3cncnc23)c1",
    "COc1ncc(-c2cc(N3CC(CC(=O)O)C(F)(F)C3)c3nccn3n2)c(OC)n1",
    "O=C1CCC(N2C(=O)c3ccc(N4CCC(CN5CCS(=O)(=NCC6CCNCC6)CC5)CC4)cc3C2=O)C(=O)N1",
    "Cc1ccc(NC(=O)c2cc(C(F)(F)F)ccn2)cc1C1=Cc2cnc(Nc3nccs3)nc2N2CCN=C12",
}

CLASSES: dict[str, dict[str, object]] = {
    "benzamide": {"smarts": "cC(=O)N", "min_heavy": 14, "max_heavy": 38},
    "sulfonamide": {"smarts": "c[S](=O)(=O)N", "min_heavy": 16, "max_heavy": 42},
    "morpholine_aryl": {"smarts": "cN1CCOCC1", "min_heavy": 16, "max_heavy": 42},
    "pyrazole_aryl": {"smarts": "c1nn[cH]c1", "min_heavy": 16, "max_heavy": 42},
    "piperidine_aryl": {"smarts": "cN1CCCCC1", "min_heavy": 16, "max_heavy": 42},
    "urea_aryl": {"smarts": "NC(=O)Nc", "min_heavy": 16, "max_heavy": 42},
    "thioether_aryl": {"smarts": "cSc", "min_heavy": 16, "max_heavy": 45},
    "quinazoline": {"smarts": "c1ccc2c(c1)ncnc2", "min_heavy": 16, "max_heavy": 42},
    "indazole": {"smarts": "c1ccc2[nH]ncc2c1", "min_heavy": 16, "max_heavy": 42},
    "aryl_ether": {"smarts": "cOC", "min_heavy": 14, "max_heavy": 38},
    "benzimidazole": {"smarts": "c1ccc2[nH]cnc2c1", "min_heavy": 16, "max_heavy": 42},
    "lactam": {"smarts": "C1NC(=O)CC1", "min_heavy": 14, "max_heavy": 38},
    "amide_secondary": {"smarts": "CNC(=O)", "min_heavy": 14, "max_heavy": 38},
    "pyridyl_amine": {"smarts": "cNC", "min_heavy": 14, "max_heavy": 38},
    "fluoro_aryl": {"smarts": "cF", "min_heavy": 14, "max_heavy": 40},
}


@dataclass(frozen=True)
class Candidate:
    class_key: str
    smiles: str
    heavy: int
    producers: int
    full_chains: int
    prefixes: int
    terminals: int
    prefix_support: int
    example_prefix: tuple[int, ...]
    overlap_existing: int


def mine_target_cached(
    target_smiles: str,
    records: dict,
    predecessors: dict,
    *,
    cap: int = 5000,
) -> list[tuple[int, ...]]:
    target = Chem.CanonSmiles(target_smiles)
    if not target:
        return []
    terminals = sorted(idx for idx, rec in records.items() if target in rec.products)
    chains: set[tuple[int, ...]] = set()
    from task16_truncated_synthesis_graph import backward_chains

    for terminal in terminals:
        for chain in backward_chains(predecessors, terminal, chain_length=FULL_CHAIN_LENGTH, cap=cap):
            if is_clean_synthesis_chain(chain, target_smiles, records):
                chains.add(chain)
            if len(chains) >= cap:
                break
        if len(chains) >= cap:
            break
    return sorted(chains)


def main() -> None:
    lines = load_lines(DATASET_PATH)
    print(f"Loaded {len(lines)} reactions; parsing once...")
    records = parse_records_from_lines(lines)
    predecessors = build_predecessor_graph(records)

    product_to_rxns: dict[str, set[int]] = defaultdict(set)
    heavy_atoms: dict[str, int] = {}
    for rec in records.values():
        for smi in rec.products:
            product_to_rxns[smi].add(rec.index)
            if smi not in heavy_atoms:
                mol = Chem.MolFromSmiles(smi)
                heavy_atoms[smi] = mol.GetNumHeavyAtoms() if mol else 0

    patterns = {k: Chem.MolFromSmarts(str(v["smarts"])) for k, v in CLASSES.items()}

    existing_support_by_q: dict[str, set[int]] = {}
    existing_prefixes: set[tuple[int, ...]] = set()
    if CHAINS_JSON.exists():
        data = json.loads(CHAINS_JSON.read_text(encoding="utf-8"))
        for qid, payload in data.items():
            support: set[int] = set()
            for prefix in payload["prefix_chains"]:
                support.update(prefix)
                existing_prefixes.add(tuple(prefix))
            existing_support_by_q[qid] = support

    candidates: list[Candidate] = []
    seen = set(EXISTING_TARGETS)

    def max_per_question_overlap(prefix: tuple[int, ...]) -> int:
        if not existing_support_by_q:
            return 0
        return max(len(set(prefix) & support) for support in existing_support_by_q.values())

    def is_orthogonal(prefixes: tuple[tuple[int, ...], ...], example: tuple[int, ...]) -> bool:
        if any(tuple(p) in existing_prefixes for p in prefixes):
            return False
        # Allow sparse overlap with any one existing question, but not a full shared prefix.
        if max_per_question_overlap(example) > 1:
            return False
        return True

    for class_key, spec in CLASSES.items():
        pattern = patterns[class_key]
        pool: list[tuple[int, int, str]] = []
        for smi, prods in product_to_rxns.items():
            if smi in seen:
                continue
            heavy = heavy_atoms.get(smi, 0)
            if heavy < int(spec["min_heavy"]) or heavy > int(spec["max_heavy"]):
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None or not mol.HasSubstructMatch(pattern):
                continue
            if len(prods) < 2:
                continue
            pool.append((len(prods), heavy, smi))
        pool.sort(reverse=True)
        print(f"\n{class_key}: matching_products={len(pool)}")

        found = 0
        for _, heavy, smi in pool[:500]:
            chains = mine_target_cached(smi, records, predecessors, cap=2000)
            if len(chains) < 8:
                continue
            prefixes = prefixes_from_full_chains(tuple(chains))
            if len(prefixes) < 8 or len(prefixes) > 400:
                continue
            example_prefix = prefixes[0]
            if not is_orthogonal(prefixes, example_prefix):
                continue
            terminals = terminal_indices_from_full_chains(tuple(chains), prefixes)
            prefix_support = len({idx for prefix in prefixes for idx in prefix})
            overlap = max_per_question_overlap(example_prefix)
            candidates.append(
                Candidate(
                    class_key=class_key,
                    smiles=smi,
                    heavy=heavy,
                    producers=len(product_to_rxns[smi]),
                    full_chains=len(chains),
                    prefixes=len(prefixes),
                    terminals=len(terminals),
                    prefix_support=prefix_support,
                    example_prefix=example_prefix,
                    overlap_existing=overlap,
                )
            )
            seen.add(smi)
            found += 1
            print(
                f"  + prefixes={len(prefixes)} chains={len(chains)} heavy={heavy} "
                f"terms={len(terminals)} overlap={overlap} {smi[:75]}"
            )
            if found >= 4:
                break

    candidates.sort(key=lambda c: (-c.prefixes, -c.full_chains, c.heavy, c.overlap_existing))

    print("\n=== TOP 25 ===")
    for c in candidates[:25]:
        print(
            f"{c.class_key:16} pref={c.prefixes:4} chains={c.full_chains:4} "
            f"h={c.heavy:2} terms={c.terminals} ov={c.overlap_existing} {c.smiles[:85]}"
        )

    picked: list[Candidate] = []
    used_classes: set[str] = set()
    for max_overlap in (0, 1):
        for c in candidates:
            if c.class_key in used_classes or c in picked:
                continue
            if c.overlap_existing > max_overlap:
                continue
            picked.append(c)
            used_classes.add(c.class_key)
            if len(picked) == 5:
                break
        if len(picked) == 5:
            break

    picked.sort(key=lambda c: (-c.prefixes, -c.full_chains))

    print("\n=== PICKED 5 (simple -> hard) ===")
    for i, c in enumerate(picked, start=1):
        print(
            f"{i}. {c.class_key} prefixes={c.prefixes} chains={c.full_chains} "
            f"heavy={c.heavy} terminals={c.terminals} overlap={c.overlap_existing}"
        )
        print(f"   smiles: {c.smiles}")
        print(f"   example_prefix: {list(c.example_prefix)}")


if __name__ == "__main__":
    main()
