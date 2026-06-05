"""Generate hardcoded ground-truth mappings for tier2 tasks.

Usage:
  /home/bhagavan/rlms/.rlm/bin/python generate_hardcoded_ground_truth.py
"""

from __future__ import annotations

from pathlib import Path
from pprint import pformat
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
TASK2_THRESHOLDS = [150, 184, 218, 252, 286, 320]
TASK3_THRESHOLDS = [1, 2, 3, 4, 5]
TASK4_THRESHOLDS = [1, 2, 3, 4, 5]
TASK5_WEIGHT_THRESHOLDS_DA = [100, 150]
TASK5_RING_X_VALUES = [1, 2]


def parse_reaction_sides(line: str) -> tuple[str, str]:
    parts = line.split(">")
    return parts[0].strip(), parts[-1].strip()


def heaviest_component_weight(side_smiles: str) -> Optional[float]:
    if not side_smiles:
        return None
    weights: list[float] = []
    for comp in side_smiles.split("."):
        comp = comp.strip()
        if not comp:
            continue
        mol = Chem.MolFromSmiles(comp)
        if mol is None:
            continue
        weights.append(Descriptors.MolWt(mol))
    if not weights:
        return None
    return max(weights)


def component_ring_counts(side_smiles: str) -> list[int]:
    if not side_smiles:
        return []
    ring_counts: list[int] = []
    for comp in side_smiles.split("."):
        comp = comp.strip()
        if not comp:
            continue
        mol = Chem.MolFromSmiles(comp)
        if mol is None:
            continue
        ring_counts.append(int(rdMolDescriptors.CalcNumRings(mol)))
    return ring_counts


def reaction_delta_weight(line: str) -> Optional[float]:
    reactants, products = parse_reaction_sides(line)
    heaviest_reactant = heaviest_component_weight(reactants)
    heaviest_product = heaviest_component_weight(products)
    if heaviest_reactant is None or heaviest_product is None:
        return None
    return heaviest_product - heaviest_reactant


def reaction_delta_rings(line: str) -> Optional[int]:
    reactants, products = parse_reaction_sides(line)
    reactant_ring_counts = component_ring_counts(reactants)
    product_ring_counts = component_ring_counts(products)
    if not reactant_ring_counts or not product_ring_counts:
        return None
    pairwise_deltas = [
        product_rings - reactant_rings
        for product_rings in product_ring_counts
        for reactant_rings in reactant_ring_counts
    ]
    return max(pairwise_deltas)


def component_aromatic_ring_counts(side_smiles: str) -> list[int]:
    if not side_smiles:
        return []
    aromatic_ring_counts: list[int] = []
    for comp in side_smiles.split("."):
        comp = comp.strip()
        if not comp:
            continue
        mol = Chem.MolFromSmiles(comp)
        if mol is None:
            continue
        aromatic_ring_counts.append(int(rdMolDescriptors.CalcNumAromaticRings(mol)))
    return aromatic_ring_counts


def reaction_delta_aromatic_rings(line: str) -> Optional[int]:
    reactants, products = parse_reaction_sides(line)
    reactant_counts = component_aromatic_ring_counts(reactants)
    product_counts = component_aromatic_ring_counts(products)
    if not reactant_counts or not product_counts:
        return None
    aromatic_rings_reactants = sum(reactant_counts)
    aromatic_rings_products = sum(product_counts)
    return aromatic_rings_products - aromatic_rings_reactants


def load_raw_lines(dataset_path: str = DATASET_PATH) -> list[str]:
    with open(dataset_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def compute_task2_gt(
    raw_lines: list[str], thresholds: list[int] = TASK2_THRESHOLDS
) -> dict[int, list[int]]:
    delta_by_index: dict[int, float] = {}
    for idx, line in enumerate(raw_lines):
        delta = reaction_delta_weight(line)
        if delta is not None:
            delta_by_index[idx] = delta
    return {
        threshold: [idx for idx, delta in delta_by_index.items() if delta > threshold]
        for threshold in thresholds
    }


def compute_task3_gt(
    raw_lines: list[str], thresholds: list[int] = TASK3_THRESHOLDS
) -> dict[int, list[int]]:
    delta_by_index: dict[int, int] = {}
    for idx, line in enumerate(raw_lines):
        delta = reaction_delta_rings(line)
        if delta is not None:
            delta_by_index[idx] = delta
    return {
        threshold: [idx for idx, delta in delta_by_index.items() if delta == threshold]
        for threshold in thresholds
    }


def compute_task4_gt(
    raw_lines: list[str], thresholds: list[int] = TASK4_THRESHOLDS
) -> dict[int, list[int]]:
    delta_by_index: dict[int, int] = {}
    for idx, line in enumerate(raw_lines):
        delta = reaction_delta_aromatic_rings(line)
        if delta is not None:
            delta_by_index[idx] = delta
    return {
        threshold: [idx for idx, delta in delta_by_index.items() if delta == threshold]
        for threshold in thresholds
    }


def compute_task5_gt(
    raw_lines: list[str],
    weight_thresholds_da: list[int] = TASK5_WEIGHT_THRESHOLDS_DA,
    ring_x_values: list[int] = TASK5_RING_X_VALUES,
) -> dict[tuple[int, int], list[int]]:
    weight_delta_by_index: dict[int, float] = {}
    ring_delta_by_index: dict[int, int] = {}
    for idx, line in enumerate(raw_lines):
        weight_delta = reaction_delta_weight(line)
        ring_delta = reaction_delta_rings(line)
        if weight_delta is not None:
            weight_delta_by_index[idx] = weight_delta
        if ring_delta is not None:
            ring_delta_by_index[idx] = ring_delta

    mapping: dict[tuple[int, int], list[int]] = {}
    for weight_threshold in weight_thresholds_da:
        for ring_x in ring_x_values:
            mapping[(weight_threshold, ring_x)] = [
                idx
                for idx, weight_delta in weight_delta_by_index.items()
                if idx in ring_delta_by_index
                and weight_delta > weight_threshold
                and ring_delta_by_index[idx] == ring_x
            ]
    return mapping


def write_mapping_module(
    out_path: Path,
    *,
    doc_title: str,
    thresholds_var_name: str,
    mapping_var_name: str,
    thresholds: list[int],
    mapping: dict[int, list[int]],
) -> None:
    with open(out_path, "w") as f:
        f.write(f'"""{doc_title}\n')
        f.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit extraction.\n"
        )
        f.write('"""\n\n')
        f.write(f"{thresholds_var_name} = {pformat(thresholds, width=100)}\n\n")
        f.write(f"{mapping_var_name} = ")
        f.write(pformat(mapping, width=100))
        f.write("\n")


def main() -> None:
    raw_lines = load_raw_lines()
    tier2_dir = Path(__file__).parent

    task2_mapping = compute_task2_gt(raw_lines)
    task2_path = tier2_dir / "task2_hardcoded_ground_truth.py"
    write_mapping_module(
        task2_path,
        doc_title="Hardcoded ground-truth indices for tier2 task2.",
        thresholds_var_name="TASK2_THRESHOLDS",
        mapping_var_name="TASK2_HARDCODED_GROUND_TRUTH_INDICES",
        thresholds=TASK2_THRESHOLDS,
        mapping=task2_mapping,
    )

    task3_mapping = compute_task3_gt(raw_lines)
    task3_path = tier2_dir / "task3_hardcoded_ground_truth.py"
    write_mapping_module(
        task3_path,
        doc_title="Hardcoded ground-truth indices for tier2 task3.",
        thresholds_var_name="TASK3_THRESHOLDS",
        mapping_var_name="TASK3_HARDCODED_GROUND_TRUTH_INDICES",
        thresholds=TASK3_THRESHOLDS,
        mapping=task3_mapping,
    )

    task4_mapping = compute_task4_gt(raw_lines)
    task4_path = tier2_dir / "task4_hardcoded_ground_truth.py"
    write_mapping_module(
        task4_path,
        doc_title="Hardcoded ground-truth indices for tier2 task4.",
        thresholds_var_name="TASK4_THRESHOLDS",
        mapping_var_name="TASK4_HARDCODED_GROUND_TRUTH_INDICES",
        thresholds=TASK4_THRESHOLDS,
        mapping=task4_mapping,
    )

    task5_mapping = compute_task5_gt(raw_lines)
    task5_path = tier2_dir / "task5_hardcoded_ground_truth.py"
    with open(task5_path, "w") as f:
        f.write('"""Hardcoded ground-truth indices for tier2 task5.\n')
        f.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit extraction.\n"
        )
        f.write('"""\n\n')
        f.write(
            f"TASK5_WEIGHT_THRESHOLDS_DA = {pformat(TASK5_WEIGHT_THRESHOLDS_DA, width=100)}\n"
        )
        f.write(f"TASK5_RING_X_VALUES = {pformat(TASK5_RING_X_VALUES, width=100)}\n\n")
        f.write("TASK5_HARDCODED_GROUND_TRUTH_INDICES = ")
        f.write(pformat(task5_mapping, width=100))
        f.write("\n")

    print(f"Wrote {task2_path}")
    print(f"Wrote {task3_path}")
    print(f"Wrote {task4_path}")
    print(f"Wrote {task5_path}")
    for t in TASK2_THRESHOLDS:
        print(f"task2 threshold {t}: {len(task2_mapping[t])}")
    for t in TASK3_THRESHOLDS:
        print(f"task3 threshold {t}: {len(task3_mapping[t])}")
    for t in TASK4_THRESHOLDS:
        print(f"task4 threshold {t}: {len(task4_mapping[t])}")
    for weight_threshold in TASK5_WEIGHT_THRESHOLDS_DA:
        for ring_x in TASK5_RING_X_VALUES:
            print(
                f"task5 weight>{weight_threshold} & ring=={ring_x}: "
                f"{len(task5_mapping[(weight_threshold, ring_x)])}"
            )


if __name__ == "__main__":
    main()
