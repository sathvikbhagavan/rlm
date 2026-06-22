"""Generate hardcoded ground-truth mappings for tier3 tasks.

Usage:
  /home/bhagavan/rlms/.rlm/bin/python generate_hardcoded_ground_truth.py
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations
from pathlib import Path
from pprint import pformat

from rdkit import Chem
from rdkit.Chem import rdChemReactions


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
PYRIDINE_SMARTS = "c1ccncc1"
QUINOLINE_SMARTS = "c1ccc2ncccc2c1"
TASK17_RING_SYSTEMS: dict[str, str] = {
    "indole": "c1ccc2[nH]ccc2c1",
    "benzofuran": "c1ccc2occc2c1",
    "benzothiazole": "c1ccc2scnc2c1",
    "benzimidazole": "c1ccc2[nH]cnc2c1",
}
TASK6_AMIDE_COUPLING_SMIRKS: dict[str, str] = {
    "acyl_chloride_with_primary_amine": "[*:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[Cl].[#7;H2;!$(N[O,N]);D1;+0:5]>>[*:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[#7;H1;D2;+0:5]",
    "acyl_chloride_with_secondary_amine": "[*:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[Cl].[#7;H1;D2;+0:5]>>[*:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[#7;H0;D3;+0:5]",
    "carboxylic_acid_with_primary_amine": "[CX3;+0:2](=[O;H0;D1;+0:3])-[O;H1;D1;+0].[#7;H2;D1;+0:5]>>[CX3;+0:2](=[O;H0;D1;+0:3])-[#7;H1;D2;+0:5]",
    "ester_with_primary_amine": "[#6:1]-[C;H0;D3;+0:2](=[O;!$(OC(C)(C)C);H0;D1;+0:3])-[O;H0;D2;+0].[#7;H2;D1;+0:5]>>[#6:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[#7;H1;D2;+0:5]",
    "ester_with_secondary_amine": "[#6:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[O;!$(OC(C)(C)C);H0;D2;+0].[#7;H1;D2;+0:5]>>[#6:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[#7;H0;D3;+0:5]",
}
TASK7_TO_FG_SMIRKS: dict[str, str] = {
    "grignard_ketone_to_tertiary_alcohol": "[#6;+0:1]-[Mg]-[Br,I,Cl].[*:2]-[C;H0;D3;+0:3](=[O;H0;D1;+0:4])-[#6;+0:5]>>[*:2]-[C;H0;D4;+0:3](-[O;H1;D1;+0:4])(-[#6;+0:5])-[#6;+0:1]",
    "grignard_aldehyde_to_secondary_alcohol": "[#6;+0:1]-[Mg]-[Br,I,Cl].[*:2]-[C;H1;D2;+0:3](=[O;H0;D1;+0:4])>>[*:2]-[C;H1;D3;+0:3](-[O;H1;D1;+0:4])-[#6;+0:1]",
    "nitrile_to_amine": "[#6;+0:0]-[C;H0;D2;+0:1]#[N;H0;D1;+0:2]>>[#6;+0:0]-[C;H2;D2;+0:1]-[N;H2;D1;+0:2]",
    "nitro_groups_to_amines": "[N;H0;D3;+1:1](=[O;H0;D1;+0])-[O;H0;D1;-1]>>[N;H2;D1;+0:1]",
    "alcohol_to_azide": "[#6;+0:0]-[O;H1;D1;+0]>>[#6;+0:0]-[N;H0;D2;+0]=[N;H0;D2;+1]=[N;H0;D1;-1]",
    "alcohol_to_carboxylic_acid": "[C;H2;D2;+0:1]-[O;H1;D1;+0:2]>>[C;H0;D3;+0:1](=[O;H0;D1;+0:2])-[O;H1;D1;+0]",
}
TASK8_BOC_SMIRKS: dict[str, str] = {
    "boc_primary_amine_deprotection": "[C;H3;D1;+0]-[C;H0;D4;+0](-[C;H3;D1;+0])(-[C;H3;D1;+0])-[O;H0;D2;+0]-[C;H0;D3;+0](=[O;H0;D1;+0])-[#7;H1;+0:1]>>[#7;H2;+0:1]",
    "boc_secondary_amine_deprotection": "[C;H3;D1;+0]-[C;H0;D4;+0](-[C;H3;D1;+0])(-[C;H3;D1;+0])-[O;H0;D2;+0]-[C;H0;D3;+0](=[O;H0;D1;+0])-[#7;H0;+0:1]>>[#7;H1;+0:1]",
    "boc_amine_protection_of_secondary_amine": "[C;H3;D1;+0:1]-[C;H0;D4;+0:2](-[C;H3;D1;+0:3])(-[C;H3;D1;+0:4])-[O;H0;D2;+0:5]-[C;H0;D3;+0:6](=[O;H0;D1;+0:7])-[O;H0;D2;+0].[#7;H1;+0:8]>>[C;H3;D1;+0:1]-[C;H0;D4;+0:2](-[C;H3;D1;+0:3])(-[C;H3;D1;+0:4])-[O;H0;D2;+0:5]-[C;H0;D3;+0:6](=[O;H0;D1;+0:7])-[#7;H0;+0:8]",
    "boc_amine_protection_of_primary_amine": "[C;H3;D1;+0:1]-[C;H0;D4;+0:2](-[C;H3;D1;+0:3])(-[C;H3;D1;+0:4])-[O;H0;D2;+0:5]-[C;H0;D3;+0:6](=[O;H0;D1;+0:7])-[O;H0;D2;+0].[#7;H2;+0:8]>>[C;H3;D1;+0:1]-[C;H0;D4;+0:2](-[C;H3;D1;+0:3])(-[C;H3;D1;+0:4])-[O;H0;D2;+0:5]-[C;H0;D3;+0:6](=[O;H0;D1;+0:7])-[#7;H1;+0:8]",
}
TASK9_NAMED_REACTIONS_SMIRKS: dict[str, str] = {
    "suzuki_coupling_with_boronic_acids": "[#6;$([#6]:[#6]),$([#6]=[#6]),$([#6]#[#6]);+0:1]-[B;H0;D3;+0](-[O;H1;D1;+0])-[O;H1;D1;+0].[#6;$([#6]=[#6]),$([#6]~[#6]:[#6]),$([#6]~n);+0:2][Cl,Br,I]>>[#6;$([#6]:[#6]),$([#6]=[#6]),$([#6]#[#6]);+0:1]-[#6;$([#6]=[#6]),$([#6]~[#6]:[#6]),$([#6]~n);+0:2]",
    "mitsunobu_sulfonamide": "[C;H1&$(C([#6])[#6]),H2&$(C[#6]):1][OH1].[NH1;$(N([#6])S(=O)=O):2]>>[C:1][N:2]",
    "sonogashira_coupling_terminal_alkyne_with_aryl_halide": "[c:0]-[Cl,Br,I].[#6;H0;D2;+0:1]#[C;H1;D1;+0:2]>>[c:0]-[#6;H0;D2;+0:1]#[C;H1;D1;+0:2]",
    "buchwald_hartwig_n_arylation_primary_amine": "[c;H0;D3;+0:0]-[F,Cl,Br,I].[#6;+0:1]-[N;H2;D1;+0:2]>>[c;H0;D3;+0:0]-[N;H1;D2;+0:2]-[#6;+0:1]",
    "stille_reaction_aryl": "[C;H2,H3;+0]-[Sn;H0;D4;+0](-[C;H2,H3;+0])(-[C;H2,H3;+0])-[c;H0;D3;+0:0].[#6;+0:2]-[F,Cl,Br,I]>>[#6;+0:2]-[c;H0;D3;+0:0]",
    "wittig_with_phosphonium": "[#6:1]-[#6;+0:2](=O).[P;+1]-[C;H2;D2;+0:3]-[*:4]>>[#6:1]-[#6;+0:2]=[C;H1;D2;+0:3]-[*:4]",
}


def load_indexed_lines(dataset_path: str = DATASET_PATH) -> list[str]:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        raw_lines = [line.strip() for line in handle if line.strip()]
    return [f"{i} {line}" for i, line in enumerate(raw_lines)]


def parse_reaction_sides(indexed_line: str) -> tuple[int, list[str], list[str]]:
    idx_str, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    if len(parts) != 3:
        raise ValueError("Reaction must have reactants>reagents>products format.")
    reactant_smiles = [s for s in parts[0].split(".") if s]
    product_smiles = [s for s in parts[2].split(".") if s]
    return int(idx_str), reactant_smiles, product_smiles


def mols_from_smiles(smiles_list: list[str]) -> list[Chem.Mol]:
    mols: list[Chem.Mol] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        mols.append(mol)
    return mols


def count_cn_connections(mols: list[Chem.Mol]) -> int:
    count = 0
    for mol in mols:
        for bond in mol.GetBonds():
            atomic_nums = {
                bond.GetBeginAtom().GetAtomicNum(),
                bond.GetEndAtom().GetAtomicNum(),
            }
            if atomic_nums == {6, 7}:
                count += 1
    return count


def compute_task11_gt(lines: list[str]) -> tuple[list[int], int, int]:
    indices: list[int] = []
    valid_reactions = 0
    skipped_reactions = 0
    for line in lines:
        try:
            idx, reactant_smiles, product_smiles = parse_reaction_sides(line)
            reactant_mols = mols_from_smiles(reactant_smiles)
            product_mols = mols_from_smiles(product_smiles)
            reactant_cn_count = count_cn_connections(reactant_mols)
            product_cn_count = count_cn_connections(product_mols)
            delta = product_cn_count - reactant_cn_count
            valid_reactions += 1
            if delta > 0:
                indices.append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    indices.sort()
    return indices, valid_reactions, skipped_reactions


def bond_multiset(mols: list[Chem.Mol]) -> Counter[tuple[int, int, Chem.BondType]]:
    counts: Counter[tuple[int, int, Chem.BondType]] = Counter()
    for mol in mols:
        for bond in mol.GetBonds():
            begin_atomic_num = bond.GetBeginAtom().GetAtomicNum()
            end_atomic_num = bond.GetEndAtom().GetAtomicNum()
            key = (
                min(begin_atomic_num, end_atomic_num),
                max(begin_atomic_num, end_atomic_num),
                bond.GetBondType(),
            )
            counts[key] += 1
    return counts


def breaks_at_least_one_co_bond(reactant_mols: list[Chem.Mol], product_mols: list[Chem.Mol]) -> bool:
    bond_difference = bond_multiset(reactant_mols) - bond_multiset(product_mols)
    for (atom_a, atom_b, _), count in bond_difference.items():
        if count > 0 and {atom_a, atom_b} == {6, 8}:
            return True
    return False


def count_nitrogen_atoms(mols: list[Chem.Mol]) -> int:
    count = 0
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                count += 1
    return count


def compute_task13_gt(lines: list[str]) -> tuple[list[int], int, int]:
    indices: list[int] = []
    valid_reactions = 0
    skipped_reactions = 0
    for line in lines:
        try:
            idx, reactant_smiles, product_smiles = parse_reaction_sides(line)
            reactant_mols = mols_from_smiles(reactant_smiles)
            product_mols = mols_from_smiles(product_smiles)
            reactant_n_count = count_nitrogen_atoms(reactant_mols)
            product_n_count = count_nitrogen_atoms(product_mols)
            valid_reactions += 1
            if product_n_count > reactant_n_count:
                indices.append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    indices.sort()
    return indices, valid_reactions, skipped_reactions


def compute_task12_gt(lines: list[str]) -> tuple[list[int], int, int]:
    indices: list[int] = []
    valid_reactions = 0
    skipped_reactions = 0
    for line in lines:
        try:
            idx, reactant_smiles, product_smiles = parse_reaction_sides(line)
            reactant_mols = mols_from_smiles(reactant_smiles)
            product_mols = mols_from_smiles(product_smiles)
            valid_reactions += 1
            if breaks_at_least_one_co_bond(reactant_mols, product_mols):
                indices.append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    indices.sort()
    return indices, valid_reactions, skipped_reactions


def write_task11_module(
    out_path: Path,
    indices: list[int],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task11.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit extraction.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK11_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK11_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK11_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(f"TASK11_POSITIVE_REACTIONS = {len(indices)}\n\n")
        handle.write("TASK11_HARDCODED_GROUND_TRUTH_INDICES = ")
        handle.write(pformat(indices, width=100))
        handle.write("\n")


def is_cc_bond_key(key: tuple[int, int, Chem.BondType]) -> bool:
    atom_a, atom_b, _ = key
    return atom_a == 6 and atom_b == 6


def cc_bond_change_counts(
    reactant_mols: list[Chem.Mol], product_mols: list[Chem.Mol]
) -> tuple[int, int]:
    reactant_bonds = bond_multiset(reactant_mols)
    product_bonds = bond_multiset(product_mols)
    formed = product_bonds - reactant_bonds
    broken = reactant_bonds - product_bonds
    cc_formed_count = sum(
        count for key, count in formed.items() if is_cc_bond_key(key) and count > 0
    )
    cc_broken_count = sum(
        count for key, count in broken.items() if is_cc_bond_key(key) and count > 0
    )
    return cc_formed_count, cc_broken_count


def matches_exactly_one_cc_formed_zero_broken(
    reactant_mols: list[Chem.Mol], product_mols: list[Chem.Mol]
) -> bool:
    cc_formed_count, cc_broken_count = cc_bond_change_counts(reactant_mols, product_mols)
    return cc_formed_count == 1 and cc_broken_count == 0


def compute_task15_gt(lines: list[str]) -> tuple[list[int], int, int]:
    indices: list[int] = []
    valid_reactions = 0
    skipped_reactions = 0
    for line in lines:
        try:
            idx, reactant_smiles, product_smiles = parse_reaction_sides(line)
            reactant_mols = mols_from_smiles(reactant_smiles)
            product_mols = mols_from_smiles(product_smiles)
            valid_reactions += 1
            if matches_exactly_one_cc_formed_zero_broken(reactant_mols, product_mols):
                indices.append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    indices.sort()
    return indices, valid_reactions, skipped_reactions


def compute_task14_gt(task11_indices: list[int], task12_indices: list[int]) -> list[int]:
    return sorted(set(task11_indices) & set(task12_indices))


def product_contains_substructure(product_smiles: list[str], smarts: str) -> bool:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise ValueError(f"Could not parse SMARTS: {smarts}")
    for smiles in product_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        if mol.HasSubstructMatch(pattern):
            return True
    return False


def mol_has_shared_ring_bond(mol: Chem.Mol) -> bool:
    ring_info = mol.GetRingInfo()
    for bond in mol.GetBonds():
        if ring_info.NumBondRings(bond.GetIdx()) >= 2:
            return True
    return False


def smiles_list_has_fused_rings(smiles_list: list[str]) -> bool:
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        if mol_has_shared_ring_bond(mol):
            return True
    return False


def task6_parse_reaction_mols(indexed_line: str) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    _, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    reactant_smiles = [s for s in parts[0].split(".") if s]
    product_smiles = [s for s in parts[2].split(".") if s]
    reactants = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    products = [Chem.MolFromSmiles(s) for s in product_smiles]
    return [m for m in reactants if m is not None], [m for m in products if m is not None]


def task6_canonical_smiles_set(mols: list[Chem.Mol]) -> set[str]:
    return {Chem.MolToSmiles(m) for m in mols if m is not None}


def task6_reaction_matches(
    indexed_line: str, query_reaction: rdChemReactions.ChemicalReaction
) -> bool:
    reactants, products = task6_parse_reaction_mols(indexed_line)
    template = query_reaction
    template.Initialize()
    actual_product_smiles = task6_canonical_smiles_set(products)
    num_template_reactants = template.GetNumReactantTemplates()
    for perm in permutations(reactants, min(num_template_reactants, len(reactants))):
        if len(perm) != num_template_reactants:
            continue
        try:
            product_sets = template.RunReactants(perm)
        except Exception:
            continue
        for prod_set in product_sets:
            generated_smiles = set()
            for mol in prod_set:
                try:
                    Chem.SanitizeMol(mol)
                    generated_smiles.add(Chem.MolToSmiles(mol))
                except Exception:
                    continue
            if generated_smiles and generated_smiles.issubset(actual_product_smiles):
                return True
    return False


def task6_build_reaction_query(smarts: str) -> rdChemReactions.ChemicalReaction:
    query = rdChemReactions.ReactionFromSmarts(smarts)
    if query is None:
        raise ValueError(f"Failed to parse reaction SMARTS: {smarts}")
    return query


def compute_task6_gt(lines: list[str]) -> tuple[dict[str, list[int]], int, int]:
    indices_by_reaction: dict[str, list[int]] = {
        reaction_key: [] for reaction_key in TASK6_AMIDE_COUPLING_SMIRKS
    }
    valid_reactions = 0
    skipped_reactions = 0
    query_reactions = {
        reaction_key: task6_build_reaction_query(smarts)
        for reaction_key, smarts in TASK6_AMIDE_COUPLING_SMIRKS.items()
    }
    for line in lines:
        try:
            idx_str, _ = line.split(" ", 1)
            idx = int(idx_str)
            valid_reactions += 1
            for reaction_key, query_reaction in query_reactions.items():
                if task6_reaction_matches(line, query_reaction):
                    indices_by_reaction[reaction_key].append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    for reaction_key in indices_by_reaction:
        indices_by_reaction[reaction_key].sort()
    return indices_by_reaction, valid_reactions, skipped_reactions


def compute_task7_gt(lines: list[str]) -> tuple[dict[str, list[int]], int, int]:
    indices_by_reaction: dict[str, list[int]] = {
        reaction_key: [] for reaction_key in TASK7_TO_FG_SMIRKS
    }
    valid_reactions = 0
    skipped_reactions = 0
    query_reactions = {
        reaction_key: task6_build_reaction_query(smarts)
        for reaction_key, smarts in TASK7_TO_FG_SMIRKS.items()
    }
    for line in lines:
        try:
            idx_str, _ = line.split(" ", 1)
            idx = int(idx_str)
            valid_reactions += 1
            for reaction_key, query_reaction in query_reactions.items():
                if task6_reaction_matches(line, query_reaction):
                    indices_by_reaction[reaction_key].append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    for reaction_key in indices_by_reaction:
        indices_by_reaction[reaction_key].sort()
    return indices_by_reaction, valid_reactions, skipped_reactions


def compute_task8_gt(lines: list[str]) -> tuple[dict[str, list[int]], int, int]:
    indices_by_reaction: dict[str, list[int]] = {
        reaction_key: [] for reaction_key in TASK8_BOC_SMIRKS
    }
    valid_reactions = 0
    skipped_reactions = 0
    query_reactions = {
        reaction_key: task6_build_reaction_query(smarts)
        for reaction_key, smarts in TASK8_BOC_SMIRKS.items()
    }
    for line in lines:
        try:
            idx_str, _ = line.split(" ", 1)
            idx = int(idx_str)
            valid_reactions += 1
            for reaction_key, query_reaction in query_reactions.items():
                if task6_reaction_matches(line, query_reaction):
                    indices_by_reaction[reaction_key].append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    for reaction_key in indices_by_reaction:
        indices_by_reaction[reaction_key].sort()
    return indices_by_reaction, valid_reactions, skipped_reactions


def compute_task9_gt(lines: list[str]) -> tuple[dict[str, list[int]], int, int]:
    indices_by_reaction: dict[str, list[int]] = {
        reaction_key: [] for reaction_key in TASK9_NAMED_REACTIONS_SMIRKS
    }
    valid_reactions = 0
    skipped_reactions = 0
    query_reactions = {
        reaction_key: task6_build_reaction_query(smarts)
        for reaction_key, smarts in TASK9_NAMED_REACTIONS_SMIRKS.items()
    }
    for line in lines:
        try:
            idx_str, _ = line.split(" ", 1)
            idx = int(idx_str)
            valid_reactions += 1
            for reaction_key, query_reaction in query_reactions.items():
                if task6_reaction_matches(line, query_reaction):
                    indices_by_reaction[reaction_key].append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    for reaction_key in indices_by_reaction:
        indices_by_reaction[reaction_key].sort()
    return indices_by_reaction, valid_reactions, skipped_reactions


def compute_task16_gt(lines: list[str]) -> tuple[list[int], int, int]:
    indices: list[int] = []
    valid_reactions = 0
    skipped_reactions = 0
    for line in lines:
        try:
            idx, _reactant_smiles, product_smiles = parse_reaction_sides(line)
            valid_reactions += 1
            if product_contains_substructure(product_smiles, PYRIDINE_SMARTS):
                indices.append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    indices.sort()
    return indices, valid_reactions, skipped_reactions


def product_contains_any_ring_system(
    product_smiles: list[str], ring_systems: dict[str, str]
) -> bool:
    patterns: list[Chem.Mol] = []
    for smarts in ring_systems.values():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(f"Could not parse SMARTS: {smarts}")
        patterns.append(pattern)
    for smiles in product_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        if any(mol.HasSubstructMatch(pattern) for pattern in patterns):
            return True
    return False


def compute_task19_gt(lines: list[str]) -> tuple[list[int], int, int]:
    indices: list[int] = []
    valid_reactions = 0
    skipped_reactions = 0
    for line in lines:
        try:
            idx, reactant_smiles, product_smiles = parse_reaction_sides(line)
            valid_reactions += 1
            product_has_quinoline = product_contains_substructure(
                product_smiles, QUINOLINE_SMARTS
            )
            reactant_has_quinoline = product_contains_substructure(
                reactant_smiles, QUINOLINE_SMARTS
            )
            if product_has_quinoline and not reactant_has_quinoline:
                indices.append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    indices.sort()
    return indices, valid_reactions, skipped_reactions


def compute_task20_gt(lines: list[str]) -> tuple[list[int], int, int]:
    indices: list[int] = []
    valid_reactions = 0
    skipped_reactions = 0
    for line in lines:
        try:
            idx, reactant_smiles, product_smiles = parse_reaction_sides(line)
            valid_reactions += 1
            product_has_fused = smiles_list_has_fused_rings(product_smiles)
            reactant_has_fused = smiles_list_has_fused_rings(reactant_smiles)
            if product_has_fused and not reactant_has_fused:
                indices.append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    indices.sort()
    return indices, valid_reactions, skipped_reactions


def compute_task17_gt(lines: list[str]) -> tuple[list[int], int, int]:
    indices: list[int] = []
    valid_reactions = 0
    skipped_reactions = 0
    for line in lines:
        try:
            idx, _reactant_smiles, product_smiles = parse_reaction_sides(line)
            valid_reactions += 1
            if product_contains_any_ring_system(product_smiles, TASK17_RING_SYSTEMS):
                indices.append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    indices.sort()
    return indices, valid_reactions, skipped_reactions


def write_task17_module(
    out_path: Path,
    indices: list[int],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task17.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit SMARTS matching.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK17_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK17_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK17_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(f"TASK17_POSITIVE_REACTIONS = {len(indices)}\n")
        handle.write("TASK17_RING_SYSTEMS = ")
        handle.write(pformat(TASK17_RING_SYSTEMS, width=100))
        handle.write("\n\n")
        handle.write("TASK17_HARDCODED_GROUND_TRUTH_INDICES = ")
        handle.write(pformat(indices, width=100))
        handle.write("\n")


def cluster_ring_systems(mol: Chem.Mol) -> list[tuple[int, ...]]:
    rings = [set(ring_atoms) for ring_atoms in mol.GetRingInfo().AtomRings()]
    systems = list(rings)
    changed = True
    while changed:
        changed = False
        merged: list[set[int]] = []
        for system in systems:
            placed = False
            for index, existing in enumerate(merged):
                if system & existing:
                    merged[index] = existing | system
                    placed = True
                    changed = True
                    break
            if not placed:
                merged.append(system)
        systems = merged
    return [tuple(sorted(system)) for system in systems]


def ring_system_query(mol: Chem.Mol, atom_ids: tuple[int, ...]) -> Chem.Mol | None:
    smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atom_ids), canonical=True)
    if not smiles:
        return None
    query = Chem.MolFromSmiles(smiles)
    if query is None:
        return None
    Chem.SanitizeMol(query)
    return query


def ring_systems_equivalent(
    mol_a: Chem.Mol, atoms_a: tuple[int, ...], mol_b: Chem.Mol, atoms_b: tuple[int, ...]
) -> bool:
    if len(atoms_a) != len(atoms_b):
        return False
    query_a = ring_system_query(mol_a, atoms_a)
    query_b = ring_system_query(mol_b, atoms_b)
    if query_a is None or query_b is None:
        return False
    return query_a.HasSubstructMatch(query_b) and query_b.HasSubstructMatch(query_a)


def reactant_has_equivalent_ring_system(
    reactant_mols: list[Chem.Mol], product_mol: Chem.Mol, product_atom_ids: tuple[int, ...]
) -> bool:
    for reactant_mol in reactant_mols:
        for reactant_atom_ids in cluster_ring_systems(reactant_mol):
            if ring_systems_equivalent(
                reactant_mol, reactant_atom_ids, product_mol, product_atom_ids
            ):
                return True
    return False


def reaction_constructs_new_ring_system(
    reactant_smiles: list[str], product_smiles: list[str]
) -> bool:
    reactant_mols = mols_from_smiles(reactant_smiles)
    product_mols = mols_from_smiles(product_smiles)
    for product_mol in product_mols:
        for product_atom_ids in cluster_ring_systems(product_mol):
            if not reactant_has_equivalent_ring_system(
                reactant_mols, product_mol, product_atom_ids
            ):
                return True
    return False


def compute_task18_gt(lines: list[str]) -> tuple[list[int], int, int]:
    indices: list[int] = []
    valid_reactions = 0
    skipped_reactions = 0
    for line in lines:
        try:
            idx, reactant_smiles, product_smiles = parse_reaction_sides(line)
            valid_reactions += 1
            if reaction_constructs_new_ring_system(reactant_smiles, product_smiles):
                indices.append(idx)
        except Exception:
            skipped_reactions += 1
            continue
    indices.sort()
    return indices, valid_reactions, skipped_reactions


def write_task6_module(
    out_path: Path,
    indices_by_reaction: dict[str, list[int]],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    positive_counts = {
        reaction_key: len(indices) for reaction_key, indices in indices_by_reaction.items()
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task6.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit reaction SMIRKS matching.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK6_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK6_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK6_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(
            "TASK6_GROUND_TRUTH_DEFINITION = "
            '"reaction SMIRKS template match via RDKit RunReactants"\n'
        )
        handle.write("TASK6_AMIDE_COUPLING_SMIRKS = ")
        handle.write(pformat(TASK6_AMIDE_COUPLING_SMIRKS, width=100))
        handle.write("\n")
        handle.write("TASK6_POSITIVE_REACTIONS_BY_KEY = ")
        handle.write(pformat(positive_counts, width=100))
        handle.write("\n\n")
        handle.write("TASK6_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION = ")
        handle.write(pformat(indices_by_reaction, width=100))
        handle.write("\n")


def write_task7_module(
    out_path: Path,
    indices_by_reaction: dict[str, list[int]],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    positive_counts = {
        reaction_key: len(indices) for reaction_key, indices in indices_by_reaction.items()
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task7.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit reaction SMIRKS matching.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK7_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK7_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK7_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(
            "TASK7_GROUND_TRUTH_DEFINITION = "
            '"reaction SMIRKS template match via RDKit RunReactants"\n'
        )
        handle.write("TASK7_TO_FG_SMIRKS = ")
        handle.write(pformat(TASK7_TO_FG_SMIRKS, width=100))
        handle.write("\n")
        handle.write("TASK7_POSITIVE_REACTIONS_BY_KEY = ")
        handle.write(pformat(positive_counts, width=100))
        handle.write("\n\n")
        handle.write("TASK7_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION = ")
        handle.write(pformat(indices_by_reaction, width=100))
        handle.write("\n")


def write_task8_module(
    out_path: Path,
    indices_by_reaction: dict[str, list[int]],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    positive_counts = {
        reaction_key: len(indices) for reaction_key, indices in indices_by_reaction.items()
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task8.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit reaction SMIRKS matching.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK8_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK8_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK8_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(
            "TASK8_GROUND_TRUTH_DEFINITION = "
            '"reaction SMIRKS template match via RDKit RunReactants"\n'
        )
        handle.write("TASK8_BOC_SMIRKS = ")
        handle.write(pformat(TASK8_BOC_SMIRKS, width=100))
        handle.write("\n")
        handle.write("TASK8_POSITIVE_REACTIONS_BY_KEY = ")
        handle.write(pformat(positive_counts, width=100))
        handle.write("\n\n")
        handle.write("TASK8_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION = ")
        handle.write(pformat(indices_by_reaction, width=100))
        handle.write("\n")


def write_task9_module(
    out_path: Path,
    indices_by_reaction: dict[str, list[int]],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    positive_counts = {
        reaction_key: len(indices) for reaction_key, indices in indices_by_reaction.items()
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task9.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit reaction SMIRKS matching.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK9_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK9_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK9_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(
            "TASK9_GROUND_TRUTH_DEFINITION = "
            '"reaction SMIRKS template match via RDKit RunReactants"\n'
        )
        handle.write("TASK9_NAMED_REACTIONS_SMIRKS = ")
        handle.write(pformat(TASK9_NAMED_REACTIONS_SMIRKS, width=100))
        handle.write("\n")
        handle.write("TASK9_POSITIVE_REACTIONS_BY_KEY = ")
        handle.write(pformat(positive_counts, width=100))
        handle.write("\n\n")
        handle.write("TASK9_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION = ")
        handle.write(pformat(indices_by_reaction, width=100))
        handle.write("\n")


def write_task18_module(
    out_path: Path,
    indices: list[int],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task18.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit ring-system comparison.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK18_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK18_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK18_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(f"TASK18_POSITIVE_REACTIONS = {len(indices)}\n")
        handle.write(
            "TASK18_GROUND_TRUTH_DEFINITION = "
            '"equal ring-atom count plus mutual sanitized substructure match on clustered ring systems"\n\n'
        )
        handle.write("TASK18_HARDCODED_GROUND_TRUTH_INDICES = ")
        handle.write(pformat(indices, width=100))
        handle.write("\n")


def write_task19_module(
    out_path: Path,
    indices: list[int],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task19.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit SMARTS matching.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK19_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK19_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK19_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(f"TASK19_POSITIVE_REACTIONS = {len(indices)}\n")
        handle.write(
            'TASK19_GROUND_TRUTH_DEFINITION = '
            '"product contains quinoline and no reactant contains quinoline"\n'
        )
        handle.write(f'TASK19_QUINOLINE_SMARTS = "{QUINOLINE_SMARTS}"\n\n')
        handle.write("TASK19_HARDCODED_GROUND_TRUTH_INDICES = ")
        handle.write(pformat(indices, width=100))
        handle.write("\n")


def write_task20_module(
    out_path: Path,
    indices: list[int],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task20.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit ring-bond analysis.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK20_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK20_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK20_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(f"TASK20_POSITIVE_REACTIONS = {len(indices)}\n")
        handle.write(
            "TASK20_GROUND_TRUTH_DEFINITION = "
            '"product has fused rings (shared ring bond) and no reactant has fused rings"\n\n'
        )
        handle.write("TASK20_HARDCODED_GROUND_TRUTH_INDICES = ")
        handle.write(pformat(indices, width=100))
        handle.write("\n")


def write_task16_module(
    out_path: Path,
    indices: list[int],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task16.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit SMARTS matching.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK16_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK16_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK16_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(f"TASK16_POSITIVE_REACTIONS = {len(indices)}\n")
        handle.write(f'TASK16_PYRIDINE_SMARTS = "{PYRIDINE_SMARTS}"\n\n')
        handle.write("TASK16_HARDCODED_GROUND_TRUTH_INDICES = ")
        handle.write(pformat(indices, width=100))
        handle.write("\n")


def write_task15_module(
    out_path: Path,
    indices: list[int],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task15.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit extraction.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK15_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK15_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK15_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(f"TASK15_POSITIVE_REACTIONS = {len(indices)}\n\n")
        handle.write("TASK15_HARDCODED_GROUND_TRUTH_INDICES = ")
        handle.write(pformat(indices, width=100))
        handle.write("\n")


def write_task14_module(
    out_path: Path,
    indices: list[int],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
    task11_positive_reactions: int,
    task12_positive_reactions: int,
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task14.\n')
        handle.write(
            "Generated as intersection of task11 (C-N forming) and task12 (C-O breaking) positives.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK14_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK14_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK14_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(f"TASK14_POSITIVE_REACTIONS = {len(indices)}\n")
        handle.write(f"TASK14_TASK11_POSITIVE_REACTIONS = {task11_positive_reactions}\n")
        handle.write(f"TASK14_TASK12_POSITIVE_REACTIONS = {task12_positive_reactions}\n\n")
        handle.write("TASK14_HARDCODED_GROUND_TRUTH_INDICES = ")
        handle.write(pformat(indices, width=100))
        handle.write("\n")


def write_task13_module(
    out_path: Path,
    indices: list[int],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task13.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit extraction.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK13_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK13_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK13_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(f"TASK13_POSITIVE_REACTIONS = {len(indices)}\n\n")
        handle.write("TASK13_HARDCODED_GROUND_TRUTH_INDICES = ")
        handle.write(pformat(indices, width=100))
        handle.write("\n")


def write_task12_module(
    out_path: Path,
    indices: list[int],
    total_reactions: int,
    valid_reactions: int,
    skipped_reactions: int,
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write('"""Hardcoded ground-truth indices for tier3 task12.\n')
        handle.write(
            "Generated from reactionSmilesFigShareUSPTO2023_cleaned.txt with RDKit extraction.\n"
        )
        handle.write('"""\n\n')
        handle.write(f"TASK12_TOTAL_REACTIONS = {total_reactions}\n")
        handle.write(f"TASK12_VALID_REACTIONS = {valid_reactions}\n")
        handle.write(f"TASK12_SKIPPED_REACTIONS = {skipped_reactions}\n")
        handle.write(f"TASK12_POSITIVE_REACTIONS = {len(indices)}\n\n")
        handle.write("TASK12_HARDCODED_GROUND_TRUTH_INDICES = ")
        handle.write(pformat(indices, width=100))
        handle.write("\n")


def main() -> None:
    lines = load_indexed_lines()

    task11_indices, task11_valid, task11_skipped = compute_task11_gt(lines)
    task11_out_path = Path(__file__).parent / "task11_hardcoded_ground_truth.py"
    write_task11_module(
        out_path=task11_out_path,
        indices=task11_indices,
        total_reactions=len(lines),
        valid_reactions=task11_valid,
        skipped_reactions=task11_skipped,
    )
    print(f"Wrote {task11_out_path}")
    print(f"task11 positives: {len(task11_indices)}")
    print(f"task11 valid reactions: {task11_valid}")
    print(f"task11 skipped reactions: {task11_skipped}")

    task12_indices, task12_valid, task12_skipped = compute_task12_gt(lines)
    task12_out_path = Path(__file__).parent / "task12_hardcoded_ground_truth.py"
    write_task12_module(
        out_path=task12_out_path,
        indices=task12_indices,
        total_reactions=len(lines),
        valid_reactions=task12_valid,
        skipped_reactions=task12_skipped,
    )
    print(f"Wrote {task12_out_path}")
    print(f"task12 positives: {len(task12_indices)}")
    print(f"task12 valid reactions: {task12_valid}")
    print(f"task12 skipped reactions: {task12_skipped}")

    task13_indices, task13_valid, task13_skipped = compute_task13_gt(lines)
    task13_out_path = Path(__file__).parent / "task13_hardcoded_ground_truth.py"
    write_task13_module(
        out_path=task13_out_path,
        indices=task13_indices,
        total_reactions=len(lines),
        valid_reactions=task13_valid,
        skipped_reactions=task13_skipped,
    )
    print(f"Wrote {task13_out_path}")
    print(f"task13 positives: {len(task13_indices)}")
    print(f"task13 valid reactions: {task13_valid}")
    print(f"task13 skipped reactions: {task13_skipped}")

    task14_indices = compute_task14_gt(task11_indices, task12_indices)
    task14_out_path = Path(__file__).parent / "task14_hardcoded_ground_truth.py"
    write_task14_module(
        out_path=task14_out_path,
        indices=task14_indices,
        total_reactions=len(lines),
        valid_reactions=task11_valid,
        skipped_reactions=task11_skipped,
        task11_positive_reactions=len(task11_indices),
        task12_positive_reactions=len(task12_indices),
    )
    print(f"Wrote {task14_out_path}")
    print(f"task14 positives: {len(task14_indices)}")
    print(
        f"task14 intersection of task11 ({len(task11_indices)}) "
        f"and task12 ({len(task12_indices)})"
    )

    task15_indices, task15_valid, task15_skipped = compute_task15_gt(lines)
    task15_out_path = Path(__file__).parent / "task15_hardcoded_ground_truth.py"
    write_task15_module(
        out_path=task15_out_path,
        indices=task15_indices,
        total_reactions=len(lines),
        valid_reactions=task15_valid,
        skipped_reactions=task15_skipped,
    )
    print(f"Wrote {task15_out_path}")
    print(f"task15 positives: {len(task15_indices)}")
    print(f"task15 valid reactions: {task15_valid}")
    print(f"task15 skipped reactions: {task15_skipped}")

    task16_indices, task16_valid, task16_skipped = compute_task16_gt(lines)
    task16_out_path = Path(__file__).parent / "task16_hardcoded_ground_truth.py"
    write_task16_module(
        out_path=task16_out_path,
        indices=task16_indices,
        total_reactions=len(lines),
        valid_reactions=task16_valid,
        skipped_reactions=task16_skipped,
    )
    print(f"Wrote {task16_out_path}")
    print(f"task16 positives: {len(task16_indices)}")
    print(f"task16 valid reactions: {task16_valid}")
    print(f"task16 skipped reactions: {task16_skipped}")

    task17_indices, task17_valid, task17_skipped = compute_task17_gt(lines)
    task17_out_path = Path(__file__).parent / "task17_hardcoded_ground_truth.py"
    write_task17_module(
        out_path=task17_out_path,
        indices=task17_indices,
        total_reactions=len(lines),
        valid_reactions=task17_valid,
        skipped_reactions=task17_skipped,
    )
    print(f"Wrote {task17_out_path}")
    print(f"task17 positives: {len(task17_indices)}")
    print(f"task17 valid reactions: {task17_valid}")
    print(f"task17 skipped reactions: {task17_skipped}")

    task18_indices, task18_valid, task18_skipped = compute_task18_gt(lines)
    task18_out_path = Path(__file__).parent / "task18_hardcoded_ground_truth.py"
    write_task18_module(
        out_path=task18_out_path,
        indices=task18_indices,
        total_reactions=len(lines),
        valid_reactions=task18_valid,
        skipped_reactions=task18_skipped,
    )
    print(f"Wrote {task18_out_path}")
    print(f"task18 positives: {len(task18_indices)}")
    print(f"task18 valid reactions: {task18_valid}")
    print(f"task18 skipped reactions: {task18_skipped}")

    task19_indices, task19_valid, task19_skipped = compute_task19_gt(lines)
    task19_out_path = Path(__file__).parent / "task19_hardcoded_ground_truth.py"
    write_task19_module(
        out_path=task19_out_path,
        indices=task19_indices,
        total_reactions=len(lines),
        valid_reactions=task19_valid,
        skipped_reactions=task19_skipped,
    )
    print(f"Wrote {task19_out_path}")
    print(f"task19 positives: {len(task19_indices)}")
    print(f"task19 valid reactions: {task19_valid}")
    print(f"task19 skipped reactions: {task19_skipped}")

    task20_indices, task20_valid, task20_skipped = compute_task20_gt(lines)
    task20_out_path = Path(__file__).parent / "task20_hardcoded_ground_truth.py"
    write_task20_module(
        out_path=task20_out_path,
        indices=task20_indices,
        total_reactions=len(lines),
        valid_reactions=task20_valid,
        skipped_reactions=task20_skipped,
    )
    print(f"Wrote {task20_out_path}")
    print(f"task20 positives: {len(task20_indices)}")
    print(f"task20 valid reactions: {task20_valid}")
    print(f"task20 skipped reactions: {task20_skipped}")

    task6_indices_by_reaction, task6_valid, task6_skipped = compute_task6_gt(lines)
    task6_out_path = Path(__file__).parent / "task6_hardcoded_ground_truth.py"
    write_task6_module(
        out_path=task6_out_path,
        indices_by_reaction=task6_indices_by_reaction,
        total_reactions=len(lines),
        valid_reactions=task6_valid,
        skipped_reactions=task6_skipped,
    )
    print(f"Wrote {task6_out_path}")
    for reaction_key, indices in task6_indices_by_reaction.items():
        print(f"task6 positives [{reaction_key}]: {len(indices)}")
    print(f"task6 valid reactions: {task6_valid}")
    print(f"task6 skipped reactions: {task6_skipped}")

    task7_indices_by_reaction, task7_valid, task7_skipped = compute_task7_gt(lines)
    task7_out_path = Path(__file__).parent / "task7_hardcoded_ground_truth.py"
    write_task7_module(
        out_path=task7_out_path,
        indices_by_reaction=task7_indices_by_reaction,
        total_reactions=len(lines),
        valid_reactions=task7_valid,
        skipped_reactions=task7_skipped,
    )
    print(f"Wrote {task7_out_path}")
    for reaction_key, indices in task7_indices_by_reaction.items():
        print(f"task7 positives [{reaction_key}]: {len(indices)}")
    print(f"task7 valid reactions: {task7_valid}")
    print(f"task7 skipped reactions: {task7_skipped}")

    task8_indices_by_reaction, task8_valid, task8_skipped = compute_task8_gt(lines)
    task8_out_path = Path(__file__).parent / "task8_hardcoded_ground_truth.py"
    write_task8_module(
        out_path=task8_out_path,
        indices_by_reaction=task8_indices_by_reaction,
        total_reactions=len(lines),
        valid_reactions=task8_valid,
        skipped_reactions=task8_skipped,
    )
    print(f"Wrote {task8_out_path}")
    for reaction_key, indices in task8_indices_by_reaction.items():
        print(f"task8 positives [{reaction_key}]: {len(indices)}")
    print(f"task8 valid reactions: {task8_valid}")
    print(f"task8 skipped reactions: {task8_skipped}")

    task9_indices_by_reaction, task9_valid, task9_skipped = compute_task9_gt(lines)
    task9_out_path = Path(__file__).parent / "task9_hardcoded_ground_truth.py"
    write_task9_module(
        out_path=task9_out_path,
        indices_by_reaction=task9_indices_by_reaction,
        total_reactions=len(lines),
        valid_reactions=task9_valid,
        skipped_reactions=task9_skipped,
    )
    print(f"Wrote {task9_out_path}")
    for reaction_key, indices in task9_indices_by_reaction.items():
        print(f"task9 positives [{reaction_key}]: {len(indices)}")
    print(f"task9 valid reactions: {task9_valid}")
    print(f"task9 skipped reactions: {task9_skipped}")


if __name__ == "__main__":
    main()
