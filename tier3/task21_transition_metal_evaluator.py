"""Evaluator for tier3 task21 transition-metal reagent detection."""

from __future__ import annotations

from collections import Counter

from rdkit import Chem, rdBase

rdBase.DisableLog("rdApp.*")

REACTION_KEY = "transition_metal_reagent"

TRANSITION_METAL_ATOMIC_NUMS = (
    set(range(21, 31))
    | set(range(39, 49))
    | set(range(72, 81))
    | set(range(104, 113))
)

TRANSITION_METAL_SYMBOLS = (
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
)

TASK21_GROUND_TRUTH_DEFINITION = (
    "reaction reagent slot contains at least one atom from the explicit transition-metal "
    "set: Sc-Zn, Y-Cd, Hf-Hg, Rf-Cn"
)


def mol_from_smiles_lenient(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol
    return Chem.MolFromSmiles(smiles, sanitize=False)


def mol_has_transition_metal(mol: Chem.Mol | None) -> bool:
    if mol is None:
        return False
    return any(
        atom.GetAtomicNum() in TRANSITION_METAL_ATOMIC_NUMS
        for atom in mol.GetAtoms()
    )


def transition_metals_in_mol(mol: Chem.Mol | None) -> set[str]:
    if mol is None:
        return set()
    return {
        atom.GetSymbol()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() in TRANSITION_METAL_ATOMIC_NUMS
    }


def parse_reagent_smiles(reaction_smiles: str) -> list[str] | None:
    parts = reaction_smiles.strip().split(">")
    if len(parts) != 3:
        return None
    reagent_slot = parts[1]
    if not reagent_slot:
        return []
    return [smiles.strip() for smiles in reagent_slot.split(".") if smiles.strip()]


def reaction_has_transition_metal_reagent(reaction_smiles: str) -> bool:
    reagent_smiles = parse_reagent_smiles(reaction_smiles)
    if reagent_smiles is None:
        return False
    for smiles in reagent_smiles:
        if mol_has_transition_metal(mol_from_smiles_lenient(smiles)):
            return True
    return False


def transition_metals_in_reagent_slot(reaction_smiles: str) -> set[str]:
    reagent_smiles = parse_reagent_smiles(reaction_smiles)
    if reagent_smiles is None:
        return set()
    metals = set()
    for smiles in reagent_smiles:
        metals.update(transition_metals_in_mol(mol_from_smiles_lenient(smiles)))
    return metals


def compute_ground_truth_indices(lines: list[str]) -> tuple[list[int], int]:
    positives = []
    skipped = 0
    for line in lines:
        try:
            idx_str, reaction = line.split(" ", 1)
            idx = int(idx_str)
        except ValueError:
            skipped += 1
            continue
        try:
            if reaction_has_transition_metal_reagent(reaction):
                positives.append(idx)
        except Exception:
            skipped += 1
    return positives, skipped


def compute_metal_frequency(lines: list[str], indices: list[int]) -> Counter[str]:
    line_by_idx = {}
    for line in lines:
        try:
            idx_str, reaction = line.split(" ", 1)
            line_by_idx[int(idx_str)] = reaction
        except ValueError:
            continue

    metals: Counter[str] = Counter()
    for idx in indices:
        reaction = line_by_idx.get(idx)
        if reaction is None:
            continue
        metals.update(transition_metals_in_reagent_slot(reaction))
    return metals
