"""Evaluator for tier3 task24 E-configured C=C double bond in products."""

from __future__ import annotations

import re

from rdkit import Chem, rdBase
from rdkit.Chem import BondStereo

rdBase.DisableLog("rdApp.*")

REACTION_KEY = "e_double_bond_product"

TASK24_GROUND_TRUTH_DEFINITION = (
    "at least one product SMILES contains a non-aromatic C=C double bond with CIP-based "
    "E configuration after AssignStereochemistry; unannotated double bonds are excluded"
)


def count_e_double_bonds(mol: Chem.Mol | None) -> int:
    if mol is None:
        return 0
    try:
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        return 0

    count = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if begin_atom.GetAtomicNum() != 6 or end_atom.GetAtomicNum() != 6:
            continue
        if bond.GetIsAromatic():
            continue
        stereo = bond.GetStereo()
        if stereo == BondStereo.STEREOE or stereo == BondStereo.STEREOTRANS:
            count += 1
    return count


def parse_reaction(reaction_smiles: str) -> tuple[list[str], list[str], list[str]] | tuple[None, None, None]:
    parts = reaction_smiles.strip().split(">")
    if len(parts) != 3:
        return None, None, None
    reactants = [smiles for smiles in parts[0].split(".") if smiles]
    reagents = [smiles for smiles in parts[1].split(".") if smiles]
    products = [smiles for smiles in parts[2].split(".") if smiles]
    return reactants, reagents, products


def reaction_has_e_double_bond_product(reaction_smiles: str) -> bool:
    _, _, products = parse_reaction(reaction_smiles)
    if not products:
        return False

    for smiles in products:
        mol = Chem.MolFromSmiles(smiles)
        if count_e_double_bonds(mol) > 0:
            return True
    return False


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
            if reaction_has_e_double_bond_product(reaction):
                positives.append(idx)
        except Exception:
            skipped += 1
    return positives, skipped


def format_ground_truth(matching_indices: set[int] | list[int]) -> str:
    sorted_indices = sorted(matching_indices)
    return f"<answer>{','.join(str(i) for i in sorted_indices)}</answer>"


def parse_model_answer(answer_text: str) -> set[int]:
    match = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL)
    if not match:
        return set()

    content = match.group(1).strip()
    if not content:
        return set()

    indices: set[int] = set()
    for token in re.split(r"[,\\s]+", content):
        token = token.strip()
        if token.lstrip("-").isdigit():
            indices.add(int(token))
    return indices


def score(predicted_indices: set[int], ground_truth_indices: set[int]) -> dict[str, float | int]:
    tp = len(predicted_indices & ground_truth_indices)
    fp = len(predicted_indices - ground_truth_indices)
    fn = len(ground_truth_indices - predicted_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_count": len(predicted_indices),
        "ground_truth_count": len(ground_truth_indices),
    }


def evaluate(model_response: str, reaction_smiles_list: list[str]) -> dict[str, float | int]:
    ground_truth = {
        idx
        for idx, reaction in enumerate(reaction_smiles_list)
        if reaction_has_e_double_bond_product(reaction)
    }
    predicted = parse_model_answer(model_response)
    return score(predicted, ground_truth)
