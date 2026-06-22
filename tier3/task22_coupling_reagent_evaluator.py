"""Evaluator for tier3 task22 HATU+PF6 or T3P reagent detection."""

from __future__ import annotations

from rdkit import Chem, rdBase

rdBase.DisableLog("rdApp.*")

REACTION_KEY = "hatu_pf6_or_t3p_reagent"

HATU_PF6_SMILES = "CN(C)C(=[N+](C)C)On1nnc2cccnc21.F[P-](F)(F)(F)(F)F"
T3P_SMILES = "CCCP1(=O)OP(=O)(CCC)OP(=O)(CCC)O1"

TASK22_GROUND_TRUTH_DEFINITION = (
    "reaction reagent slot contains canonical HATU+PF6 fragments or canonical T3P"
)


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def canonical_fragments(smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    return {
        Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=False)
        for fragment in Chem.GetMolFrags(mol, asMols=True)
    }


HATU_PF6_CANONICAL_FRAGMENTS = frozenset(canonical_fragments(HATU_PF6_SMILES))
T3P_CANONICAL_SMILES = canonical_smiles(T3P_SMILES)


def parse_reagent_smiles(reaction_smiles: str) -> list[str] | None:
    parts = reaction_smiles.strip().split(">")
    if len(parts) != 3:
        return None
    reagent_slot = parts[1]
    if not reagent_slot:
        return []
    return [smiles.strip() for smiles in reagent_slot.split(".") if smiles.strip()]


def canonical_reagent_components(reaction_smiles: str) -> set[str] | None:
    reagent_smiles = parse_reagent_smiles(reaction_smiles)
    if reagent_smiles is None:
        return None
    canonical = set()
    for smiles in reagent_smiles:
        component = canonical_smiles(smiles)
        if component is not None:
            canonical.add(component)
    return canonical


def reaction_has_hatu_pf6_or_t3p_reagent(reaction_smiles: str) -> bool:
    reagent_components = canonical_reagent_components(reaction_smiles)
    if reagent_components is None:
        return False
    has_hatu_pf6 = HATU_PF6_CANONICAL_FRAGMENTS.issubset(reagent_components)
    has_t3p = (
        T3P_CANONICAL_SMILES is not None
        and T3P_CANONICAL_SMILES in reagent_components
    )
    return has_hatu_pf6 or has_t3p


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
            if reaction_has_hatu_pf6_or_t3p_reagent(reaction):
                positives.append(idx)
        except Exception:
            skipped += 1
    return positives, skipped


def classify_reagent_match(reaction_smiles: str) -> set[str]:
    reagent_components = canonical_reagent_components(reaction_smiles)
    if reagent_components is None:
        return set()
    matches = set()
    if HATU_PF6_CANONICAL_FRAGMENTS.issubset(reagent_components):
        matches.add("HATU+PF6")
    if T3P_CANONICAL_SMILES is not None and T3P_CANONICAL_SMILES in reagent_components:
        matches.add("T3P")
    return matches
