"""Mechanism evaluator for tier3 task10 mechanism-family questions.

This module intentionally keeps the hidden RDKit cascade out of the RLM prompt.
It is used to generate the hardcoded ground truth for task10.
"""

from __future__ import annotations

from itertools import permutations

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

rdBase.DisableLog("rdApp.*")

REACTION_KEYS = (
    "ester_hydrolysis_deprotection_with_oh",
    "mitsunobu_reaction_family",
    "wittig_olefination",
    "knoevenagel_aldol_condensation",
    "azide_alkyne_huisgen_cycloaddition",
)
TASK10_GROUND_TRUTH_DEFINITION = (
    "RDKit cascade through selected mechanism stages; actual "
    "products must be a connectivity-equivalent subset of the final state and include "
    "at least one final mechanism product"
)

# Literal stages from the "Ester hydrolysis" / "Deprotection with OH-" entry.
ESTER_HYDROLYSIS_OH_STAGES = {
    0: {
        "Templates": [
            "[#8;H1;-1:5].[#6:4]-[#8;H0;+0:1]-[#6;H0;+0:2]=[#8:3]>>[#6:4]-[#8;H0;+0:1]-[#6;H0;+0:2]([#8;D1;H1;+0:5])-[#8;H0;-1:3]",
            "[Li,Na,K:9][O;H1;+0:5].[#6:4]-[O;H0;+0:1]-[C;H0;+0:2]=[O:3]>>[Li,Na,K;H0;+1:9].[#6:4]-[O;H0;+0:1]-[C;H0;+0:2]([#8;D1;H1;+0:5])-[O;H0;-1:3]",
        ],
    },
    1: {
        "Templates": [
            "[#6:4]-[O;H0;+0:1]-[C;H0;+0:2]([#8;+0:5])-[O;H0;-1:3]>>[O;H0;+0:3]=[C;H0;+0:2]-[#8;+0:5].[#6:4]-[O;H0;-1:1]",
        ],
    },
}

MITSUNOBU_REACTION_STAGES = {
    0: {
        "Templates": [
            "[P;!D0;+0:1].[N;H0;+0:2]=[N;H0;+0:3]>>[P;+1:1]-[N;H0;+0:2]-[N;H0;-1:3]",
        ],
    },
    1: {
        "Templates": [
            "[P;+1:1]-[N;H0;+0:2]-[N;H0;-1:3].[#7,#16;H1;+0:4]>>[P;+1:1]-[N;H0;+0:2]-[N;H1;+0:3].[#7,#16;H0;-1:4]",
            "[P;+1:1]-[N;H0;+0:2]-[N;H0;-1:3].[#8;H1;+0:4]-[c:5]>>[P;+1:1]-[N;H0;+0:2]-[N;H1;+0:3].[#8;H0;-1:4]-[c:5]",
            "[P;+1:1]-[N;H0;+0:2]-[N;H0;-1:3].[#8;H1;+0:4]-[C:5]=[O:6]>>[P;+1:1]-[N;H0;+0:2]-[N;H1;+0:3].[#8;H0;-1:4]-[C:5]=[O:6]",
        ],
    },
    2: {
        "Templates": [
            "[P;+1:1]-[N;H0;+0:2]-[N;H1;+0:3].[#8;H1;+0:5]>>[#8;H1;+1:5]-[P;+0:1]-[N;H0;+0:2]-[N;H1;+0:3]",
        ],
    },
    3: {
        "Templates": [
            "[#8;H1;+1:5]-[P;+0:1]-[N;H0;+0:2]-[N;H1;+0:3]>>[#8;H0;+0:5]-[P;+0:1]-[N;H0;+0:2]-[N;H1;+0:3]",
        ],
    },
    4: {
        "Templates": [
            "[#8;H0;+0:5]-[P;+0:1]-[N;H0;+0:2]-[N;H1;+0:3]>>[#8;H0;+0:5]-[P;+0:1]-[N;H1;+1:2]-[N;H1;+0:3]",
        ],
    },
    5: {
        "Templates": [
            "[#8;H0;+0:5]-[P;+0:1]-[N;H1;+1:2]-[N;H1;+0:3]>>[#8;H0;+0:5]-[P;+1:1].[N;H1;+0:2]-[N;H1;+0:3]",
            "[#8;H0;+0:5]-[P;+0:1]-[N;H0;+0:2]-[N;H1;+0:3]>>[#8;H0;+0:5]-[P;+1:1].[N;H0;-1:2]-[N;H1;+0:3]",
        ],
    },
    6: {
        "Templates": [
            "[P;+1:1]-[#8;H0;+0:5]-[#6;+0:6].[#7,#8,#16;H0;-1:4]>>[P;+0:1]=[#8;H0;+0:5].[#6;+0:6]-[#7,#8,#16;H0;+0:4]",
        ],
    },
}

WITTIG_OLEFINATION_STAGES = {
    0: {
        "Templates": [
            "[P;+1:1]-[*;-1:2].[#6:3]=[#8:4]>>[P;+0:1]1-[*:2]-[#6:3]-[#8:4]1",
            "[P;+0:1]=[*:2].[#6:3]=[#8:4]>>[P;+0:1]1-[*:2]-[#6:3]-[#8:4]1",
        ],
    },
    1: {
        "Templates": [
            "[P;+0:1]1-[*:2]-[#6:3]-[#8:4]1>>[P;+0:1]=[#8:4].[*:2]=[#6:3]",
        ],
    },
}

KNOEVENAGEL_ALDOL_STAGES = {
    0: {
        "Templates": [
            "[#8:1]=[#6:2]-[#6;H2:3]-[#6:4]=[#8:5]>>[#8;-1:1]-[#6:2]=[#6;H1;+0:3]-[#6:4]=[#8:5]",
            "[#8:1]=[#6:2](-[!#8:4])-[#6;H3:3]>>[#8;-1:1]-[#6:2](-[!#8:4])=[#6;H2;+0:3]",
            "[#8:1]=[#6:2]-[#6;H2:3]>>[#8;-1:1]-[#6:2]=[#6;H1;+0:3]",
            "[#7:1]#[#6:2]-[#6;H3:3]>>[#7:1]#[#6:2]-[#6;H2;-1:3]",
            "[#7:1]#[#6:2]-[#6;H2:3]>>[#7:1]#[#6:2]-[#6;H1;-1:3]",
        ],
    },
    1: {
        "Templates": [
            "[#6;D1;H2;-1:3].[#6:4]=[#8:5]>>[#6;D2;H2;+0:3]-[#6:4]-[#8;-1:5]",
            "[#6;D2;H1;-1:3].[#6:4]=[#8:5]>>[#6;D3;H1;+0:3]-[#6:4]-[#8;-1:5]",
            "[#8;-1:1]-[#6:2]=[#6;+0:3].[#6:4]=[#8:5]>>[#8;+0:1]=[#6:2]-[#6;+0:3]-[#6:4]-[#8;-1:5]",
        ],
    },
    2: {
        "Templates": [
            "[#6:4]-[#8;H0;-1:5]>>[#6:4]-[#8;H1;+0:5]",
        ],
    },
    3: {
        "Templates": [
            "[#8:1]=[#6:2]-[#6;H1;+0:3]-[#6:4]-[#8;H1;+0:5]>>[#8;-1:1]-[#6:2]=[#6;H0;+0:3]-[#6:4]-[#8;H1;+0:5]",
            "[#8:1]=[#6:2]-[#6;H1;+0:3](-[#6:4]-[#8;H1;+0:5])-[#6:6]=[#8:7]>>[#8;-1:1]-[#6:2]=[#6;H0;+0:3](-[#6:4]-[#8;H1;+0:5])-[#6:6]=[#8:7]",
            "[#8:1]=[#6:2]-[#6;H2;+0:3]-[#6:4]-[#8;H1;+0:5]>>[#8;-1:1]-[#6:2]=[#6;H1;+0:3]-[#6:4]-[#8;H1;+0:5]",
            "[#8:1]=[#6:2]-[#6;H2;+0:3](-[#6:4]-[#8;H1;+0:5])-[#6:6]=[#8:7]>>[#8;-1:1]-[#6:2]=[#6;H1;+0:3](-[#6:4]-[#8;H1;+0:5])-[#6:6]=[#8:7]",
            "[#7:1]#[#6:2]-[#6;H1;+0:3]-[#6:4]-[#8;H1;+0:5]>>[#7;-1:1]=[#6:2]=[#6;H0;+0:3]-[#6:4]-[#8;H1;+0:5]",
        ],
    },
    4: {
        "Templates": [
            "[#8;-1:1]-[#6:2]=[#6;H0;+0:3]-[#6:4]-[#8;H1;+0:5]>>[#8;+0:1]=[#6:2][#6;H0;+0:3]=[#6:4].[#8;H1;-1:5]",
            "[#8;-1:1]-[#6:2]=[#6;H0;+0:3](-[#6:4]-[#8;H1;+0:5])-[#6:6]=[#8:7]>>[#8;+0:1]=[#6:2][#6;H0;+0:3](=[#6:4])-[#6:6]=[#8:7].[#8;H1;-1:5]",
            "[#8;-1:1]-[#6:2]=[#6;H1;+0:3]-[#6:4]-[#8;H1;+0:5]>>[#8;+0:1]=[#6:2][#6;H1;+0:3]=[#6:4].[#8;H1;-1:5]",
            "[#8;-1:1]-[#6:2]=[#6;H1;+0:3](-[#6:4]-[#8;H1;+0:5])-[#6:6]=[#8:7]>>[#8;+0:1]=[#6:2][#6;H1;+0:3](=[#6:4])-[#6:6]=[#8:7].[#8;H1;-1:5]",
            "[#7;-1:1]=[#6:2]=[#6;H0;+0:3]-[#6:4]-[#8;H1;+0:5]>>[#7:1]#[#6:2]-[#6;H0;+0:3]=[#6:4].[#8;H1;-1:5]",
        ],
    },
}

AZIDE_ALKYNE_HUISGEN_STAGES = {
    0: {
        "Templates": [
            "[N;-1:1]=[N;+1:2]=[N:3].[*:4]#[#6:5]>>[#7;+0:1]1=[#7;+0:2][#7:3][*:4]=[#6:5]1",
        ],
    },
}

TASK10_MECHANISM_STAGES = {
    "ester_hydrolysis_deprotection_with_oh": ESTER_HYDROLYSIS_OH_STAGES,
    "mitsunobu_reaction_family": MITSUNOBU_REACTION_STAGES,
    "wittig_olefination": WITTIG_OLEFINATION_STAGES,
    "knoevenagel_aldol_condensation": KNOEVENAGEL_ALDOL_STAGES,
    "azide_alkyne_huisgen_cycloaddition": AZIDE_ALKYNE_HUISGEN_STAGES,
}


def parse_reaction_components(reaction_line: str) -> tuple[list[str], list[str]] | None:
    reaction = reaction_line.strip()
    if not reaction:
        return None
    if " " in reaction and reaction.split(" ", 1)[0].isdigit():
        reaction = reaction.split(" ", 1)[1]
    parts = reaction.split(">")
    if len(parts) != 3:
        return None
    reactants, reagents, products = parts
    starting_smiles = split_smiles_side(reactants) + split_smiles_side(reagents)
    product_smiles = split_smiles_side(products)
    if not starting_smiles or not product_smiles:
        return None
    return starting_smiles, product_smiles


def split_smiles_side(side: str) -> list[str]:
    return [part.strip() for part in side.split(".") if part.strip()]


def canonical_smiles_set(mols) -> set[str]:
    out = set()
    for mol in mols:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            # The mechanism templates encode connectivity, not stereochemical outcome.
            out.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
        except Exception:
            continue
    return out


def apply_stage(
    current_state: tuple[Chem.Mol, ...],
    smirks_list: list[str],
) -> list[tuple[tuple[Chem.Mol, ...], tuple[Chem.Mol, ...]]]:
    next_states = []
    for smirks in smirks_list:
        rxn = AllChem.ReactionFromSmarts(smirks)
        if rxn is None:
            continue
        rxn.Initialize()
        n_templates = rxn.GetNumReactantTemplates()
        if n_templates > len(current_state):
            continue
        for idx in permutations(range(len(current_state)), n_templates):
            perm = tuple(current_state[i] for i in idx)
            untouched = tuple(
                current_state[i] for i in range(len(current_state)) if i not in idx
            )
            try:
                outcomes = rxn.RunReactants(perm)
            except Exception:
                continue
            for outcome in outcomes:
                clean = []
                ok = True
                for mol in outcome:
                    try:
                        Chem.SanitizeMol(mol)
                        clean.append(mol)
                    except Exception:
                        ok = False
                        break
                if ok:
                    stage_products = tuple(clean)
                    next_states.append((stage_products + untouched, stage_products))
    return next_states


def reaction_matches_mechanism(
    reactant_smiles: list[str],
    product_smiles: list[str],
    stages: dict[int, dict[str, list[str]]],
) -> bool:
    reactant_mols = [Chem.MolFromSmiles(smiles) for smiles in reactant_smiles if smiles]
    if not reactant_mols or any(mol is None for mol in reactant_mols):
        return False

    target = canonical_smiles_set(Chem.MolFromSmiles(smiles) for smiles in product_smiles if smiles)
    if not target:
        return False

    states: list[tuple[tuple[Chem.Mol, ...], tuple[Chem.Mol, ...]]] = [
        (tuple(reactant_mols), tuple())
    ]
    for stage_key in sorted(stages.keys()):
        templates = stages[stage_key]["Templates"]
        new_states = []
        for state, _last_stage_products in states:
            new_states.extend(apply_stage(state, templates))
        if not new_states:
            return False
        states = new_states

    for state, final_mechanism_products in states:
        final_state = canonical_smiles_set(state)
        generated_products = canonical_smiles_set(final_mechanism_products)
        if target.issubset(final_state) and target & generated_products:
            return True
    return False


def reaction_line_matches_mechanism(reaction_line: str, reaction_key: str) -> bool:
    components = parse_reaction_components(reaction_line)
    if components is None:
        return False
    reactant_smiles, product_smiles = components
    return reaction_matches_mechanism(
        reactant_smiles=reactant_smiles,
        product_smiles=product_smiles,
        stages=TASK10_MECHANISM_STAGES[reaction_key],
    )


def compute_ground_truth_indices(lines: list[str], reaction_key: str) -> tuple[list[int], int]:
    skipped = 0
    positives = []
    for line in lines:
        try:
            idx_str, reaction = line.split(" ", 1)
            idx = int(idx_str)
        except ValueError:
            skipped += 1
            continue
        try:
            if reaction_line_matches_mechanism(reaction, reaction_key):
                positives.append(idx)
        except Exception:
            skipped += 1
    return positives, skipped


def compute_ground_truth_indices_by_reaction(lines: list[str]) -> tuple[dict[str, list[int]], int]:
    indices_by_reaction = {}
    max_skipped = 0
    for reaction_key in REACTION_KEYS:
        indices, skipped = compute_ground_truth_indices(lines, reaction_key)
        indices_by_reaction[reaction_key] = indices
        max_skipped = max(max_skipped, skipped)
    return indices_by_reaction, max_skipped
