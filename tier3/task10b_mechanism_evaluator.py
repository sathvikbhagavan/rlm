"""Mechanism evaluator for tier3 task10b mechanism-family questions.

This module intentionally keeps the hidden RDKit cascades out of the model prompts.
It is used to generate the hardcoded ground truth for task10b.
"""

from __future__ import annotations

from rdkit import Chem

from task10_mechanism_evaluator import (
    apply_stage,
    canonical_smiles_set,
    parse_reaction_components,
    reaction_matches_mechanism,
)

REACTION_KEYS = (
    "keto_alpha_alkylation",
    "base_catalyzed_transesterification",
    "grignard_carbonyl_addition_two_stage",
    "staudinger_reduction_without_duplicate_n2_step",
    "alpha_ketone_bromination",
)

TASK10B_GROUND_TRUTH_DEFINITION = (
    "RDKit cascade through selected mechanism stages; actual "
    "products must be a connectivity-equivalent subset of the final state and include "
    "at least one direct mechanism product"
)

KETO_ALPHA_ALKYLATION_STAGES = {
    0: {
        "Templates": [
            "[#8:1]=[#6:2]-[#6;H2:3]-[#6:4]=[#8:5]>>[#8:1]=[#6:2]-[#6;H1;-1:3]-[#6:4]=[#8:5]",
            "[#8:1]=[#6:2]-[#6;H3:3]>>[#8:1]=[#6:2]-[#6;H2;-1:3]",
            "[#8:1]=[#6:2]-[#6;H2:3]>>[#8:1]=[#6:2]-[#6;H1;-1:3]",
            "[#7:1]#[#6:2]-[#6;H3:3]>>[#7:1]#[#6:2]-[#6;H2;-1:3]",
            "[#7:1]#[#6:2]-[#6;H2:3]>>[#7:1]#[#6:2]-[#6;H1;-1:3]",
        ],
    },
    1: {
        "Templates": [
            "[#6;H2;-1:3].[F,Cl,Br,I:4]-[#6;+0:5]>>[#6;H2;+0:3]-[#6;+0:5].[F,Cl,Br,I;-1:4]",
            "[#6;H1;-1:3].[F,Cl,Br,I:4]-[#6;+0:5]>>[#6;H1;+0:3]-[#6;+0:5].[F,Cl,Br,I;-1:4]",
            "[#6;H0;-1:3].[F,Cl,Br,I:4]-[#6;+0:5]>>[#6;H0;+0:3]-[#6;+0:5].[F,Cl,Br,I;-1:4]",
            "[#6;H2;-1:3].[S:6]-[O:4]-[#6;+0:5]>>[#6;H2;+0:3]-[#6;+0:5].[S:6]-[O;-1:4]",
            "[#6;H1;-1:3].[S:6]-[O:4]-[#6;+0:5]>>[#6;H1;+0:3]-[#6;+0:5].[S:6]-[O;-1:4]",
            "[#6;H0;-1:3].[S:6]-[O:4]-[#6;+0:5]>>[#6;H0;+0:3]-[#6;+0:5].[S:6]-[O;-1:4]",
        ],
    },
}

BASE_CATALYZED_TRANSESTERIFICATION_STAGES = {
    0: {
        "Templates": [
            "[O;H1:5]-[#6:6]>>[O;H0;-1:5]-[#6:6]",
        ],
    },
    1: {
        "Templates": [
            "[O:1]=[C:2]-[O:3]-[#6:4].[O;H0;-1:5]-[#6:6]>>[O;H0;-1:1]-[C:2](-[O:3]-[#6:4])-[O;H0;+0:5]-[#6:6]",
        ],
    },
    2: {
        "Templates": [
            "[O;H0;-1:1]-[C:2](-[O:3]-[#6:4])-[O;H0;+0:5]-[#6:6]>>[O;H0;+0:1]=[C:2]-[O;H0;+0:5]-[#6:6].[O;H0;-1:3]-[#6:4]",
        ],
    },
    3: {
        "Templates": [
            "[O;H0;-1:3]-[#6:4]>>[O;H1;+0:3]-[#6:4]",
        ],
    },
}

GRIGNARD_CARBONYL_ADDITION_TWO_STAGE_STAGES = {
    0: {
        "Templates": [
            "[C:1]=[#8:2].[Cl,Br,I:3]-[Mg:4]-[*:5]>>[#8;-1:2]-[C:1]-[*:5].[Mg;+1:4](-[Cl,Br,I:3])",
            "[c:1]=[#8:2].[Cl,Br,I:3]-[Mg:4]-[*:5]>>[#8;-1:2]-[c:1]-[*:5].[Mg;+1:4](-[Cl,Br,I:3])",
        ],
    },
    1: {
        "Templates": [
            "[#8;-1:2]-[#6:1]-[O:5]>>[#8;+0:2]=[#6:1].[O;H0;-1:5]",
            "[#8;-1:2]-[#6:1]-[#6:5]>>[#8;H1;+0:2]-[#6:1]-[#6:5]",
        ],
    },
}

STAUDINGER_REDUCTION_STAGES = {
    0: {
        "Templates": [
            "[#7;+0:1]=[#7;+1:2]=[#7;-1:3].[P;D3:4]>>[#7;-1:1]-[#7;+0:2]=[#7;+0:3]-[P;+1:4]",
        ],
    },
    1: {
        "Templates": [
            "[#7;-1:1]-[#7;+0:2]=[#7;+0:3]-[P;+1:4]>>[P;+0:4]1-[#7;+0:1]-[#7;+0:2]=[#7;+0:3]1",
        ],
    },
    2: {
        "Templates": [
            "[P;+0:4]1-[#7:1]-[#7;+0:2]=[#7;+0:3]1>>[P;+0:4]=[#7:1].[#7;+0:2]#[#7;+0:3]",
        ],
    },
    3: {
        "Templates": [
            "[O;H2:1]>>[O;H1;-1:1]",
        ],
    },
    4: {
        "Templates": [
            "[P;+0:4]=[#7;H0:1].[O;H1;-1:5]>>[#7;H0;-1:1]-[P;H0;+0:4]-[O;H1;+0:5]",
        ],
    },
    5: {
        "Templates": [
            "[#7;H0;-1:1]-[P;H0;+0:4]-[O;H1;+0:5]>>[#7;H1;+0:1]-[P;H0;+0:4]-[O;H1;+0:5]",
        ],
    },
    6: {
        "Templates": [
            "[#7;H1;+0:1]-[P;H0;+0:4]-[O;H1;+0:5]>>[#7;H2;+1:1]-[P;H0;+0:4]-[O;H1;+0:5]",
        ],
    },
    7: {
        "Templates": [
            "[#7;H2;+1:1]-[P;H0;+0:4]-[O;H1;+0:5]>>[#7;H2;+1:1]-[P;H0;+0:4]-[O;H0;-1:5]",
        ],
    },
    8: {
        "Templates": [
            "[#7;H2;+1:1]-[P;H0;+0:4]-[O;H0;-1:5]>>[#7;H2;+0:1].[P;H0;+0:4]=[O;H0;+0:5]",
        ],
    },
}

ALPHA_KETONE_BROMINATION_STAGES = {
    0: {
        "Templates": [
            "[C;+0:1]-[C:2]=[O;H0;+0:3]>>[C;+0:1]-[C;+0:2]=[O;H1;+1:3]",
        ],
    },
    1: {
        "Templates": [
            "[#8:1]=[#6:2]-[#6;H2:3]-[#6:4]=[#8:5]>>[#8:1]=[#6:2]-[#6;H1;-1:3]-[#6:4]=[#8:5]",
            "[#8:1]=[#6:2]-[#6;H1:3]-[#6:4]=[#8:5]>>[#8:1]=[#6:2]-[#6;H0;-1:3]-[#6:4]=[#8:5]",
            "[#8:1]=[#6:2]-[#6;H3:3]>>[#8:1]=[#6:2]-[#6;H2;-1:3]",
            "[#8:1]=[#6:2]-[#6;H2:3]>>[#8:1]=[#6:2]-[#6;H1;-1:3]",
            "[#8:1]=[#6:2]-[#6;H1:3]>>[#8:1]=[#6:2]-[#6;H0;-1:3]",
            "[#7:1]#[#6:2]-[#6;H3:3]>>[#7:1]#[#6:2]-[#6;H2;-1:3]",
            "[#7:1]#[#6:2]-[#6;H2:3]>>[#7:1]#[#6:2]-[#6;H1;-1:3]",
            "[#8;H1;+1:1]=[#6:2]-[#6;H2:3]-[#6:4]=[#8:5]>>[#8;H1;+0:1]-[#6:2]=[#6;H1;+0:3]-[#6:4]=[#8:5]",
            "[#8;H1;+1:1]=[#6:2]-[#6;H1:3]-[#6:4]=[#8:5]>>[#8;H1;+0:1]-[#6:2]=[#6;H0;+0:3]-[#6:4]=[#8:5]",
            "[#8;H1;+1:1]=[#6:2]-[#6;H3:3]>>[#8;H1;+0:1]-[#6:2]=[#6;H2;+0:3]",
            "[#8;H1;+1:1]=[#6:2]-[#6;H2:3]>>[#8;H1;+0:1]-[#6:2]=[#6;H1;+0:3]",
            "[#8;H1;+1:1]=[#6:2]-[#6;H1:3]>>[#8;H1;+0:1]-[#6:2]=[#6;H0;+0:3]",
        ],
    },
    2: {
        "Templates": [
            "[C;+0:1]=[C:2]-[O;H1:3].[Br:4][Br:5]>>[Br;H0;-1:5].[Br:4][C;+0:1]-[C:2]=[O;H1;+1:3]",
            "[C;-1:1]-[C:2]=[O;H0:3].[Br:4][Br:5]>>[Br;H0;-1:5].[Br:4][C;+0:1]-[C:2]=[O;H0:3]",
        ],
    },
    3: {
        "Templates": [
            "[Br:4][C:1]-[C:2]=[O;H1;+1:3]>>[Br:4][C:1]-[C:2]=[O;H0;+0:3]",
        ],
    },
}

TASK10B_MECHANISM_STAGES = {
    "keto_alpha_alkylation": KETO_ALPHA_ALKYLATION_STAGES,
    "base_catalyzed_transesterification": BASE_CATALYZED_TRANSESTERIFICATION_STAGES,
    "grignard_carbonyl_addition_two_stage": GRIGNARD_CARBONYL_ADDITION_TWO_STAGE_STAGES,
    "staudinger_reduction_without_duplicate_n2_step": STAUDINGER_REDUCTION_STAGES,
    "alpha_ketone_bromination": ALPHA_KETONE_BROMINATION_STAGES,
}

TASK10B_DIRECT_PRODUCT_STAGE = {
    "base_catalyzed_transesterification": 2,
}


def reaction_matches_mechanism_with_direct_product_stage(
    reactant_smiles: list[str],
    product_smiles: list[str],
    stages: dict[int, dict[str, list[str]]],
    direct_product_stage: int,
) -> bool:
    reactant_mols = [Chem.MolFromSmiles(smiles) for smiles in reactant_smiles if smiles]
    if not reactant_mols or any(mol is None for mol in reactant_mols):
        return False

    target = canonical_smiles_set(
        Chem.MolFromSmiles(smiles) for smiles in product_smiles if smiles
    )
    if not target:
        return False

    def state_key(mols: tuple[Chem.Mol, ...]) -> tuple[str, ...]:
        return tuple(sorted(canonical_smiles_set(mols)))

    states: list[tuple[tuple[Chem.Mol, ...], tuple[Chem.Mol, ...]]] = [
        (tuple(reactant_mols), tuple())
    ]
    for stage_key in sorted(stages.keys()):
        templates = stages[stage_key]["Templates"]
        new_states = []
        seen_states = set()
        for state, key_stage_products in states:
            for next_state, stage_products in apply_stage(state, templates):
                if stage_key == direct_product_stage:
                    next_key_stage_products = stage_products
                else:
                    next_key_stage_products = key_stage_products
                dedupe_key = (
                    state_key(next_state),
                    state_key(next_key_stage_products),
                )
                if dedupe_key in seen_states:
                    continue
                seen_states.add(dedupe_key)
                new_states.append((next_state, next_key_stage_products))
        if not new_states:
            return False
        states = new_states

    for state, key_stage_products in states:
        final_state = canonical_smiles_set(state)
        generated_products = canonical_smiles_set(key_stage_products)
        if target.issubset(final_state) and target & generated_products:
            return True
    return False


def reaction_line_matches_mechanism(reaction_line: str, reaction_key: str) -> bool:
    components = parse_reaction_components(reaction_line)
    if components is None:
        return False
    reactant_smiles, product_smiles = components
    if reaction_key in TASK10B_DIRECT_PRODUCT_STAGE:
        return reaction_matches_mechanism_with_direct_product_stage(
            reactant_smiles=reactant_smiles,
            product_smiles=product_smiles,
            stages=TASK10B_MECHANISM_STAGES[reaction_key],
            direct_product_stage=TASK10B_DIRECT_PRODUCT_STAGE[reaction_key],
        )
    return reaction_matches_mechanism(
        reactant_smiles=reactant_smiles,
        product_smiles=product_smiles,
        stages=TASK10B_MECHANISM_STAGES[reaction_key],
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
