"""Answer-free executable predicates for the Tier-3 oracle control.

This module intentionally contains no ground-truth indices, answer counts, dataset
paths, or sampling helpers.  It is safe to name in an oracle prompt: the supplied
functions define chemical membership for one reaction at a time.
"""

from __future__ import annotations

import hashlib
from functools import cache
from itertools import permutations
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions
from task10_mechanism_evaluator import reaction_line_matches_mechanism
from task23_stereocenter_evaluator import reaction_creates_stereocenter_from_achiral

ORACLE_PREDICATE_VERSION = "tier3-v1"
ORACLE_PREDICATE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

TASK6_AMIDE_COUPLING_SMIRKS: dict[str, str] = {
    "acyl_chloride_with_primary_amine": "[*:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[Cl].[#7;H2;!$(N[O,N]);D1;+0:5]>>[*:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[#7;H1;D2;+0:5]",
    "carboxylic_acid_with_primary_amine": "[CX3;+0:2](=[O;H0;D1;+0:3])-[O;H1;D1;+0].[#7;H2;D1;+0:5]>>[CX3;+0:2](=[O;H0;D1;+0:3])-[#7;H1;D2;+0:5]",
    "ester_with_primary_amine": "[#6:1]-[C;H0;D3;+0:2](=[O;!$(OC(C)(C)C);H0;D1;+0:3])-[O;H0;D2;+0].[#7;H2;D1;+0:5]>>[#6:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[#7;H1;D2;+0:5]",
    "ester_with_secondary_amine": "[#6:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[O;!$(OC(C)(C)C);H0;D2;+0].[#7;H1;D2;+0:5]>>[#6:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[#7;H0;D3;+0:5]",
}


def parse_indexed_reaction(indexed_line: str) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    reaction = indexed_line.split(" ", 1)[1] if " " in indexed_line else indexed_line
    parts = reaction.split(">")
    if len(parts) != 3:
        return [], []
    reactants = [Chem.MolFromSmiles(smiles) for smiles in parts[0].split(".") if smiles]
    products = [Chem.MolFromSmiles(smiles) for smiles in parts[2].split(".") if smiles]
    return [mol for mol in reactants if mol is not None], [
        mol for mol in products if mol is not None
    ]


def canonical_smiles_set(molecules: list[Chem.Mol]) -> set[str]:
    return {Chem.MolToSmiles(molecule) for molecule in molecules}


@cache
def task6_reaction_query(reaction_key: str) -> rdChemReactions.ChemicalReaction:
    smirks = TASK6_AMIDE_COUPLING_SMIRKS[reaction_key]
    query = rdChemReactions.ReactionFromSmarts(smirks)
    if query is None:
        raise ValueError(f"Failed to parse reaction SMARTS for {reaction_key}")
    query.Initialize()
    return query


def task6_reaction_matches(indexed_line: str, reaction_key: str) -> bool:
    """Apply the benchmark's exact Task-6 SMIRKS membership rule to one row."""

    query = task6_reaction_query(reaction_key)
    reactants, products = parse_indexed_reaction(indexed_line)
    actual_products = canonical_smiles_set(products)
    reactant_count = query.GetNumReactantTemplates()
    for selected in permutations(reactants, reactant_count):
        try:
            generated_sets = query.RunReactants(selected)
        except Exception:
            continue
        for generated_set in generated_sets:
            generated: set[str] = set()
            for molecule in generated_set:
                try:
                    Chem.SanitizeMol(molecule)
                    generated.add(Chem.MolToSmiles(molecule))
                except Exception:
                    continue
            if generated and generated.issubset(actual_products):
                return True
    return False


def task10_reaction_matches(indexed_line: str, reaction_key: str) -> bool:
    """Apply the benchmark's exact staged mechanism predicate to one row."""

    reaction = indexed_line.split(" ", 1)[1] if " " in indexed_line else indexed_line
    return reaction_line_matches_mechanism(reaction, reaction_key)


def task23_reaction_matches(indexed_line: str) -> bool:
    """Apply the benchmark's exact assigned-stereocenter predicate to one row."""

    reaction = indexed_line.split(" ", 1)[1] if " " in indexed_line else indexed_line
    return reaction_creates_stereocenter_from_achiral(reaction)


def task6_oracle_guidance(reaction_key: str) -> str:
    return f"""
    Oracle predicate condition:
    - The chemical abstraction is supplied and authoritative; do not replace it with a heuristic.
    - For every context row call:
      `from oracle_predicates import task6_reaction_matches`
      `task6_reaction_matches(indexed_line, {reaction_key!r})`
    - Return exactly the indices of rows for which the function returns True.
    - The helper contains no answer indices or corpus-derived answer counts.
    """


def task10_oracle_guidance(reaction_key: str) -> str:
    return f"""
    Oracle predicate condition:
    - The validated staged mechanism abstraction is supplied and authoritative.
    - For every context row call:
      `from oracle_predicates import task10_reaction_matches`
      `task10_reaction_matches(indexed_line, {reaction_key!r})`
    - Return exactly the indices of rows for which the function returns True.
    - The helper contains no answer indices or corpus-derived answer counts.
    """


def task23_oracle_guidance() -> str:
    return """
    Oracle predicate condition:
    - The validated stereochemistry abstraction is supplied and authoritative.
    - For every context row call:
      `from oracle_predicates import task23_reaction_matches`
      `task23_reaction_matches(indexed_line)`
    - Return exactly the indices of rows for which the function returns True.
    - The helper contains no answer indices or corpus-derived answer counts.
    """
