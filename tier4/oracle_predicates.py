"""Answer-free chemistry predicates for the Tier-4 oracle control.

The functions annotate one molecule or one reaction.  They deliberately do not
construct reaction graphs, search paths, pair events, sample contexts, or expose
ground-truth answers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from rdkit import Chem

ORACLE_PREDICATE_VERSION = "tier4-v1"
ORACLE_PREDICATE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

TASK13_FUNCTIONAL_GROUP_SMARTS: dict[str, tuple[str, ...]] = {
    "primary_alcohol": ("[CX4H2][OX2H]",),
    "secondary_alcohol": ("[CX4H1]([#6])[OX2H]",),
    "tertiary_alcohol": ("[CX4H0]([#6])([#6])[OX2H]",),
    "aldehyde": ("[CX3H1](=O)[#6]",),
    "ketone": ("[#6][CX3](=O)[#6]",),
    "carboxylic_acid": ("[CX3](=O)[OX2H1]",),
    "ester": ("[CX3](=O)[OX2][#6]",),
    "acid_chloride": ("[CX3](=O)Cl",),
    "primary_amide": ("[CX3](=O)[NX3H2]",),
    "secondary_amide": ("[CX3](=O)[NX3H1][#6]",),
    "tertiary_amide": ("[CX3](=O)[NX3]([#6])[#6]",),
    "primary_amine": ("[NX3H2][#6]",),
    "secondary_amine": ("[NX3H1]([#6])[#6]",),
    "tertiary_amine": ("[NX3H0]([#6])([#6])[#6]",),
    "nitrile": ("[CX2]#N",),
    "alkyl_halide": ("[CX4][Cl,Br,I]",),
    "alkyl_sulfonate": ("[CX4][OX2]S(=O)(=O)[#6]",),
    "alkene": ("C=C",),
    "alkyne": ("C#C",),
}

TASK14_PROTECTED_SMARTS: dict[str, tuple[str, ...]] = {
    "Boc_N": ("[NX3][CX3](=O)[OX2][C;X4]([CH3])([CH3])[CH3]",),
    "benzyl_O_N": ("[O,N]Cc1ccccc1",),
}
MIN_HEAVY_ATOMS = 3
MAX_HEAVY_ATOMS = 90


def compile_patterns(definitions: dict[str, tuple[str, ...]]) -> dict[str, tuple[Chem.Mol, ...]]:
    compiled: dict[str, tuple[Chem.Mol, ...]] = {}
    for label, smarts_values in definitions.items():
        maybe_patterns = tuple(Chem.MolFromSmarts(smarts) for smarts in smarts_values)
        if any(pattern is None for pattern in maybe_patterns):
            raise ValueError(f"Invalid oracle SMARTS for {label}")
        compiled[label] = cast(tuple[Chem.Mol, ...], maybe_patterns)
    return compiled


TASK13_PATTERNS = compile_patterns(TASK13_FUNCTIONAL_GROUP_SMARTS)
TASK14_PATTERNS = compile_patterns(TASK14_PROTECTED_SMARTS)


def task13_detect_functional_groups(smiles: str) -> tuple[str, ...]:
    """Return benchmark-defined functional groups present in one molecule."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ()
    return tuple(
        sorted(
            label
            for label, patterns in TASK13_PATTERNS.items()
            if any(molecule.HasSubstructMatch(pattern) for pattern in patterns)
        )
    )


def canonical_components(side: str) -> tuple[str, ...]:
    values: list[str] = []
    for smiles in side.split("."):
        if not smiles:
            continue
        try:
            values.append(Chem.CanonSmiles(smiles))
        except Exception:
            continue
    return tuple(values)


def task14_has_protecting_group(smiles: str, pg_label: str) -> bool:
    molecule = Chem.MolFromSmiles(smiles)
    return molecule is not None and any(
        molecule.HasSubstructMatch(pattern) for pattern in TASK14_PATTERNS[pg_label]
    )


def task14_in_size_window(smiles: str) -> bool:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False
    return MIN_HEAVY_ATOMS <= molecule.GetNumHeavyAtoms() <= MAX_HEAVY_ATOMS


def task14_stripped_scaffold_keys(smiles: str, pg_label: str) -> tuple[str, ...]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ()
    keys: set[str] = set()
    for pattern in TASK14_PATTERNS[pg_label]:
        for match in molecule.GetSubstructMatches(pattern):
            editable = Chem.RWMol(molecule)
            for atom_index in sorted(match[1:], reverse=True):
                editable.RemoveAtom(atom_index)
            try:
                stripped = editable.GetMol()
                Chem.SanitizeMol(stripped)
                keys.add(Chem.MolToSmiles(stripped, canonical=True))
            except Exception:
                continue
    return tuple(sorted(keys))


@dataclass(frozen=True)
class ProtectionEvent:
    direction: str
    scaffold_key: str


def task14_protection_events(indexed_line: str, pg_label: str) -> tuple[ProtectionEvent, ...]:
    """Classify install/remove events in one row without pairing across the corpus."""

    reaction = indexed_line.split(" ", 1)[1] if " " in indexed_line else indexed_line
    parts = reaction.split(">")
    if len(parts) != 3:
        return ()
    reactants = tuple(s for s in canonical_components(parts[0]) if task14_in_size_window(s))
    products = tuple(s for s in canonical_components(parts[2]) if task14_in_size_window(s))
    events: set[ProtectionEvent] = set()
    protected_reactants = [s for s in reactants if task14_has_protecting_group(s, pg_label)]
    protected_products = [s for s in products if task14_has_protecting_group(s, pg_label)]
    for free_smiles in reactants:
        for protected_smiles in protected_products:
            if free_smiles in task14_stripped_scaffold_keys(protected_smiles, pg_label):
                events.add(ProtectionEvent("install", free_smiles))
    for protected_smiles in protected_reactants:
        stripped = task14_stripped_scaffold_keys(protected_smiles, pg_label)
        for free_smiles in products:
            if free_smiles in stripped:
                events.add(ProtectionEvent("remove", free_smiles))
    return tuple(sorted(events, key=lambda event: (event.direction, event.scaffold_key)))


def task13_oracle_guidance() -> str:
    return """
    Oracle predicate condition:
    - Functional-group recognition is supplied and authoritative; do not replace it with heuristics.
    - Classify every canonical molecule with:
      `from oracle_predicates import task13_detect_functional_groups`
      `task13_detect_functional_groups(canonical_smiles)`
    - You must still parse the corpus, apply all stated filters, construct exact-SMILES links,
      enumerate the reaction chains, and format the answer yourself.
    - The helper contains no reaction indices, paths, corpus-derived counts, or graph-search code.
    """


def task14_oracle_guidance(pg_label: str) -> str:
    return f"""
    Oracle predicate condition:
    - Protecting-group event recognition is supplied and authoritative; do not replace it with heuristics.
    - Classify every context row with:
      `from oracle_predicates import task14_protection_events`
      `task14_protection_events(indexed_line, {pg_label!r})`
    - Each returned event has `direction` and `scaffold_key`. You must still scan the corpus,
      join install/remove events on scaffold, enforce index ordering, and format the answer yourself.
    - The helper contains no reaction indices, pairs, corpus-derived counts, or pairing code.
    """
