"""One-pass scan of candidate protecting groups for install/remove pair yield."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rlm.codeact_helpers import load_lines
from task14_protecting_group_graph import (
    MIN_HEAVY_ATOMS,
    MAX_HEAVY_ATOMS,
    ProtectingGroupSpec,
    free_scaffold_key,
    heavy_atom_count,
    parse_records_from_lines,
)

RDLogger.DisableLog("rdApp.*")

CANDIDATES: tuple[ProtectingGroupSpec, ...] = (
    ProtectingGroupSpec("Boc_N", "amine", ("[NX3][CX3](=O)[OX2][C;X4]([CH3])([CH3])[CH3]",), ("Boc",), "Boc"),
    ProtectingGroupSpec("Cbz_N", "amine", ("[NX3][CX3](=O)[OX2]Cc1ccccc1",), ("Cbz",), "Cbz"),
    ProtectingGroupSpec("Fmoc_N", "amine", ("[NX3][CX3](=O)[OX2]CC1c2ccccc2-c2ccccc21",), ("Fmoc",), "Fmoc"),
    ProtectingGroupSpec("Alloc_N", "amine", ("[NX3][CX3](=O)[OX2]CC=C",), ("Alloc",), "Alloc"),
    ProtectingGroupSpec("benzyl_O_N", "alcohol_or_amine", ("[O,N]Cc1ccccc1",), ("Bn",), "benzyl"),
    ProtectingGroupSpec("PMB_O", "alcohol", ("[OX2]Cc1ccc(OC)cc1",), ("PMB",), "PMB"),
    ProtectingGroupSpec("THP_O", "alcohol", ("[OX2]C1CCCCO1",), ("THP",), "THP"),
    ProtectingGroupSpec("MOM_O", "alcohol", ("[OX2]COC",), ("MOM",), "MOM"),
    ProtectingGroupSpec("Acetyl_O_N", "alcohol_or_amine", ("[O,N]C(=O)C",), ("Ac",), "acetyl"),
    ProtectingGroupSpec("TMS_O", "alcohol", ("[OX2][Si](C)(C)C",), ("TMS",), "TMS"),
    ProtectingGroupSpec("TBS_O", "alcohol", ("[OX2][Si](C)(C)C(C)(C)C",), ("TBS",), "TBS"),
    ProtectingGroupSpec("TES_O", "alcohol", ("[OX2][Si](CC)(CC)CC",), ("TES",), "TES"),
    ProtectingGroupSpec("silyl_O_broad", "alcohol", ("[OX2][Si]",), ("silyl",), "silyl"),
    ProtectingGroupSpec("Tr_O_N", "alcohol_or_amine", ("[O,N]C(c1ccccc1)(c1ccccc1)c1ccccc1",), ("Tr",), "trityl"),
    ProtectingGroupSpec("SEM_O", "alcohol", ("[OX2]CC[Si](C)(C)C",), ("SEM",), "SEM"),
    ProtectingGroupSpec("Ts_N", "amine", ("[NX3]S(=O)(=O)c1ccc(C)cc1",), ("Ts",), "tosyl"),
    ProtectingGroupSpec("Ms_O", "alcohol", ("[OX2]S(=O)(=O)C",), ("Ms",), "mesyl"),
    ProtectingGroupSpec("Tf_O", "alcohol", ("[OX2]S(=O)(=O)C(F)(F)F",), ("Tf",), "triflate"),
    ProtectingGroupSpec("Bz_O", "alcohol", ("[OX2]C(=O)c1ccccc1",), ("Bz",), "benzoate"),
    ProtectingGroupSpec("carbamate_N_broad", "amine", ("[NX3]C(=O)O",), ("carbamate",), "carbamate"),
)


@dataclass(frozen=True)
class CompiledSpec:
    spec: ProtectingGroupSpec
    patterns: tuple[Chem.Mol, ...]


def compile_candidates() -> list[CompiledSpec]:
    compiled: list[CompiledSpec] = []
    for spec in CANDIDATES:
        patterns = tuple(Chem.MolFromSmarts(smarts) for smarts in spec.protected_smarts)
        compiled.append(CompiledSpec(spec=spec, patterns=patterns))
    return compiled


def stripped_keys(smiles: str, patterns: tuple[Chem.Mol, ...]) -> tuple[str, ...]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tuple()
    keys: set[str] = set()
    for pattern in patterns:
        for match in mol.GetSubstructMatches(pattern):
            atoms_to_remove = sorted(match[1:], reverse=True)
            editable = Chem.RWMol(mol)
            for atom_idx in atoms_to_remove:
                editable.RemoveAtom(atom_idx)
            try:
                stripped = editable.GetMol()
                Chem.SanitizeMol(stripped)
                keys.add(Chem.MolToSmiles(stripped, canonical=True))
            except Exception:
                continue
    return tuple(sorted(keys))


def main() -> None:
    compiled = compile_candidates()
    lines = load_lines("/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt")
    records = parse_records_from_lines(lines)

    events: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    heavy_cache: dict[str, bool] = {}
    has_pg_cache: dict[tuple[str, str], bool] = {}
    stripped_cache: dict[tuple[str, str], tuple[str, ...]] = {}
    free_key_cache: dict[str, str | None] = {}

    def in_window(smiles: str) -> bool:
        if smiles not in heavy_cache:
            heavy = heavy_atom_count(smiles)
            heavy_cache[smiles] = MIN_HEAVY_ATOMS <= heavy <= MAX_HEAVY_ATOMS
        return heavy_cache[smiles]

    def has_label(smiles: str, label: str, patterns: tuple[Chem.Mol, ...]) -> bool:
        key = (smiles, label)
        if key not in has_pg_cache:
            mol = Chem.MolFromSmiles(smiles)
            has_pg_cache[key] = (
                mol is not None and any(mol.HasSubstructMatch(pattern) for pattern in patterns)
            )
        return has_pg_cache[key]

    def stripped_for(smiles: str, label: str, patterns: tuple[Chem.Mol, ...]) -> tuple[str, ...]:
        key = (smiles, label)
        if key not in stripped_cache:
            stripped_cache[key] = stripped_keys(smiles, patterns)
        return stripped_cache[key]

    def free_key(smiles: str) -> str | None:
        if smiles not in free_key_cache:
            free_key_cache[smiles] = free_scaffold_key(smiles)
        return free_key_cache[smiles]

    for rec in records.values():
        reactants = [smi for smi in rec.reactants if in_window(smi)]
        products = [smi for smi in rec.products if in_window(smi)]
        if not reactants or not products:
            continue

        for item in compiled:
            label = item.spec.label
            reactant_pg = [smi for smi in reactants if has_label(smi, label, item.patterns)]
            product_pg = [smi for smi in products if has_label(smi, label, item.patterns)]
            if not reactant_pg and not product_pg:
                continue

            for free_smi in reactants:
                fk = free_key(free_smi)
                if fk is None:
                    continue
                for protected_smi in product_pg:
                    if fk in stripped_for(protected_smi, label, item.patterns):
                        events[label].append((rec.index, "install", fk))

            for protected_smi in reactant_pg:
                protected_keys = stripped_for(protected_smi, label, item.patterns)
                if not protected_keys:
                    continue
                for free_smi in products:
                    fk = free_key(free_smi)
                    if fk is None:
                        continue
                    if fk in protected_keys:
                        events[label].append((rec.index, "remove", fk))

    rows: list[tuple[str, int, int, int, int]] = []
    for item in compiled:
        label = item.spec.label
        by_scaffold: dict[str, dict[str, set[int]]] = defaultdict(
            lambda: {"install": set(), "remove": set()}
        )
        for idx, direction, scaffold in events[label]:
            by_scaffold[scaffold][direction].add(idx)

        both = 0
        pairs = 0
        for grouped in by_scaffold.values():
            installs = grouped["install"]
            removals = grouped["remove"]
            if installs and removals:
                both += 1
                if any(i < r for i in installs for r in removals):
                    pairs += 1
        rows.append((label, len(events[label]), len(by_scaffold), both, pairs))

    rows.sort(key=lambda row: (-row[4], -row[1]))
    print("label\tevents\tscaffolds\tboth_dirs\tvalid_pairs")
    for row in rows:
        print("\t".join(map(str, row)))


if __name__ == "__main__":
    main()
