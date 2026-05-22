from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, RDLogger


DEFAULT_INPUT_PATH = Path("/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt")
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_PATH.with_name(
    f"{DEFAULT_INPUT_PATH.stem}_cleaned{DEFAULT_INPUT_PATH.suffix}"
)


@dataclass
class CleanStats:
    total: int = 0
    written: int = 0
    blank: int = 0
    malformed: int = 0
    unparseable: int = 0
    duplicate: int = 0


@dataclass(frozen=True)
class CleanedReaction:
    smiles: str
    dedupe_key: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


def canonicalize_molecule(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_side(side: str, *, allow_empty: bool) -> tuple[str, ...] | None:
    side = side.strip()
    if not side:
        return () if allow_empty else None
    canonical_molecules: list[str] = []
    for molecule_smiles in side.split("."):
        molecule_smiles = molecule_smiles.strip()
        if not molecule_smiles:
            return None
        canonical_smiles = canonicalize_molecule(molecule_smiles)
        if canonical_smiles is None:
            return None
        canonical_molecules.append(canonical_smiles)
    return tuple(sorted(canonical_molecules))


def clean_reaction_smiles(reaction_smiles: str) -> CleanedReaction | None:
    parts = reaction_smiles.strip().split(">")
    if len(parts) != 3:
        return None
    reactants = canonicalize_side(parts[0], allow_empty=False)
    reagents = canonicalize_side(parts[1], allow_empty=True)
    products = canonicalize_side(parts[2], allow_empty=False)
    if reactants is None or reagents is None or products is None:
        return None
    cleaned_smiles = f"{'.'.join(reactants)}>{'.'.join(reagents)}>{'.'.join(products)}"
    return CleanedReaction(
        smiles=cleaned_smiles,
        dedupe_key=(reactants, reagents, products),
    )


def clean_dataset(input_path: Path, output_path: Path, *, quiet_rdkit: bool) -> CleanStats:
    if quiet_rdkit:
        RDLogger.DisableLog("rdApp.*")
    stats = CleanStats()
    seen: set[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = set()
    with input_path.open("r", encoding="utf-8") as input_file, output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for line in input_file:
            stats.total += 1
            reaction_smiles = line.strip()
            if not reaction_smiles:
                stats.blank += 1
                continue
            if reaction_smiles.count(">") != 2:
                stats.malformed += 1
                continue
            cleaned = clean_reaction_smiles(reaction_smiles)
            if cleaned is None:
                stats.unparseable += 1
                continue
            if cleaned.dedupe_key in seen:
                stats.duplicate += 1
                continue
            seen.add(cleaned.dedupe_key)
            output_file.write(f"{cleaned.smiles}\n")
            stats.written += 1
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanitize, atom-map-strip, canonicalize, and deduplicate full reaction SMILES"
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--show-rdkit-errors",
        action="store_true",
        help="Do not suppress RDKit parse/sanitization log messages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = clean_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        quiet_rdkit=not args.show_rdkit_errors,
    )
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Total lines: {stats.total}")
    print(f"Written: {stats.written}")
    print(f"Blank: {stats.blank}")
    print(f"Malformed reaction SMILES: {stats.malformed}")
    print(f"Unparseable/sanitization failures: {stats.unparseable}")
    print(f"Duplicates dropped: {stats.duplicate}")


if __name__ == "__main__":
    main()
