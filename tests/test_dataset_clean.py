from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from dataset.clean import canonicalize_side, clean_dataset, clean_reaction_smiles


def test_clean_reaction_canonicalizes_sorts_and_removes_atom_maps() -> None:
    cleaned = clean_reaction_smiles("[CH3:1]O.CC>O.C>CCO")

    assert cleaned is not None
    assert cleaned.smiles == "CC.CO>C.O>CCO"


def test_clean_reaction_preserves_stereochemistry() -> None:
    cleaned = clean_reaction_smiles("N[C@@H](C)C(=O)O>>N[C@H](C)CO")

    assert cleaned is not None
    assert "@@" in cleaned.smiles
    assert "@" in cleaned.smiles


@pytest.mark.parametrize("side", ["C..O", ".C", "C.", "not-smiles"])
def test_canonicalize_side_rejects_invalid_components(side: str) -> None:
    assert canonicalize_side(side, allow_empty=False) is None


def test_clean_dataset_rejects_invalid_and_deduplicates(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.txt"
    output_path = tmp_path / "cleaned.txt"
    input_path.write_text("CO.CC>O>CCO\nCC.CO>O>CCO\ninvalid\nC>not-smiles>CO\n\n")

    stats = clean_dataset(input_path, output_path, quiet_rdkit=True)

    assert output_path.read_text() == "CC.CO>O>CCO\n"
    assert stats.total == 5
    assert stats.written == 1
    assert stats.blank == 1
    assert stats.malformed == 1
    assert stats.unparseable == 1
    assert stats.duplicate == 1
