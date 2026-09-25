from pathlib import Path
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles

DATA = Path(__file__).resolve().parents[1] / "reactionSmilesFigShareUSPTO2023_cleaned.txt"


def load_reactions(path=DATA):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def canonicalize(smiles):
    return MolToSmiles(MolFromSmiles(smiles))


if __name__ == "__main__":
    rxns = load_reactions()
    assert len(rxns) == 122456 and all([rxn.count(">") == 2 for rxn in rxns])
    print(len(rxns), rxns[0])
