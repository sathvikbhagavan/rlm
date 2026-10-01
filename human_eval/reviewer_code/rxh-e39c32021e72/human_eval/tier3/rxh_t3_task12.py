from collections import Counter

from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

from ..utils import load_reactions

RDLogger.DisableLog('rdApp.*')


def co_bond_counts(side_smiles):

    list_mols_raw = [MolFromSmiles(x) for x in side_smiles.split('.') if x]

    if len(list_mols_raw) == 0 or any(x is None for x in list_mols_raw):
        raise ValueError

    else:
        counts = Counter()
        for mol in list_mols_raw:
            for bond in mol.GetBonds():
                symbols = {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
                if symbols == {'C', 'O'}:
                    counts[bond.GetBondType()] += 1
        return counts


reactions = load_reactions()

list_results = []
for i, rxn in tqdm(enumerate(reactions)):
    try:
        sides = rxn.split('>')
        counts_react = co_bond_counts(sides[0])
        counts_prod = co_bond_counts(sides[2])

        if counts_react-counts_prod:
            list_results.append(str(i))
    except Exception:
        continue

print(','.join(list_results) or "-1")
