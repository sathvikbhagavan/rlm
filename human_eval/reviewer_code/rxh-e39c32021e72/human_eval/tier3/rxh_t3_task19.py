from rdkit import RDLogger
from rdkit.Chem import MolFromSmarts, MolFromSmiles
from tqdm import tqdm

from ..utils import load_reactions

RDLogger.DisableLog('rdApp.*')

QUINOLINE = MolFromSmarts('c1ccc2ncccc2c1')


def has_quinoline(side_smiles):

    list_mols_raw = [MolFromSmiles(x) for x in side_smiles.split('.') if x]

    if len(list_mols_raw) == 0 or any(x is None for x in list_mols_raw):
        raise ValueError

    else:
        return any(x.HasSubstructMatch(QUINOLINE) for x in list_mols_raw)


reactions = load_reactions()

list_results = []
for i, rxn in tqdm(enumerate(reactions)):
    try:
        sides = rxn.split('>')
        if has_quinoline(sides[2]) and not has_quinoline(sides[0]):
            list_results.append(str(i))
    except Exception:
        continue

print(','.join(list_results) or "-1")
