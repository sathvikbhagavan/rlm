from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from tqdm import tqdm

from ..utils import load_reactions

RDLogger.DisableLog('rdApp.*')


def valid_mols(side_smiles):

    list_mols_raw = [MolFromSmiles(x) for x in side_smiles.split('.') if x]
    list_mols = [x for x in list_mols_raw if x is not None and x.GetNumAtoms()]

    if len(list_mols) == 0:
        raise ValueError

    else:
        return list_mols


reactions = load_reactions()

list_results = []
for i, rxn in tqdm(enumerate(reactions)):
    try:
        sides = rxn.split('>')
        mols_react = valid_mols(sides[0])
        mols_prod = valid_mols(sides[2])

        weight_diff = max([MolWt(x) for x in mols_prod])-max([MolWt(x) for x in mols_react])
        ring_diff = max([CalcNumRings(x) for x in mols_prod])-min([CalcNumRings(x) for x in mols_react])

        if weight_diff>100 and ring_diff==2:
            list_results.append(str(i))
    except Exception:
        continue

print(','.join(list_results) or "-1")
