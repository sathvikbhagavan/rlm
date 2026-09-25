from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings
from tqdm import tqdm

from ..utils import load_reactions

RDLogger.DisableLog('rdApp.*')


def aromatic_rings(side_smiles):

    list_mols_raw = [MolFromSmiles(x) for x in side_smiles.split('.') if x]
    list_mols = [x for x in list_mols_raw if x is not None and x.GetNumAtoms()]

    if len(list_mols) == 0:
        raise ValueError

    else:
        return sum([CalcNumAromaticRings(x) for x in list_mols])


reactions = load_reactions()

list_results = []
for i, rxn in tqdm(enumerate(reactions)):
    try:
        sides = rxn.split('>')
        new_aromatic_rings = aromatic_rings(sides[2])-aromatic_rings(sides[0])
        if new_aromatic_rings==5:
            list_results.append(str(i))
    except Exception:
        continue

print(','.join(list_results) or "-1")
