import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import MolWt
from tqdm import tqdm

from ..utils import canonicalize, load_reactions


def heaviest_species(side_smiles):

    list_mols = [m for m in (MolFromSmiles(x) for x in side_smiles.split('.') if x) if m is not None and m.GetNumAtoms()]

    if len(list_mols) == 0:
        raise ValueError

    else:
        return max([MolWt(x) for x in list_mols])



reactions = load_reactions()

print(reactions[3])
exit()

list_results = []
for i, rxn in tqdm(enumerate(reactions)):
    try:
        weight_diff = heaviest_species(rxn.split('>')[2])-heaviest_species(rxn.split('>')[0])
        if weight_diff>218:
            list_results.append(str(i))
    except Exception as e:
        print(f"Problem with {i}: {e}")

print(','.join(list_results))

