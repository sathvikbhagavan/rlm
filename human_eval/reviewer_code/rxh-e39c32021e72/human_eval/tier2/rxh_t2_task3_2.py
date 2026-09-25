from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from tqdm import tqdm

from ..utils import load_reactions

RDLogger.DisableLog('rdApp.*')


def ring_counts(side_smiles):
    mols = [m for m in (MolFromSmiles(x) for x in side_smiles.split('.') if x)
            if m is not None and m.GetNumAtoms()]
    if not mols:
        raise ValueError("no valid molecules on this side")
    return [CalcNumRings(m) for m in mols]


reactions = load_reactions()

list_results = []
for i, rxn in tqdm(enumerate(reactions)):
    try:
        sides = rxn.split('>')
        if max(ring_counts(sides[2])) - min(ring_counts(sides[0])) == 2:
            list_results.append(str(i))
    except Exception:
        continue

print(','.join(list_results) or "-1")
