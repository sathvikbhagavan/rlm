from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles
from tqdm import tqdm

from ..utils import load_reactions, canonicalize

RDLogger.DisableLog('rdApp.*')


HATU = canonicalize("CN(C)C(=[N+](C)C)On1c2c(cccn2)nn1")
PF6 = canonicalize("F[P-](F)(F)(F)(F)F")
T3P = canonicalize("CCCP1(=O)OP(=O)(OP(=O)(O1)CCC)CCC")


def reagent_set(reagent_smiles):
    list_mols_raw = [MolFromSmiles(x) for x in reagent_smiles.split('.') if x]

    if any(x is None for x in list_mols_raw):
        raise ValueError

    else:
        return {MolToSmiles(x) for x in list_mols_raw}


reactions = load_reactions()

list_results = []
for i, rxn in tqdm(enumerate(reactions)):
    try:
        sides = rxn.split('>')
        reagents = reagent_set(sides[1])

        if (HATU in reagents and PF6 in reagents) or T3P in reagents:
            list_results.append(str(i))
    except Exception:
        continue

print(','.join(list_results) or "-1")
