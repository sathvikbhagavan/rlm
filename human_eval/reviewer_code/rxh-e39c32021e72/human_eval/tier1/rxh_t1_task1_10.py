from tqdm import tqdm
from ..utils import canonicalize, load_reactions

target_prod = canonicalize("CC(C)(C)OC(=O)N1CCC(c2ccc(F)cn2)CC1")

reactions = load_reactions()

list_matches = []
for i, rxn in tqdm(enumerate(reactions)):
    if canonicalize(rxn.split('>')[2]) == target_prod:
        list_matches.append(str(i))

print(','.join(list_matches))




