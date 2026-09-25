from tqdm import tqdm
from ..utils import canonicalize, load_reactions


target_prod = canonicalize("CC(C)(C)OC(=O)c1ccnc(-c2ccc3nccc(N)c3c2)n1")

reactions = load_reactions()
print(reactions[0], len(reactions))


list_matches = []
for i, rxn in tqdm(enumerate(reactions)):
    if canonicalize(rxn.split('>')[2]) == target_prod:
        list_matches.append(str(i))

print(','.join(list_matches))

