from collections import defaultdict, deque

from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles
from tqdm import tqdm

from ..utils import load_reactions

RDLogger.DisableLog('rdApp.*')

cache = {}


def canon_side(side_smiles):

    list_canon = []
    for x in side_smiles.split('.'):
        if not x:
            continue
        if x not in cache:
            mol = MolFromSmiles(x)
            cache[x] = MolToSmiles(mol) if mol is not None else None
        if cache[x] is not None:
            list_canon.append(cache[x])

    return set(list_canon)


reactions = load_reactions()


# --- step 1: where does each molecule appear as a product / as a reactant? ---

producers = defaultdict(list)
# ponytail: indices arrive ascending, so maxlen=3 keeps the 3 LARGEST -- enough to test ">= 3 after p"
consumers = defaultdict(lambda: deque(maxlen=3))

for i, rxn in tqdm(enumerate(reactions)):
    sides = rxn.split('>')
    for mol in canon_side(sides[0]):
        consumers[mol].append(i)
    for mol in canon_side(sides[2]):
        producers[mol].append(i)


produced_once = {mol: idx[0] for mol, idx in producers.items() if len(idx) == 1}
consumed_thrice = {mol: idx for mol, idx in consumers.items() if len(idx) == 3}

print(f"distinct product molecules:  {len(producers)}")
print(f"produced exactly once:       {len(produced_once)}")
print(f"distinct reactant molecules: {len(consumers)}")
print(f"consumed at least 3 times:   {len(consumed_thrice)}")


# --- step 2: produced once, then consumed 3+ times strictly after that ---

list_results = []
for mol, prod_idx in produced_once.items():
    idx = consumed_thrice.get(mol)
    if idx is not None and idx[0] > prod_idx:
        list_results.append(mol)

print(f"matches: {len(list_results)}")
print('\n'.join(sorted(list_results)) or "-1")
