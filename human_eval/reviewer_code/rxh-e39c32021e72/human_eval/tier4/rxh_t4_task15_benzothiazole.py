import pickle
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import RDLogger
from rdkit.Chem import MolFromSmarts, MolFromSmiles, MolToSmiles
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from tqdm import tqdm

from ..utils import load_reactions

RDLogger.DisableLog('rdApp.*')

BENZOTHIAZOLE = MolFromSmarts('c1ccc2scnc2c1')
CACHE = Path('scratch_t15.pkl')

HEAVY, RINGS, BT = 0, 1, 2


def parse_all():

    cache = {}
    parsed = []
    freq = Counter()

    for rxn in tqdm(load_reactions()):
        sides = rxn.split('>')
        if len(sides) != 3:
            parsed.append(None)
            continue

        out = []
        for side in (sides[0], sides[2]):
            species = set()
            for x in side.split('.'):
                if not x:
                    continue
                if x not in cache:
                    mol = MolFromSmiles(x)
                    cache[x] = None if mol is None else (
                        MolToSmiles(mol),
                        mol.GetNumHeavyAtoms(),
                        CalcNumRings(mol),
                        mol.HasSubstructMatch(BENZOTHIAZOLE),
                    )
                if cache[x] is not None:
                    species.add(cache[x][0])
            out.append(species)

        parsed.append(tuple(out))
        freq.update(out[0] | out[1])

    info = {v[0]: v[1:] for v in cache.values() if v is not None}
    ok = {s for s, n in freq.items() if n <= 200 and 3 <= info[s][HEAVY] <= 90}

    filtered = [None if x is None else (x[0] & ok, x[1] & ok) for x in parsed]
    return filtered, info


if CACHE.exists():
    filtered, info = pickle.loads(CACHE.read_bytes())
else:
    filtered, info = parse_all()
    CACHE.write_bytes(pickle.dumps((filtered, info)))


# --- which reactions produce a given species ---

producers = defaultdict(list)
for i, f in enumerate(filtered):
    if f:
        for s in f[1]:
            producers[s].append(i)


def predecessors(mol):
    """(reaction index, reactant component) pairs where that reaction makes `mol`."""
    for i in producers.get(mol, ()):
        for s in filtered[i][0]:
            if s != mol:
                yield i, s


# --- walk back 3 reactions from every benzothiazole-bearing product ---

list_results = []

for m3 in tqdm([s for s in producers if info[s][BT]]):
    for r2, m2 in predecessors(m3):
        if info[m2][BT]:
            continue
        for r1, m1 in predecessors(m2):
            if info[m1][BT]:
                continue
            for r0, m0 in predecessors(m1):
                # m_0 acyclic, no benzothiazole; all reactions and molecules distinct
                if info[m0][BT] or info[m0][RINGS] != 0:
                    continue
                if len({r0, r1, r2}) != 3 or len({m0, m1, m2, m3}) != 4:
                    continue
                list_results.append(([r0, r1, r2], [m0, m1, m2, m3]))

print(f'valid chains: {len(list_results)}')
if list_results:
    chain, mols = min(list_results)
    print(','.join(str(k) for k in chain))
    for k, m in enumerate(mols):
        print(f'   m_{k}: {m}')
else:
    print(-1)
