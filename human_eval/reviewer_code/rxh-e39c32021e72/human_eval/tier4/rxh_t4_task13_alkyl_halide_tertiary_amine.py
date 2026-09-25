import pickle
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import RDLogger
from rdkit.Chem import MolFromSmarts, MolFromSmiles, MolToSmiles
from tqdm import tqdm

from ..utils import load_reactions

RDLogger.DisableLog('rdApp.*')

ALKYL_HALIDE = MolFromSmarts('[CX4][F,Cl,Br,I]')
TERT_AMINE = MolFromSmarts('[NX3;H0;!$(NC=[O,S,N]);!$(NS(=O)=O);!$(N[N,O,S])]([#6])([#6])[#6]')

CHAIN_LEN = 7
CACHE = Path('scratch_t13.pkl')


def parse_all():

    cache = {}
    reactions = load_reactions()
    parsed = []
    freq = Counter()

    for rxn in tqdm(reactions):
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
                        mol.HasSubstructMatch(ALKYL_HALIDE),
                        mol.HasSubstructMatch(TERT_AMINE),
                    )
                if cache[x] is not None:
                    species.add(cache[x][0])
            out.append(species)

        parsed.append(tuple(out))
        freq.update(out[0] | out[1])

    info = {v[0]: v[1:] for v in cache.values() if v is not None}
    # frequency + heavy-atom filters from the spec
    ok = {s for s, n in freq.items() if n <= 200 and 3 <= info[s][0] <= 90}

    filtered = [None if x is None else (x[0] & ok, x[1] & ok) for x in parsed]
    return filtered, info


if CACHE.exists():
    filtered, info = pickle.loads(CACHE.read_bytes())
else:
    filtered, info = parse_all()
    CACHE.write_bytes(pickle.dumps((filtered, info)))


def starts_chain(i):
    r = filtered[i]
    return r is not None and any(info[s][1] and not info[s][2] for s in r[0])


def ends_chain(i):
    r = filtered[i]
    return r is not None and any(info[s][2] and not info[s][1] for s in r[1])


# --- backward pruning: reach[k][i] = a k-reaction chain can start at i and end validly ---

consumers = defaultdict(list)
for i, f in enumerate(filtered):
    if f:
        for s in f[0]:
            consumers[s].append(i)


def successors(i):
    out = set()
    for s in filtered[i][1]:
        out.update(consumers.get(s, ()))
    out.discard(i)
    return out


n = len(filtered)
reach = [[False] * n for _ in range(CHAIN_LEN + 1)]
for i in range(n):
    reach[1][i] = ends_chain(i)

for k in range(2, CHAIN_LEN + 1):
    for i in range(n):
        if filtered[i]:
            reach[k][i] = any(reach[k - 1][j] for j in successors(i))
    print(f'  reach[{k}]: {sum(reach[k]):,} nodes')


# --- enumerate, pruned by reach, indices kept distinct ---

list_results = []


def walk(chain, used):
    depth = len(chain)
    if depth == CHAIN_LEN:
        list_results.append(','.join(str(k) for k in chain))
        return
    for j in successors(chain[-1]):
        if j not in used and reach[CHAIN_LEN - depth][j]:
            chain.append(j)
            used.add(j)
            walk(chain, used)
            chain.pop()
            used.discard(j)


for i in tqdm(range(n)):
    if reach[CHAIN_LEN][i] and starts_chain(i):
        walk([i], {i})

print(f'chains: {len(list_results)}')
print('\n'.join(sorted(list_results)) or "-1")
