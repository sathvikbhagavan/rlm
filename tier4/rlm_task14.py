import argparse
import json
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
import os

from rdkit import Chem, RDLogger
from rlm import RLM
from rlm.tracing import init_tracing, using_tracing_attributes

try:
    import wandb
except ImportError:
    wandb = None

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
SEED = 42

NUM_QUESTIONS = 5
MAX_MOLECULE_FREQ = 200
MIN_PATH_REACTIONS = 3
MAX_PATH_REACTIONS = 5
MAX_LONGEST_PATH_REACTIONS = 7
MIN_HEAVY_ATOMS = 3
MAX_HEAVY_ATOMS = 90

# The first query is the task described in the prompt. Additional FG pairs can be
# added here without changing the evaluator.
FIXED_FG_QUERIES: list[tuple[str, str]] = [
    ("primary_alcohol", "tertiary_amide"),
    ("primary_alcohol", "carboxylic_acid"),
    ("alkyl_halide", "tertiary_amine"),
    ("ester", "tertiary_amide"),
    ("nitrile", "primary_amide")
]

FUNCTIONAL_GROUP_SMARTS: dict[str, list[str]] = {
    "primary_alcohol": ["[CX4H2][OX2H]"],
    "secondary_alcohol": ["[CX4H1]([#6])[OX2H]"],
    "tertiary_alcohol": ["[CX4H0]([#6])([#6])[OX2H]"],
    "aldehyde": ["[CX3H1](=O)[#6]"],
    "ketone": ["[#6][CX3](=O)[#6]"],
    "carboxylic_acid": ["[CX3](=O)[OX2H1]"],
    "ester": ["[CX3](=O)[OX2][#6]"],
    "acid_chloride": ["[CX3](=O)Cl"],
    "primary_amide": ["[CX3](=O)[NX3H2]"],
    "secondary_amide": ["[CX3](=O)[NX3H1][#6]"],
    "tertiary_amide": ["[CX3](=O)[NX3]([#6])[#6]"],
    "primary_amine": ["[NX3H2][#6]"],
    "secondary_amine": ["[NX3H1]([#6])[#6]"],
    "tertiary_amine": ["[NX3H0]([#6])([#6])[#6]"],
    "nitrile": ["[CX2]#N"],
    "alkyl_halide": ["[CX4][Cl,Br,I]"],
    "alkyl_sulfonate": ["[CX4][OX2]S(=O)(=O)[#6]"],
    "alkene": ["C=C"],
    "alkyne": ["C#C"],
}

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class ReactionRecord:
    index: int
    raw: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]


@dataclass(frozen=True)
class MoleculeStep:
    reaction_index: int
    from_smiles: str
    to_smiles: str


@dataclass(frozen=True)
class GroundTruthPath:
    source_fg: str
    target_fg: str
    objective: str
    reaction_indices: tuple[int, ...]
    molecule_chain: tuple[str, ...]
    node_groups: tuple[tuple[str, ...], ...]
    accepted_reaction_indices: tuple[tuple[int, ...], ...] = tuple()
    accepted_molecule_chains: tuple[tuple[str, ...], ...] = tuple()


def compile_fg_patterns() -> dict[str, list[Chem.Mol]]:
    patterns: dict[str, list[Chem.Mol]] = {}
    for label, smarts_list in FUNCTIONAL_GROUP_SMARTS.items():
        compiled = []
        for smarts in smarts_list:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                raise ValueError(f"Invalid SMARTS for {label}: {smarts}")
            compiled.append(patt)
        patterns[label] = compiled
    return patterns


FG_PATTERNS = compile_fg_patterns()


def canonicalize_components(smiles_str: str) -> list[str]:
    canonical: list[str] = []
    for smi in smiles_str.split("."):
        smi = smi.strip()
        if not smi:
            continue
        try:
            csmi = Chem.CanonSmiles(smi)
            if csmi:
                canonical.append(csmi)
        except Exception:
            continue
    return canonical


def split_reaction_line(indexed_line: str) -> tuple[int, str, str]:
    idx_str, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    if len(parts) < 2:
        raise ValueError(f"Malformed reaction line: {indexed_line[:80]}")
    return int(idx_str), parts[0].strip(), parts[-1].strip()


def heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def molecule_in_size_window(smiles: str) -> bool:
    heavy = heavy_atom_count(smiles)
    return MIN_HEAVY_ATOMS <= heavy <= MAX_HEAVY_ATOMS


def detect_functional_groups(smiles: str) -> tuple[str, ...]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tuple()
    found: list[str] = []
    for label, patterns in FG_PATTERNS.items():
        if any(mol.HasSubstructMatch(pattern) for pattern in patterns):
            found.append(label)
    return tuple(sorted(found))


def parse_dataset(dataset_path: str) -> dict[int, ReactionRecord]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]

    records: dict[int, ReactionRecord] = {}
    for i, raw in enumerate(raw_lines):
        indexed_line = f"{i} {raw}"
        try:
            idx, reactants_raw, products_raw = split_reaction_line(indexed_line)
        except Exception:
            continue
        reactants = tuple(canonicalize_components(reactants_raw))
        products = tuple(canonicalize_components(products_raw))
        if not reactants or not products:
            continue
        records[idx] = ReactionRecord(
            index=idx,
            raw=raw,
            reactants=reactants,
            products=products,
        )
    return records


def build_molecule_graph(
    records: dict[int, ReactionRecord],
    max_molecule_freq: int,
) -> tuple[dict[str, list[MoleculeStep]], dict[str, tuple[str, ...]], set[str]]:
    producers: dict[str, set[int]] = defaultdict(set)
    consumers: dict[str, set[int]] = defaultdict(set)
    all_molecules: set[str] = set()

    for rec in records.values():
        all_molecules.update(rec.reactants)
        all_molecules.update(rec.products)
        for product in rec.products:
            producers[product].add(rec.index)
        for reactant in rec.reactants:
            consumers[reactant].add(rec.index)

    frequent_molecules: set[str] = set()
    for smiles, idxs in producers.items():
        if len(idxs) > max_molecule_freq:
            frequent_molecules.add(smiles)
    for smiles, idxs in consumers.items():
        if len(idxs) > max_molecule_freq:
            frequent_molecules.add(smiles)

    group_cache = {
        smiles: detect_functional_groups(smiles)
        for smiles in all_molecules
        if smiles not in frequent_molecules and molecule_in_size_window(smiles)
    }

    graph: dict[str, list[MoleculeStep]] = defaultdict(list)
    for rec in records.values():
        reactants = [
            smi for smi in rec.reactants
            if smi in group_cache and smi not in frequent_molecules
        ]
        products = [
            smi for smi in rec.products
            if smi in group_cache and smi not in frequent_molecules
        ]
        for reactant in reactants:
            for product in products:
                if reactant == product:
                    continue
                graph[reactant].append(
                    MoleculeStep(
                        reaction_index=rec.index,
                        from_smiles=reactant,
                        to_smiles=product,
                    )
                )

    for edges in graph.values():
        edges.sort(key=lambda step: (step.reaction_index, step.to_smiles))

    return dict(graph), group_cache, frequent_molecules


def shortest_fg_path(
    graph: dict[str, list[MoleculeStep]],
    group_cache: dict[str, tuple[str, ...]],
    source_fg: str,
    target_fg: str,
    min_path_reactions: int,
    max_path_reactions: int,
) -> GroundTruthPath | None:
    if source_fg not in FUNCTIONAL_GROUP_SMARTS:
        raise ValueError(f"Unknown source functional group: {source_fg}")
    if target_fg not in FUNCTIONAL_GROUP_SMARTS:
        raise ValueError(f"Unknown target functional group: {target_fg}")

    starts = sorted(
        smiles for smiles, groups in group_cache.items()
        if source_fg in groups and target_fg not in groups
    )
    paths: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = [
        (smiles, tuple(), (smiles,)) for smiles in starts
    ]

    for _depth in range(max_path_reactions):
        next_paths: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = []
        solutions: list[tuple[tuple[int, ...], tuple[str, ...]]] = []

        for current, reaction_chain, molecule_chain in paths:
            for step in graph.get(current, []):
                if step.reaction_index in reaction_chain:
                    continue
                nxt = step.to_smiles
                if nxt in molecule_chain:
                    continue
                next_reactions = reaction_chain + (step.reaction_index,)
                next_molecules = molecule_chain + (nxt,)
                next_groups = group_cache.get(nxt, tuple())
                if (
                    len(next_reactions) >= min_path_reactions
                    and target_fg in next_groups
                    and source_fg not in next_groups
                ):
                    solutions.append((next_reactions, next_molecules))
                elif len(next_reactions) < max_path_reactions:
                    next_paths.append((nxt, next_reactions, next_molecules))

        if solutions:
            return build_ground_truth_from_solutions(
                solutions=solutions,
                source_fg=source_fg,
                target_fg=target_fg,
                group_cache=group_cache,
                objective="shortest",
            )

        paths = next_paths

    return None


def longest_fg_path(
    graph: dict[str, list[MoleculeStep]],
    group_cache: dict[str, tuple[str, ...]],
    source_fg: str,
    target_fg: str,
    min_path_reactions: int,
    max_path_reactions: int,
) -> GroundTruthPath | None:
    if source_fg not in FUNCTIONAL_GROUP_SMARTS:
        raise ValueError(f"Unknown source functional group: {source_fg}")
    if target_fg not in FUNCTIONAL_GROUP_SMARTS:
        raise ValueError(f"Unknown target functional group: {target_fg}")

    starts = sorted(
        smiles for smiles, groups in group_cache.items()
        if source_fg in groups and target_fg not in groups
    )
    paths: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = [
        (smiles, tuple(), (smiles,)) for smiles in starts
    ]
    solutions: list[tuple[tuple[int, ...], tuple[str, ...]]] = []

    for _depth in range(max_path_reactions):
        next_paths: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = []

        for current, reaction_chain, molecule_chain in paths:
            for step in graph.get(current, []):
                if step.reaction_index in reaction_chain:
                    continue
                nxt = step.to_smiles
                if nxt in molecule_chain:
                    continue
                next_reactions = reaction_chain + (step.reaction_index,)
                next_molecules = molecule_chain + (nxt,)
                next_groups = group_cache.get(nxt, tuple())
                if (
                    len(next_reactions) >= min_path_reactions
                    and target_fg in next_groups
                    and source_fg not in next_groups
                ):
                    solutions.append((next_reactions, next_molecules))
                if len(next_reactions) < max_path_reactions:
                    next_paths.append((nxt, next_reactions, next_molecules))

        paths = next_paths
        if not paths:
            break

    if not solutions:
        return None

    longest_length = max(len(reaction_chain) for reaction_chain, _ in solutions)
    longest_solutions = [
        solution
        for solution in solutions
        if len(solution[0]) == longest_length
    ]
    return build_ground_truth_from_solutions(
        solutions=longest_solutions,
        source_fg=source_fg,
        target_fg=target_fg,
        group_cache=group_cache,
        objective="longest",
    )


def build_ground_truth_from_solutions(
    solutions: list[tuple[tuple[int, ...], tuple[str, ...]]],
    source_fg: str,
    target_fg: str,
    group_cache: dict[str, tuple[str, ...]],
    objective: str,
) -> GroundTruthPath:
    solutions = sorted(set(solutions), key=lambda item: (item[0], item[1]))
    accepted_reactions = tuple(sorted({reaction_chain for reaction_chain, _ in solutions}))

    molecule_by_reaction: dict[tuple[int, ...], tuple[str, ...]] = {}
    for reaction_chain, molecule_chain in solutions:
        molecule_by_reaction.setdefault(reaction_chain, molecule_chain)
    accepted_molecules = tuple(molecule_by_reaction[rxns] for rxns in accepted_reactions)

    reaction_indices = accepted_reactions[0]
    molecule_chain = accepted_molecules[0]
    node_groups = tuple(group_cache.get(smiles, tuple()) for smiles in molecule_chain)
    return GroundTruthPath(
        source_fg=source_fg,
        target_fg=target_fg,
        objective=objective,
        reaction_indices=reaction_indices,
        molecule_chain=molecule_chain,
        node_groups=node_groups,
        accepted_reaction_indices=accepted_reactions,
        accepted_molecule_chains=accepted_molecules,
    )


def parse_response(response: str) -> tuple[int, ...]:
    text = response.strip()
    if not text or text.replace(" ", "") == "-1":
        return tuple()
    seen: set[int] = set()
    indices: list[int] = []
    for match in re.findall(r"\d+", text):
        idx = int(match)
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return tuple(indices)


def verify_predicted_path(
    reaction_indices: tuple[int, ...],
    source_fg: str,
    target_fg: str,
    records: dict[int, ReactionRecord],
    frequent_molecules: set[str],
    min_path_reactions: int,
) -> tuple[bool, str, tuple[str, ...], tuple[tuple[str, ...], ...]]:
    if not reaction_indices:
        return False, "no reaction indices returned", tuple(), tuple()
    if len(reaction_indices) < min_path_reactions:
        return False, f"chain has fewer than {min_path_reactions} reactions", tuple(), tuple()
    if len(set(reaction_indices)) != len(reaction_indices):
        return False, "reaction indices contain repeats", tuple(), tuple()

    chain_records: list[ReactionRecord] = []
    for reaction_idx in reaction_indices:
        rec = records.get(reaction_idx)
        if rec is None:
            return False, f"reaction index {reaction_idx} not found", tuple(), tuple()
        chain_records.append(rec)

    def is_allowed_node(smiles: str) -> bool:
        return smiles not in frequent_molecules and molecule_in_size_window(smiles)

    states: list[tuple[str, tuple[str, ...]]] = []
    for reactant in chain_records[0].reactants:
        if not is_allowed_node(reactant):
            continue
        groups = detect_functional_groups(reactant)
        if source_fg in groups and target_fg not in groups:
            states.append((reactant, (reactant,)))

    if not states:
        return False, f"first reaction has no allowed reactant with {source_fg}", tuple(), tuple()

    for pos, rec in enumerate(chain_records):
        next_states: list[tuple[str, tuple[str, ...]]] = []
        next_rec = chain_records[pos + 1] if pos + 1 < len(chain_records) else None
        next_reactants = set(next_rec.reactants) if next_rec is not None else set()

        for current_mol, molecule_chain in states:
            if current_mol not in rec.reactants:
                continue
            for product in rec.products:
                if not is_allowed_node(product):
                    continue
                if product in molecule_chain:
                    continue
                if next_rec is not None and product not in next_reactants:
                    continue
                product_groups = detect_functional_groups(product)
                extended_chain = molecule_chain + (product,)
                if next_rec is None:
                    if target_fg in product_groups and source_fg not in product_groups:
                        node_groups = tuple(
                            detect_functional_groups(smiles) for smiles in extended_chain
                        )
                        return True, "ok", extended_chain, node_groups
                else:
                    next_states.append((product, extended_chain))

        # Deduplicate equivalent molecule states while preserving deterministic order.
        deduped: dict[str, tuple[str, ...]] = {}
        for molecule, molecule_chain in next_states:
            deduped.setdefault(molecule, molecule_chain)
        states = [(molecule, molecule_chain) for molecule, molecule_chain in deduped.items()]
        if not states:
            return False, f"no component link through reaction {rec.index}", tuple(), tuple()

    return False, f"final product lacks {target_fg} or still contains {source_fg}", tuple(), tuple()


def lcs_length(pred: tuple[int, ...], gt: tuple[int, ...]) -> int:
    n = len(pred)
    m = len(gt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def precision_recall_f1(
    predicted: set[int], ground_truth: set[int]
) -> tuple[float, float, float]:
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_prediction(
    pred_rxns: tuple[int, ...],
    gt: GroundTruthPath,
    records: dict[int, ReactionRecord],
    frequent_molecules: set[str],
    min_path_reactions: int,
) -> dict[str, float | str]:
    valid, reason, inferred_mols, inferred_groups = verify_predicted_path(
        reaction_indices=pred_rxns,
        source_fg=gt.source_fg,
        target_fg=gt.target_fg,
        records=records,
        frequent_molecules=frequent_molecules,
        min_path_reactions=min_path_reactions,
    )
    accepted_reactions = set(gt.accepted_reaction_indices or (gt.reaction_indices,))
    index_match = pred_rxns in accepted_reactions
    pred_set = set(pred_rxns)

    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_lcs = 0
    best_norm_lcs = 0.0
    for gt_rxns in accepted_reactions:
        precision, recall, f1 = precision_recall_f1(pred_set, set(gt_rxns))
        lcs = lcs_length(pred_rxns, gt_rxns)
        denom = max(len(pred_rxns), len(gt_rxns), 1)
        norm_lcs = lcs / denom
        if (f1, norm_lcs, lcs) > (best_f1, best_norm_lcs, best_lcs):
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_lcs = lcs
            best_norm_lcs = norm_lcs

    objective_length_match = float(valid and len(pred_rxns) == len(gt.reaction_indices))
    return {
        "is_correct": float(index_match or objective_length_match),
        "valid_path": float(valid),
        "validity_reason": reason,
        "index_match": float(index_match),
        "objective_length_match": objective_length_match,
        "reaction_precision": best_precision,
        "reaction_recall": best_recall,
        "reaction_f1": best_f1,
        "reaction_lcs": float(best_lcs),
        "normalized_lcs": best_norm_lcs,
        "inferred_molecule_chain": json.dumps(list(inferred_mols)),
        "inferred_node_functional_groups": json.dumps([list(groups) for groups in inferred_groups]),
    }


def build_question(
    source_fg: str,
    target_fg: str,
    objective: str,
    min_path_reactions: int,
    max_path_reactions: int,
) -> str:
    return f"""
    Context: You are given a large set of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Each side may contain multiple molecules separated by dots (.).

    Task:
    Find the {objective} reaction chain, using reactions from the dataset, that converts a molecule
    containing functional group "{source_fg}" into a molecule containing functional group "{target_fg}".

    A valid reaction chain [r_0, r_1, ..., r_k] must admit at least one molecule component
    chain [m_0, m_1, ..., m_(k+1)] such that m_i is an exact canonical-SMILES reactant
    component of r_i and m_(i+1) is an exact canonical-SMILES product component of r_i.
    For adjacent reactions, the product component m_(i+1) must also be an exact
    canonical-SMILES reactant component of r_(i+1).

    Constraints:
    - m_0 must contain "{source_fg}" and must NOT already contain "{target_fg}".
    - m_(k+1) must contain "{target_fg}" and must NOT still contain "{source_fg}".
    - Do not repeat reaction indices or molecule nodes within the chain.
    - Use exact canonical SMILES equality for reaction edges; do not use substructure
      matching for molecule identity.
    - Ignore molecules that appear in more than {MAX_MOLECULE_FREQ} reactions as
      reactants or products.
    - Ignore molecule nodes with fewer than {MIN_HEAVY_ATOMS} or more than {MAX_HEAVY_ATOMS}
      heavy atoms.
    - Search paths of at least {min_path_reactions} and at most {max_path_reactions} reactions.
    - If several {objective} chains exist, any one valid {objective} chain is acceptable.

    Guidance:
    - Use RDKit for canonicalization and heavy atom counts.
    - You may write your own SMARTS or substructure checks to recognize the functional groups.
    - Split multi-component sides on dots (.) and canonicalize each component independently.
    - Skip malformed reactions or molecules that RDKit cannot parse.
    - DO NOT assume/simulate output of code. Wait for code execution and only then return.
    - DO NOT USE `FINAL` for writing a thought/comment.

    Output format:
    - Return ONLY the reaction indices in the chain.
    - Format must be a comma-separated list of integers (e.g., 60483,60620).
    - No other text, quotes, labels, punctuation, JSON, or formatting.

    If no chain exists, return -1.
    """


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task14",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_query(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Query must be formatted as source_fg:target_fg")
    source_fg, target_fg = [part.strip() for part in value.split(":", 1)]
    if source_fg not in FUNCTIONAL_GROUP_SMARTS:
        raise argparse.ArgumentTypeError(f"Unknown source functional group: {source_fg}")
    if target_fg not in FUNCTIONAL_GROUP_SMARTS:
        raise argparse.ArgumentTypeError(f"Unknown target functional group: {target_fg}")
    return source_fg, target_fg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RLM task 14 — functional-group transformation chains."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for backend (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DATASET_PATH,
        help=f"Path to USPTO reaction dataset (default: {DATASET_PATH}).",
    )
    parser.add_argument(
        "--query",
        action="append",
        type=parse_query,
        help=(
            "Functional-group query as source_fg:target_fg. "
            "Can be provided multiple times. Default: primary_alcohol:tertiary_amide."
        ),
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=NUM_QUESTIONS,
        help=(
            "Deprecated for the default set; all fixed queries are used when "
            "--query is not provided."
        ),
    )
    parser.add_argument(
        "--min-path-reactions",
        type=int,
        default=MIN_PATH_REACTIONS,
        help=f"Minimum path length to accept (default: {MIN_PATH_REACTIONS}).",
    )
    parser.add_argument(
        "--max-path-reactions",
        type=int,
        default=MAX_PATH_REACTIONS,
        help=f"Maximum shortest-path length to search (default: {MAX_PATH_REACTIONS}).",
    )
    parser.add_argument(
        "--max-longest-path-reactions",
        type=int,
        default=MAX_LONGEST_PATH_REACTIONS,
        help=(
            "Maximum longest-path length to search "
            f"(default: {MAX_LONGEST_PATH_REACTIONS})."
        ),
    )
    parser.add_argument(
        "--mine-only",
        action="store_true",
        help="Only compute and print RDKit ground truth; skip model evaluation.",
    )
    return parser.parse_args()


def summarize_gt(gt: GroundTruthPath) -> str:
    return json.dumps(
        {
            "source_fg": gt.source_fg,
            "target_fg": gt.target_fg,
            "objective": gt.objective,
            "path_length": len(gt.reaction_indices),
            f"{gt.objective}_length": len(gt.reaction_indices),
            f"num_{gt.objective}_reaction_chains": len(gt.accepted_reaction_indices),
            f"{gt.objective}_reaction_chains": [
                list(rxns) for rxns in gt.accepted_reaction_indices
            ],
            "example_reaction_indices": list(gt.reaction_indices),
            "example_molecule_chain": list(gt.molecule_chain),
            "example_node_functional_groups": [list(groups) for groups in gt.node_groups],
        },
        separators=(",", ":"),
    )


def main(
    model_name: str,
    dataset_path: str,
    queries: list[tuple[str, str]],
    num_questions: int,
    min_path_reactions: int,
    max_path_reactions: int,
    max_longest_path_reactions: int,
    mine_only: bool,
) -> None:
    records = parse_dataset(dataset_path)
    print(f"Loaded {len(records)} parsable reactions from {dataset_path}")

    graph, group_cache, frequent_molecules = build_molecule_graph(
        records=records,
        max_molecule_freq=MAX_MOLECULE_FREQ,
    )
    print(
        f"Built molecule graph with {len(graph)} source nodes, "
        f"{len(group_cache)} labeled molecule nodes, and {len(frequent_molecules)} frequent molecules ignored."
    )

    if not queries:
        if num_questions != len(FIXED_FG_QUERIES):
            print(
                f"num_questions={num_questions} ignored; using all "
                f"{len(FIXED_FG_QUERIES)} fixed functional-group queries."
            )
        queries = list(FIXED_FG_QUERIES)
    if min_path_reactions < 1:
        raise ValueError("--min-path-reactions must be at least 1.")
    if max_path_reactions < min_path_reactions:
        raise ValueError("--max-path-reactions must be >= --min-path-reactions.")
    if max_longest_path_reactions < min_path_reactions:
        raise ValueError("--max-longest-path-reactions must be >= --min-path-reactions.")

    print(f"Using {len(queries)} functional-group queries: {queries}")

    gt_paths: list[GroundTruthPath] = []
    for source_fg, target_fg in queries:
        path_specs = (
            ("shortest", max_path_reactions, shortest_fg_path),
            ("longest", max_longest_path_reactions, longest_fg_path),
        )
        for objective, max_reactions, path_finder in path_specs:
            gt = path_finder(
                graph=graph,
                group_cache=group_cache,
                source_fg=source_fg,
                target_fg=target_fg,
                min_path_reactions=min_path_reactions,
                max_path_reactions=max_reactions,
            )
            if gt is None:
                raise ValueError(
                    f"No {objective} ground-truth path found for {source_fg}->{target_fg} "
                    f"between {min_path_reactions} and {max_reactions} reactions."
                )
            gt_paths.append(gt)
            print(f"Ground truth [{objective} {source_fg}->{target_fg}]: {summarize_gt(gt)}")

    if mine_only:
        return

    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
    context = "\n".join(f"{i} {line}" for i, line in enumerate(raw_lines))

    run = None
    if wandb is None:
        print("wandb not installed; continuing without experiment logging.")
    else:
        run = wandb.init(
            project="RLMs-Task14",
            config={
                "MODEL_NAME": model_name,
                "backend": BACKEND,
                "model_name": model_name,
                "dataset_path": dataset_path,
                "num_questions": len(gt_paths),
                "fixed_fg_queries": queries,
                "max_molecule_freq": MAX_MOLECULE_FREQ,
                "min_heavy_atoms": MIN_HEAVY_ATOMS,
                "max_heavy_atoms": MAX_HEAVY_ATOMS,
                "min_path_reactions": min_path_reactions,
                "max_path_reactions": max_path_reactions,
                "max_longest_path_reactions": max_longest_path_reactions,
                "seed": SEED,
                "rlm_init_kwargs": rlm_init_kwargs,
                "task_description": "Functional-group transformation chains via RDKit SMARTS.",
            },
        )
        wandb.define_metric("sample_iteration")
        wandb.define_metric("sample/*", step_metric="sample_iteration", summary="none")

    macro_accuracy = 0.0
    macro_valid_path = 0.0
    macro_objective_length = 0.0
    macro_reaction_f1 = 0.0
    index_match_count = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, gt in enumerate(gt_paths):
        question = build_question(
            source_fg=gt.source_fg,
            target_fg=gt.target_fg,
            objective=gt.objective,
            min_path_reactions=min_path_reactions,
            max_path_reactions=(
                max_longest_path_reactions if gt.objective == "longest" else max_path_reactions
            ),
        )
        print(
            f"\nQuestion {i + 1}/{len(gt_paths)}: "
            f"{gt.objective} "
            f"{gt.source_fg}->{gt.target_fg}, "
            f"gt_{gt.objective}_len={len(gt.reaction_indices)}, "
            f"gt_{gt.objective}_chains={len(gt.accepted_reaction_indices)}"
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(gt_paths),
                "task": "functional_group_chain",
                "objective": gt.objective,
                "source_fg": gt.source_fg,
                "target_fg": gt.target_fg,
                "gt_objective_length": len(gt.reaction_indices),
                "gt_num_objective_reaction_chains": len(gt.accepted_reaction_indices),
            },
            tags=["run_rlms", "sample", "task14_FUNCTIONAL_GROUP_CHAIN"],
        ):
            completion = rlm.completion(prompt=context, root_prompt=question)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        pred_rxns = parse_response(response)
        scores = score_prediction(
            pred_rxns=pred_rxns,
            gt=gt,
            records=records,
            frequent_molecules=frequent_molecules,
            min_path_reactions=min_path_reactions,
        )
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        macro_accuracy += float(scores["is_correct"])
        macro_valid_path += float(scores["valid_path"])
        macro_objective_length += float(scores["objective_length_match"])
        macro_reaction_f1 += float(scores["reaction_f1"])
        index_match_count += int(scores["index_match"])

        print(f"Response [sample={i}]: {response[:500]}{'...' if len(response) > 500 else ''}")
        print(f"Predicted reactions [sample={i}]: {pred_rxns}")
        print(f"{gt.objective.title()} ground-truth chains [sample={i}]: {len(gt.accepted_reaction_indices)}")
        print(
            f"Metrics [sample={i}] -> correct={scores['is_correct']:.0f} "
            f"valid={scores['valid_path']:.0f} "
            f"{gt.objective}_len={scores['objective_length_match']:.0f} "
            f"index_match={scores['index_match']:.0f} "
            f"reaction_f1={scores['reaction_f1']:.4f} "
            f"reason={scores['validity_reason']}"
        )

        if wandb is not None:
            for metric in iteration_metrics:
                wandb.log(
                    {
                        "sample_iteration": metric["iteration"],
                        f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                        f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                        f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                    }
                )

        if wandb is not None and iteration_metrics:
            last_metric = iteration_metrics[-1]
            wandb.log(
                {
                    "sample_idx": i,
                    f"sample/{i}/objective": gt.objective,
                    f"sample/{i}/source_fg": gt.source_fg,
                    f"sample/{i}/target_fg": gt.target_fg,
                    f"sample/{i}/gt_objective_length": len(gt.reaction_indices),
                    f"sample/{i}/gt_num_objective_reaction_chains": len(gt.accepted_reaction_indices),
                    f"sample/{i}/gt_example_reaction_indices": ",".join(str(x) for x in gt.reaction_indices),
                    f"sample/{i}/pred_reaction_indices": ",".join(str(x) for x in pred_rxns),
                    f"sample/{i}/response_char_count": len(response),
                    f"sample/{i}/validity_reason": scores["validity_reason"],
                    f"sample/{i}/is_correct": scores["is_correct"],
                    f"sample/{i}/valid_path": scores["valid_path"],
                    f"sample/{i}/index_match": scores["index_match"],
                    f"sample/{i}/objective_length_match": scores["objective_length_match"],
                    f"sample/{i}/reaction_precision": scores["reaction_precision"],
                    f"sample/{i}/reaction_recall": scores["reaction_recall"],
                    f"sample/{i}/reaction_f1": scores["reaction_f1"],
                    f"sample/{i}/normalized_lcs": scores["normalized_lcs"],
                    f"sample/{i}/completion_prompt_char_count": len(context),
                    f"sample/{i}/final_total_input_tokens": last_metric["total_input_tokens"],
                    f"sample/{i}/final_total_output_tokens": last_metric["total_output_tokens"],
                    f"sample/{i}/final_total_tokens": last_metric["total_tokens"],
                    f"sample/{i}/iterations": len(iteration_metrics),
                    **(
                        {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                        if sample_cost_usd is not None
                        else {}
                    ),
                }
            )

    total = len(gt_paths)
    macro_accuracy = macro_accuracy / total if total else 0.0
    macro_valid_path = macro_valid_path / total if total else 0.0
    macro_objective_length = macro_objective_length / total if total else 0.0
    macro_reaction_f1 = macro_reaction_f1 / total if total else 0.0

    print(f"\n{'=' * 60}")
    print(f"Queries evaluated: {total}")
    print(f"Index match: {index_match_count}/{total}")
    print(f"Macro accuracy (valid objective-length path): {macro_accuracy:.4f}")
    print(f"Macro valid path: {macro_valid_path:.4f}")
    print(f"Macro objective-length match: {macro_objective_length:.4f}")
    print(f"Macro reaction F1: {macro_reaction_f1:.4f}")

    if run is not None and wandb is not None:
        run.summary["queries_evaluated"] = total
        run.summary["index_match_correct"] = index_match_count
        run.summary["macro_accuracy"] = macro_accuracy
        run.summary["macro_valid_path"] = macro_valid_path
        run.summary["macro_objective_length_match"] = macro_objective_length
        run.summary["macro_reaction_f1"] = macro_reaction_f1
        run.summary["samples_with_cost"] = samples_with_cost
        if samples_with_cost > 0:
            run.summary["total_cost_usd"] = total_cost_usd
            run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        queries=args.query or [],
        num_questions=args.num_questions,
        min_path_reactions=args.min_path_reactions,
        max_path_reactions=args.max_path_reactions,
        max_longest_path_reactions=args.max_longest_path_reactions,
        mine_only=args.mine_only,
    )
