import argparse
import json
import os
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass

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
MAX_ACCEPTED_CHAINS = 200

DEFAULT_RING_QUERIES: tuple[str, ...] = (
    "quinoline",
    "indole",
    "benzofuran",
    "benzothiazole",
    "benzimidazole",
)

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
class RingSystemSpec:
    label: str
    smarts: tuple[str, ...]
    aliases: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class MoleculeAnnotation:
    acyclic: bool
    ring_systems: tuple[str, ...]


@dataclass(frozen=True)
class GroundTruthPath:
    ring_system: str
    objective: str
    reaction_indices: tuple[int, ...]
    molecule_chain: tuple[str, ...]
    node_ring_systems: tuple[tuple[str, ...], ...]
    accepted_reaction_indices: tuple[tuple[int, ...], ...] = tuple()
    accepted_molecule_chains: tuple[tuple[str, ...], ...] = tuple()


BUILTIN_RING_SYSTEMS: tuple[RingSystemSpec, ...] = (
    RingSystemSpec(
        label="quinoline",
        smarts=(
            "c1ccc2ncccc2c1",
            "c1ccc2cnccc2c1",
        ),
        aliases=("quinoline", "isoquinoline", "benzopyridine"),
        description="Fused benzene-pyridine ring systems, including quinoline and isoquinoline orientation.",
    ),
    RingSystemSpec(
        label="indole",
        smarts=(
            "c1ccc2[nH]ccc2c1",
            "c1ccc2[nH0]ccc2c1",
        ),
        aliases=("indole", "benzopyrrole"),
        description="Fused benzene-pyrrole ring system, including N-H and N-substituted indoles.",
    ),
    RingSystemSpec(
        label="benzofuran",
        smarts=("c1ccc2occc2c1",),
        aliases=("benzofuran",),
        description="Fused benzene-furan ring system.",
    ),
    RingSystemSpec(
        label="benzothiazole",
        smarts=("c1ccc2scnc2c1",),
        aliases=("benzothiazole",),
        description="Fused benzene-thiazole ring system.",
    ),
    RingSystemSpec(
        label="benzimidazole",
        smarts=(
            "c1ccc2[nH]cnc2c1",
            "c1ccc2[nH0]cnc2c1",
        ),
        aliases=("benzimidazole",),
        description="Fused benzene-imidazole ring system.",
    ),
)


def compile_ring_patterns(
    ring_systems: dict[str, RingSystemSpec],
) -> dict[str, list[Chem.Mol]]:
    patterns: dict[str, list[Chem.Mol]] = {}
    for label, spec in ring_systems.items():
        compiled = []
        for smarts in spec.smarts:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                raise ValueError(f"Invalid SMARTS for {label}: {smarts}")
            compiled.append(patt)
        patterns[label] = compiled
    return patterns


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


def annotate_molecule(
    smiles: str,
    ring_patterns: dict[str, list[Chem.Mol]],
) -> MoleculeAnnotation | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ring_info = mol.GetRingInfo()
    found = [
        label
        for label, patterns in ring_patterns.items()
        if any(mol.HasSubstructMatch(pattern) for pattern in patterns)
    ]
    return MoleculeAnnotation(
        acyclic=ring_info.NumRings() == 0,
        ring_systems=tuple(sorted(found)),
    )


def parse_dataset(dataset_path: str) -> dict[int, ReactionRecord]:
    with open(dataset_path, encoding="utf-8") as f:
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


def build_molecule_graphs(
    records: dict[int, ReactionRecord],
    ring_patterns: dict[str, list[Chem.Mol]],
    max_molecule_freq: int,
) -> tuple[
    dict[str, list[MoleculeStep]],
    dict[str, list[MoleculeStep]],
    dict[str, MoleculeAnnotation],
    set[str],
]:
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

    annotations: dict[str, MoleculeAnnotation] = {}
    for smiles in all_molecules:
        if smiles in frequent_molecules or not molecule_in_size_window(smiles):
            continue
        annotation = annotate_molecule(smiles, ring_patterns)
        if annotation is not None:
            annotations[smiles] = annotation

    forward_graph: dict[str, list[MoleculeStep]] = defaultdict(list)
    reverse_graph: dict[str, list[MoleculeStep]] = defaultdict(list)
    for rec in records.values():
        reactants = [
            smi for smi in rec.reactants
            if smi in annotations and smi not in frequent_molecules
        ]
        products = [
            smi for smi in rec.products
            if smi in annotations and smi not in frequent_molecules
        ]
        for reactant in reactants:
            for product in products:
                if reactant == product:
                    continue
                step = MoleculeStep(
                    reaction_index=rec.index,
                    from_smiles=reactant,
                    to_smiles=product,
                )
                forward_graph[reactant].append(step)
                reverse_graph[product].append(step)

    for edges in forward_graph.values():
        edges.sort(key=lambda step: (step.reaction_index, step.to_smiles))
    for edges in reverse_graph.values():
        edges.sort(key=lambda step: (step.reaction_index, step.from_smiles))

    return dict(forward_graph), dict(reverse_graph), annotations, frequent_molecules


def shortest_ring_construction_path(
    reverse_graph: dict[str, list[MoleculeStep]],
    annotations: dict[str, MoleculeAnnotation],
    ring_system: str,
    min_path_reactions: int,
    max_path_reactions: int,
) -> GroundTruthPath | None:
    targets = sorted(
        smiles for smiles, annotation in annotations.items()
        if ring_system in annotation.ring_systems
    )
    paths: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = [
        (smiles, tuple(), (smiles,)) for smiles in targets
    ]

    for _depth in range(max_path_reactions):
        next_paths: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = []
        solutions: list[tuple[tuple[int, ...], tuple[str, ...]]] = []

        for current, backward_reactions, backward_molecules in paths:
            for step in reverse_graph.get(current, []):
                if step.reaction_index in backward_reactions:
                    continue
                prev = step.from_smiles
                if prev in backward_molecules:
                    continue
                next_reactions = backward_reactions + (step.reaction_index,)
                next_molecules = backward_molecules + (prev,)
                prev_annotation = annotations.get(prev)
                if prev_annotation is None:
                    continue
                if ring_system in prev_annotation.ring_systems:
                    continue

                if (
                    len(next_reactions) >= min_path_reactions
                    and prev_annotation.acyclic
                ):
                    solutions.append(
                        (
                            tuple(reversed(next_reactions)),
                            tuple(reversed(next_molecules)),
                        )
                    )
                    if len(solutions) >= MAX_ACCEPTED_CHAINS:
                        break
                elif len(next_reactions) < max_path_reactions:
                    next_paths.append((prev, next_reactions, next_molecules))
            if len(solutions) >= MAX_ACCEPTED_CHAINS:
                break

        if solutions:
            return build_ground_truth_from_solutions(
                solutions=solutions,
                ring_system=ring_system,
                annotations=annotations,
                objective="shortest",
            )

        paths = next_paths

    return None


def longest_ring_construction_path(
    reverse_graph: dict[str, list[MoleculeStep]],
    annotations: dict[str, MoleculeAnnotation],
    ring_system: str,
    min_path_reactions: int,
    max_path_reactions: int,
) -> GroundTruthPath | None:
    targets = sorted(
        smiles for smiles, annotation in annotations.items()
        if ring_system in annotation.ring_systems
    )
    paths: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = [
        (smiles, tuple(), (smiles,)) for smiles in targets
    ]
    longest_solutions: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    longest_length = 0

    for _depth in range(max_path_reactions):
        next_paths: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = []
        depth_solutions: list[tuple[tuple[int, ...], tuple[str, ...]]] = []

        for current, backward_reactions, backward_molecules in paths:
            for step in reverse_graph.get(current, []):
                if step.reaction_index in backward_reactions:
                    continue
                prev = step.from_smiles
                if prev in backward_molecules:
                    continue
                next_reactions = backward_reactions + (step.reaction_index,)
                next_molecules = backward_molecules + (prev,)
                prev_annotation = annotations.get(prev)
                if prev_annotation is None:
                    continue
                if ring_system in prev_annotation.ring_systems:
                    continue

                if (
                    len(next_reactions) >= min_path_reactions
                    and prev_annotation.acyclic
                ):
                    depth_solutions.append(
                        (
                            tuple(reversed(next_reactions)),
                            tuple(reversed(next_molecules)),
                        )
                    )
                elif len(next_reactions) < max_path_reactions:
                    next_paths.append((prev, next_reactions, next_molecules))

        if depth_solutions:
            depth_length = len(depth_solutions[0][0])
            if depth_length > longest_length:
                longest_length = depth_length
                longest_solutions = depth_solutions[:MAX_ACCEPTED_CHAINS]
            elif depth_length == longest_length and len(longest_solutions) < MAX_ACCEPTED_CHAINS:
                remaining = MAX_ACCEPTED_CHAINS - len(longest_solutions)
                longest_solutions.extend(depth_solutions[:remaining])

        paths = next_paths
        if not paths:
            break

    if not longest_solutions:
        return None

    return build_ground_truth_from_solutions(
        solutions=longest_solutions,
        ring_system=ring_system,
        annotations=annotations,
        objective="longest",
    )


def build_ground_truth_from_solutions(
    solutions: list[tuple[tuple[int, ...], tuple[str, ...]]],
    ring_system: str,
    annotations: dict[str, MoleculeAnnotation],
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
    node_ring_systems = tuple(
        annotations[smiles].ring_systems for smiles in molecule_chain
    )
    return GroundTruthPath(
        ring_system=ring_system,
        objective=objective,
        reaction_indices=reaction_indices,
        molecule_chain=molecule_chain,
        node_ring_systems=node_ring_systems,
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
    ring_system: str,
    records: dict[int, ReactionRecord],
    ring_patterns: dict[str, list[Chem.Mol]],
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

    annotation_cache: dict[str, MoleculeAnnotation | None] = {}

    def cached_annotation(smiles: str) -> MoleculeAnnotation | None:
        if smiles not in annotation_cache:
            annotation_cache[smiles] = annotate_molecule(smiles, ring_patterns)
        return annotation_cache[smiles]

    def is_allowed_node(smiles: str) -> bool:
        return smiles not in frequent_molecules and molecule_in_size_window(smiles)

    states: list[tuple[str, tuple[str, ...]]] = []
    for reactant in chain_records[0].reactants:
        if not is_allowed_node(reactant):
            continue
        annotation = cached_annotation(reactant)
        if annotation is not None and annotation.acyclic and ring_system not in annotation.ring_systems:
            states.append((reactant, (reactant,)))

    if not states:
        return False, "first reaction has no allowed acyclic reactant", tuple(), tuple()

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
                annotation = cached_annotation(product)
                if annotation is None:
                    continue
                extended_chain = molecule_chain + (product,)
                if next_rec is None:
                    if ring_system in annotation.ring_systems:
                        node_ring_systems: list[tuple[str, ...]] = []
                        for smiles in extended_chain:
                            node_annotation = cached_annotation(smiles)
                            if node_annotation is not None:
                                node_ring_systems.append(node_annotation.ring_systems)
                        return True, "ok", extended_chain, tuple(node_ring_systems)
                else:
                    if ring_system in annotation.ring_systems:
                        continue
                    next_states.append((product, extended_chain))

        deduped: dict[str, tuple[str, ...]] = {}
        for molecule, molecule_chain in next_states:
            deduped.setdefault(molecule, molecule_chain)
        states = [(molecule, molecule_chain) for molecule, molecule_chain in deduped.items()]
        if not states:
            return False, f"no component link through reaction {rec.index}", tuple(), tuple()

    return False, f"final product lacks {ring_system}", tuple(), tuple()


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
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_prediction(
    pred_rxns: tuple[int, ...],
    gt: GroundTruthPath,
    records: dict[int, ReactionRecord],
    ring_patterns: dict[str, list[Chem.Mol]],
    frequent_molecules: set[str],
    min_path_reactions: int,
) -> dict[str, float | str]:
    valid, reason, inferred_mols, inferred_ring_systems = verify_predicted_path(
        reaction_indices=pred_rxns,
        ring_system=gt.ring_system,
        records=records,
        ring_patterns=ring_patterns,
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
        "inferred_node_ring_systems": json.dumps([list(labels) for labels in inferred_ring_systems]),
    }


def build_question(
    spec: RingSystemSpec,
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
    Find the {objective} reaction chain, using reactions from the dataset, that builds the ring system
    "{spec.label}" from a non-cyclic precursor.

    Ring system:
    - label: {spec.label}
    - aliases: {", ".join(spec.aliases)}
    - description: {spec.description}
    - SMARTS hints: {", ".join(spec.smarts)}

    A valid reaction chain [r_0, r_1, ..., r_k] must admit at least one molecule component
    chain [m_0, m_1, ..., m_(k+1)] such that m_i is an exact canonical-SMILES reactant
    component of r_i and m_(i+1) is an exact canonical-SMILES product component of r_i.
    For adjacent reactions, the product component m_(i+1) must also be an exact
    canonical-SMILES reactant component of r_(i+1).

    Constraints:
    - m_0 must be non-cyclic: RDKit ring count must be zero.
    - m_0 must NOT already contain the "{spec.label}" ring system.
    - Intermediate tracked molecule components m_1 through m_k must NOT contain
      the "{spec.label}" ring system; the ring system should first appear in m_(k+1).
    - m_(k+1) must contain the "{spec.label}" ring system.
    - Do not repeat reaction indices or molecule nodes within the chain.
    - Use exact canonical SMILES equality for reaction edges; do not use substructure
      matching for molecule identity.
    - Use SMARTS/substructure matching only to recognize the requested ring system.
    - Ignore molecules that appear in more than {MAX_MOLECULE_FREQ} reactions as
      reactants or products.
    - Ignore molecule nodes with fewer than {MIN_HEAVY_ATOMS} or more than {MAX_HEAVY_ATOMS}
      heavy atoms.
    - Search paths of at least {min_path_reactions} and at most {max_path_reactions} reactions.
    - If several {objective} chains exist, any one valid {objective} chain is acceptable.

    Guidance:
    - Use RDKit for canonicalization, ring counts, heavy atom counts, and SMARTS matching.
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
        project_name="RLMs-Task16",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_custom_ring_system(value: str) -> RingSystemSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Custom ring system must be formatted as label=SMARTS"
        )
    label, smarts = [part.strip() for part in value.split("=", 1)]
    if not label or not smarts:
        raise argparse.ArgumentTypeError(
            "Custom ring system must include both label and SMARTS"
        )
    return RingSystemSpec(
        label=label,
        smarts=(smarts,),
        aliases=(label,),
        description=f"Custom ring system defined by SMARTS {smarts}.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RLM task 16 — ring-construction chains from acyclic precursors."
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
        help=(
            "Ring-system label to search. Can be provided multiple times. "
            "Default: quinoline, indole, benzofuran, benzothiazole, and benzimidazole."
        ),
    )
    parser.add_argument(
        "--custom-ring-system",
        action="append",
        type=parse_custom_ring_system,
        default=[],
        help="Add a custom queryable ring system as label=SMARTS.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=NUM_QUESTIONS,
        help=(
            "Deprecated for the default set; all default ring systems are used "
            "when --query is not provided."
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
            "ring_system": gt.ring_system,
            "objective": gt.objective,
            "path_length": len(gt.reaction_indices),
            f"{gt.objective}_length": len(gt.reaction_indices),
            f"num_{gt.objective}_reaction_chains": len(gt.accepted_reaction_indices),
            f"{gt.objective}_reaction_chains": [
                list(rxns) for rxns in gt.accepted_reaction_indices
            ],
            "example_reaction_indices": list(gt.reaction_indices),
            "example_molecule_chain": list(gt.molecule_chain),
            "example_node_ring_systems": [list(labels) for labels in gt.node_ring_systems],
        },
        separators=(",", ":"),
    )


def build_ring_system_map(
    custom_ring_systems: list[RingSystemSpec],
) -> dict[str, RingSystemSpec]:
    ring_systems = {spec.label: spec for spec in BUILTIN_RING_SYSTEMS}
    for spec in custom_ring_systems:
        ring_systems[spec.label] = spec
    return dict(sorted(ring_systems.items()))


def main(
    model_name: str,
    dataset_path: str,
    queries: list[str],
    custom_ring_systems: list[RingSystemSpec],
    num_questions: int,
    min_path_reactions: int,
    max_path_reactions: int,
    max_longest_path_reactions: int,
    mine_only: bool,
) -> None:
    ring_systems = build_ring_system_map(custom_ring_systems)
    ring_patterns = compile_ring_patterns(ring_systems)

    if not queries:
        if num_questions != len(DEFAULT_RING_QUERIES):
            print(
                f"num_questions={num_questions} ignored; using all "
                f"{len(DEFAULT_RING_QUERIES)} default ring-system queries."
            )
        queries = list(DEFAULT_RING_QUERIES)
    unknown_queries = [query for query in queries if query not in ring_systems]
    if unknown_queries:
        raise ValueError(
            f"Unknown ring-system queries: {unknown_queries}. "
            f"Available labels: {sorted(ring_systems)}"
        )
    if min_path_reactions < 1:
        raise ValueError("--min-path-reactions must be at least 1.")
    if max_path_reactions < min_path_reactions:
        raise ValueError("--max-path-reactions must be >= --min-path-reactions.")
    if max_longest_path_reactions < min_path_reactions:
        raise ValueError("--max-longest-path-reactions must be >= --min-path-reactions.")

    records = parse_dataset(dataset_path)
    print(f"Loaded {len(records)} parsable reactions from {dataset_path}")

    _forward_graph, reverse_graph, annotations, frequent_molecules = build_molecule_graphs(
        records=records,
        ring_patterns=ring_patterns,
        max_molecule_freq=MAX_MOLECULE_FREQ,
    )
    ring_counts = {
        label: sum(1 for ann in annotations.values() if label in ann.ring_systems)
        for label in sorted(ring_systems)
    }
    print(
        f"Built molecule graph with {len(reverse_graph)} reverse nodes, "
        f"{len(annotations)} annotated molecule nodes, and {len(frequent_molecules)} frequent molecules ignored."
    )
    print(f"Annotated molecule counts by ring system: {json.dumps(ring_counts, sort_keys=True)}")
    print(f"Using {len(queries)} ring-system queries: {queries}")

    gt_paths: list[GroundTruthPath] = []
    for query in queries:
        path_specs = (
            ("shortest", max_path_reactions, shortest_ring_construction_path),
            ("longest", max_longest_path_reactions, longest_ring_construction_path),
        )
        for objective, max_reactions, path_finder in path_specs:
            gt = path_finder(
                reverse_graph=reverse_graph,
                annotations=annotations,
                ring_system=query,
                min_path_reactions=min_path_reactions,
                max_path_reactions=max_reactions,
            )
            if gt is None:
                raise ValueError(
                    f"No {objective} ground-truth path found for {query} between "
                    f"{min_path_reactions} and {max_reactions} reactions."
                )
            gt_paths.append(gt)
            print(f"Ground truth [{objective} {query}]: {summarize_gt(gt)}")

    if mine_only:
        return

    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    with open(dataset_path, encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
    context = "\n".join(f"{i} {line}" for i, line in enumerate(raw_lines))

    run = None
    if wandb is None:
        print("wandb not installed; continuing without experiment logging.")
    else:
        run = wandb.init(
            project="RLMs-Task16",
            config={
                "MODEL_NAME": model_name,
                "backend": BACKEND,
                "model_name": model_name,
                "dataset_path": dataset_path,
                "num_questions": len(gt_paths),
                "ring_system_queries": queries,
                "ring_system_specs": {
                    label: {
                        "smarts": list(spec.smarts),
                        "aliases": list(spec.aliases),
                        "description": spec.description,
                    }
                    for label, spec in ring_systems.items()
                },
                "max_molecule_freq": MAX_MOLECULE_FREQ,
                "min_heavy_atoms": MIN_HEAVY_ATOMS,
                "max_heavy_atoms": MAX_HEAVY_ATOMS,
                "min_path_reactions": min_path_reactions,
                "max_path_reactions": max_path_reactions,
                "max_longest_path_reactions": max_longest_path_reactions,
                "seed": SEED,
                "rlm_init_kwargs": rlm_init_kwargs,
                "task_description": "Ring-construction chains from acyclic precursors via RDKit SMARTS.",
            },
        )
        wandb.define_metric("sample_iteration")
        wandb.define_metric("sample/*", step_metric="sample_iteration")

    macro_accuracy = 0.0
    macro_valid_path = 0.0
    macro_objective_length = 0.0
    macro_reaction_f1 = 0.0
    index_match_count = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, gt in enumerate(gt_paths):
        spec = ring_systems[gt.ring_system]
        question = build_question(
            spec=spec,
            objective=gt.objective,
            min_path_reactions=min_path_reactions,
            max_path_reactions=(
                max_longest_path_reactions if gt.objective == "longest" else max_path_reactions
            ),
        )
        print(
            f"\nQuestion {i + 1}/{len(gt_paths)}: "
            f"{gt.objective} "
            f"ring_system={gt.ring_system}, "
            f"gt_{gt.objective}_len={len(gt.reaction_indices)}, "
            f"gt_{gt.objective}_chains={len(gt.accepted_reaction_indices)}"
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(gt_paths),
                "task": "ring_construction_chain",
                "objective": gt.objective,
                "ring_system": gt.ring_system,
                "gt_objective_length": len(gt.reaction_indices),
                "gt_num_objective_reaction_chains": len(gt.accepted_reaction_indices),
            },
            tags=["run_rlms", "sample", "task16_RING_CONSTRUCTION_CHAIN"],
        ):
            completion = rlm.completion(prompt=context, root_prompt=question)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        pred_rxns = parse_response(response)
        scores = score_prediction(
            pred_rxns=pred_rxns,
            gt=gt,
            records=records,
            ring_patterns=ring_patterns,
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

        if run is not None and wandb is not None:
            last_metric = iteration_metrics[-1] if iteration_metrics else {}
            wandb.log(
                {
                    "sample_idx": i,
                    f"sample/{i}/objective": gt.objective,
                    f"sample/{i}/ring_system": gt.ring_system,
                    f"sample/{i}/ground_truth": summarize_gt(gt),
                    f"sample/{i}/gt_objective_length": len(gt.reaction_indices),
                    f"sample/{i}/gt_num_objective_reaction_chains": len(gt.accepted_reaction_indices),
                    f"sample/{i}/pred_reaction_indices": ",".join(str(x) for x in pred_rxns),
                    f"sample/{i}/inferred_molecule_chain": scores["inferred_molecule_chain"],
                    f"sample/{i}/inferred_node_ring_systems": scores["inferred_node_ring_systems"],
                    f"sample/{i}/response_raw": response,
                    f"sample/{i}/validity_reason": scores["validity_reason"],
                    f"sample/{i}/is_correct": scores["is_correct"],
                    f"sample/{i}/valid_path": scores["valid_path"],
                    f"sample/{i}/index_match": scores["index_match"],
                    f"sample/{i}/objective_length_match": scores["objective_length_match"],
                    f"sample/{i}/reaction_precision": scores["reaction_precision"],
                    f"sample/{i}/reaction_recall": scores["reaction_recall"],
                    f"sample/{i}/reaction_f1": scores["reaction_f1"],
                    f"sample/{i}/normalized_lcs": scores["normalized_lcs"],
                    f"sample/{i}/completion_root_prompt": question,
                    f"sample/{i}/completion_prompt_char_count": len(context),
                    f"sample/{i}/final_total_input_tokens": last_metric.get("total_input_tokens", 0),
                    f"sample/{i}/final_total_output_tokens": last_metric.get("total_output_tokens", 0),
                    f"sample/{i}/final_total_tokens": last_metric.get("total_tokens", 0),
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
    print(f"Ring-system queries evaluated: {total}")
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
        custom_ring_systems=args.custom_ring_system or [],
        num_questions=args.num_questions,
        min_path_reactions=args.min_path_reactions,
        max_path_reactions=args.max_path_reactions,
        max_longest_path_reactions=args.max_longest_path_reactions,
        mine_only=args.mine_only,
    )
