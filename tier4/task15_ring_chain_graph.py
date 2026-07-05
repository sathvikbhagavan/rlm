"""Ring-construction chain graph helpers for tier4 task15."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATASET_TOTAL_REACTIONS = 122_456
PATH_LENGTH = 3
MAX_MOLECULE_FREQ_REFERENCE = 200
MIN_LOCAL_MOLECULE_FREQ = 3
MIN_PATH_REACTIONS = PATH_LENGTH
MAX_PATH_REACTIONS = PATH_LENGTH
MAX_LONGEST_PATH_REACTIONS = 7
MIN_HEAVY_ATOMS = 3
MAX_HEAVY_ATOMS = 90
MAX_ACCEPTED_CHAINS = 200
MAX_PRECOMPUTED_VERIFY = 256
FULL_CONTEXT_CHAIN_THRESHOLD = int(DATASET_TOTAL_REACTIONS * 0.99)


@dataclass(frozen=True)
class RingSystemSpec:
    label: str
    smarts: tuple[str, ...]
    aliases: tuple[str, ...]
    description: str


BUILTIN_RING_SYSTEMS: tuple[RingSystemSpec, ...] = (
    RingSystemSpec(
        label="quinoline",
        smarts=("c1ccc2ncccc2c1", "c1ccc2cnccc2c1"),
        aliases=("quinoline", "isoquinoline", "benzopyridine"),
        description="Fused benzene-pyridine ring systems, including quinoline and isoquinoline orientation.",
    ),
    RingSystemSpec(
        label="indole",
        smarts=("c1ccc2[nH]ccc2c1", "c1ccc2[nH0]ccc2c1"),
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
        smarts=("c1ccc2[nH]cnc2c1", "c1ccc2[nH0]cnc2c1"),
        aliases=("benzimidazole",),
        description="Fused benzene-imidazole ring system.",
    ),
)

RING_SYSTEM_BY_LABEL = {spec.label: spec for spec in BUILTIN_RING_SYSTEMS}


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
class MoleculeAnnotation:
    acyclic: bool
    ring_systems: tuple[str, ...]


@dataclass(frozen=True)
class ContextFilters:
    context_reaction_count: int
    molecule_freq_cap: int
    frequent_molecules: frozenset[str]


@dataclass(frozen=True)
class GroundTruthPath:
    ring_system: str
    objective: str
    reaction_indices: tuple[int, ...]
    molecule_chain: tuple[str, ...]
    node_ring_systems: tuple[tuple[str, ...], ...]
    accepted_reaction_indices: tuple[tuple[int, ...], ...] = tuple()
    accepted_molecule_chains: tuple[tuple[str, ...], ...] = tuple()


def compile_ring_patterns(
    ring_systems: dict[str, RingSystemSpec] | None = None,
) -> dict[str, list[Chem.Mol]]:
    specs = ring_systems or RING_SYSTEM_BY_LABEL
    patterns: dict[str, list[Chem.Mol]] = {}
    for label, spec in specs.items():
        compiled = []
        for smarts in spec.smarts:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                raise ValueError(f"Invalid SMARTS for {label}: {smarts}")
            compiled.append(patt)
        patterns[label] = compiled
    return patterns


RING_PATTERNS = compile_ring_patterns()


def scaled_molecule_freq_cap(
    context_reaction_count: int,
    *,
    dataset_size: int = DATASET_TOTAL_REACTIONS,
    reference_cap: int = MAX_MOLECULE_FREQ_REFERENCE,
    min_local_cap: int = MIN_LOCAL_MOLECULE_FREQ,
) -> int:
    if context_reaction_count <= 0:
        return min_local_cap
    scaled = round(reference_cap * context_reaction_count / dataset_size)
    return max(min_local_cap, scaled)


def heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def molecule_in_size_window(smiles: str) -> bool:
    heavy = heavy_atom_count(smiles)
    return MIN_HEAVY_ATOMS <= heavy <= MAX_HEAVY_ATOMS


def frequent_molecules_in_context(
    records: dict[int, ReactionRecord],
    *,
    freq_cap: int | None = None,
) -> set[str]:
    cap = freq_cap if freq_cap is not None else scaled_molecule_freq_cap(len(records))
    counts: dict[str, int] = defaultdict(int)
    for rec in records.values():
        for smiles in set(rec.reactants) | set(rec.products):
            counts[smiles] += 1
    return {smiles for smiles, count in counts.items() if count > cap}


def context_filters_from_records(records: dict[int, ReactionRecord]) -> ContextFilters:
    reaction_count = len(records)
    cap = scaled_molecule_freq_cap(reaction_count)
    frequent = frequent_molecules_in_context(records, freq_cap=cap)
    return ContextFilters(
        context_reaction_count=reaction_count,
        molecule_freq_cap=cap,
        frequent_molecules=frozenset(frequent),
    )


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


def annotate_molecule(
    smiles: str,
    ring_patterns: dict[str, list[Chem.Mol]] | None = None,
) -> MoleculeAnnotation | None:
    patterns = ring_patterns or RING_PATTERNS
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ring_info = mol.GetRingInfo()
    found = [
        label
        for label, pats in patterns.items()
        if any(mol.HasSubstructMatch(pattern) for pattern in pats)
    ]
    return MoleculeAnnotation(
        acyclic=ring_info.NumRings() == 0,
        ring_systems=tuple(sorted(found)),
    )


def parse_records_from_lines(lines: list[str]) -> dict[int, ReactionRecord]:
    records: dict[int, ReactionRecord] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            idx, reactants_raw, products_raw = split_reaction_line(line)
        except Exception:
            continue
        reactants = tuple(canonicalize_components(reactants_raw))
        products = tuple(canonicalize_components(products_raw))
        if not reactants or not products:
            continue
        records[idx] = ReactionRecord(
            index=idx,
            raw=line.split(" ", 1)[1] if " " in line else line,
            reactants=reactants,
            products=products,
        )
    return records


def build_molecule_graphs(
    records: dict[int, ReactionRecord],
    filters: ContextFilters | None = None,
    ring_patterns: dict[str, list[Chem.Mol]] | None = None,
) -> tuple[
    dict[str, list[MoleculeStep]],
    dict[str, list[MoleculeStep]],
    dict[str, MoleculeAnnotation],
    ContextFilters,
]:
    if filters is None:
        filters = context_filters_from_records(records)
    patterns = ring_patterns or RING_PATTERNS
    frequent_molecules = filters.frequent_molecules

    all_molecules: set[str] = set()
    for rec in records.values():
        all_molecules.update(rec.reactants)
        all_molecules.update(rec.products)

    annotations: dict[str, MoleculeAnnotation] = {}
    for smiles in all_molecules:
        if smiles in frequent_molecules or not molecule_in_size_window(smiles):
            continue
        annotation = annotate_molecule(smiles, patterns)
        if annotation is not None:
            annotations[smiles] = annotation

    forward_graph: dict[str, list[MoleculeStep]] = defaultdict(list)
    reverse_graph: dict[str, list[MoleculeStep]] = defaultdict(list)
    for rec in records.values():
        reactants = [smi for smi in rec.reactants if smi in annotations]
        products = [smi for smi in rec.products if smi in annotations]
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

    return dict(forward_graph), dict(reverse_graph), annotations, filters


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

                if len(next_reactions) >= min_path_reactions and prev_annotation.acyclic:
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

                if len(next_reactions) >= min_path_reactions and prev_annotation.acyclic:
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


def ring_path_finder(
    objective: str,
):
    if objective == "shortest":
        return shortest_ring_construction_path
    if objective == "longest":
        return longest_ring_construction_path
    raise ValueError(f"Unknown objective: {objective}")


def max_reactions_for_objective(
    objective: str,
    *,
    max_path_reactions: int = MAX_PATH_REACTIONS,
    max_longest_path_reactions: int = MAX_LONGEST_PATH_REACTIONS,
) -> int:
    return max_longest_path_reactions if objective == "longest" else max_path_reactions


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
    filters: ContextFilters,
    min_path_reactions: int,
    ring_patterns: dict[str, list[Chem.Mol]] | None = None,
) -> tuple[bool, str, tuple[str, ...], tuple[tuple[str, ...], ...]]:
    patterns = ring_patterns or RING_PATTERNS
    frequent_molecules = filters.frequent_molecules

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
            annotation_cache[smiles] = annotate_molecule(smiles, patterns)
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
    filters: ContextFilters,
    min_path_reactions: int,
) -> dict[str, float | str]:
    valid, reason, inferred_mols, inferred_ring_systems = verify_predicted_path(
        reaction_indices=pred_rxns,
        ring_system=gt.ring_system,
        records=records,
        filters=filters,
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


def ground_truth_ring_path_in_context(
    context_lines: list[str],
    ring_system: str,
    *,
    path_length: int = PATH_LENGTH,
) -> tuple[GroundTruthPath | None, ContextFilters]:
    records = parse_records_from_lines(context_lines)
    _forward, reverse_graph, annotations, filters = build_molecule_graphs(records)
    context_indices = set(records.keys())

    from task15_ring_chain_ground_truth import hardcoded_chains_for_question

    candidate_chains = [
        tuple(chain)
        for chain in hardcoded_chains_for_question(ring_system)
        if len(chain) == path_length and set(chain).issubset(context_indices)
    ]

    if len(candidate_chains) > MAX_PRECOMPUTED_VERIFY:
        gt = shortest_ring_construction_path(
            reverse_graph=reverse_graph,
            annotations=annotations,
            ring_system=ring_system,
            min_path_reactions=path_length,
            max_path_reactions=path_length,
        )
        return gt, filters

    solutions: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    for reaction_chain in candidate_chains:
        ok, _msg, molecule_chain, _node_ring_systems = verify_predicted_path(
            reaction_chain,
            ring_system,
            records,
            filters,
            min_path_reactions=path_length,
        )
        if ok:
            solutions.append((reaction_chain, molecule_chain))

    if solutions:
        gt = build_ground_truth_from_solutions(
            solutions=solutions[:MAX_ACCEPTED_CHAINS],
            ring_system=ring_system,
            annotations=annotations,
            objective="shortest",
        )
        return gt, filters

    gt = shortest_ring_construction_path(
        reverse_graph=reverse_graph,
        annotations=annotations,
        ring_system=ring_system,
        min_path_reactions=path_length,
        max_path_reactions=path_length,
    )
    return gt, filters


def summarize_gt(gt: GroundTruthPath) -> dict[str, object]:
    return {
        "ring_system": gt.ring_system,
        "objective": gt.objective,
        "path_length": len(gt.reaction_indices),
        f"{gt.objective}_length": len(gt.reaction_indices),
        f"num_{gt.objective}_reaction_chains": len(gt.accepted_reaction_indices),
        f"{gt.objective}_reaction_chains": [list(rxns) for rxns in gt.accepted_reaction_indices],
        "example_reaction_indices": list(gt.reaction_indices),
        "example_molecule_chain": list(gt.molecule_chain),
        "example_node_ring_systems": [list(labels) for labels in gt.node_ring_systems],
    }


def build_question(
    spec: RingSystemSpec,
    *,
    context_reaction_count: int,
    molecule_freq_cap: int,
    path_length: int = PATH_LENGTH,
) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format in the provided context, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Each side may contain multiple molecules separated by dots (.).

    Task:
    Find a reaction chain of exactly {path_length} reactions, using reactions from the provided context,
    that builds the ring system "{spec.label}" from a non-cyclic precursor.

    Ring system:
    - label: {spec.label}
    - aliases: {", ".join(spec.aliases)}
    - description: {spec.description}
    - SMARTS hints: {", ".join(spec.smarts)}

    A valid reaction chain [r_0, r_1, ..., r_(k-1)] must admit at least one molecule component
    chain [m_0, m_1, ..., m_k] such that m_i is an exact canonical-SMILES reactant
    component of r_i and m_(i+1) is an exact canonical-SMILES product component of r_i.
    For adjacent reactions, the product component m_(i+1) must also be an exact
    canonical-SMILES reactant component of r_(i+1).

    Constraints:
    - The chain must contain exactly {path_length} reactions.
    - m_0 must be non-cyclic: RDKit ring count must be zero.
    - m_0 must NOT already contain the "{spec.label}" ring system.
    - Intermediate tracked molecule components m_1 through m_(k-1) must NOT contain
      the "{spec.label}" ring system; the ring system should first appear in m_k.
    - m_k must contain the "{spec.label}" ring system.
    - Do not repeat reaction indices or molecule nodes within the chain.
    - Use exact canonical SMILES equality for reaction edges; do not use substructure
      matching for molecule identity.
    - Use SMARTS/substructure matching only to recognize the requested ring system.
    - Ignore molecules that appear in more than {molecule_freq_cap} reactions as
      reactants or products in this context of {context_reaction_count} reactions.
    - Ignore molecule nodes with fewer than {MIN_HEAVY_ATOMS} or more than {MAX_HEAVY_ATOMS}
      heavy atoms.
    - If several valid chains exist in the context, any one valid chain is acceptable.

    Output format:
    - Return ONLY the reaction indices in the chain.
    - Format must be a comma-separated list of integers (e.g., 60483,60620,60621).
    - No other text, quotes, labels, punctuation, JSON, or formatting.
    - If no chain exists in the context, return -1.
    """


RLM_CODE_GUIDANCE = """
    - Use RDKit for canonicalization, ring counts, heavy atom counts, and SMARTS matching.
    - Split multi-component sides on dots (.) and canonicalize each component independently.
    - Skip malformed reactions or molecules that RDKit cannot parse.
    - DO NOT assume/simulate output of code. Wait for code execution and only then return.
    - DO NOT USE `FINAL` for writing a thought/comment. Only use `FINAL` for the final answer.
""".strip()


def build_rlm_question(
    spec: RingSystemSpec,
    *,
    context_reaction_count: int,
    molecule_freq_cap: int,
    path_length: int = PATH_LENGTH,
) -> str:
    return (
        f"{build_question(spec, context_reaction_count=context_reaction_count, molecule_freq_cap=molecule_freq_cap, path_length=path_length)}\n\n"
        f"Guidance:\n{RLM_CODE_GUIDANCE}"
    )
