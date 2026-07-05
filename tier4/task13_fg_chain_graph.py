"""Functional-group transformation chain graph helpers for tier4 task13."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PATH_LENGTH = 7
DATASET_TOTAL_REACTIONS = 122_456
MAX_MOLECULE_FREQ_REFERENCE = 200
MIN_LOCAL_MOLECULE_FREQ = 3
MIN_HEAVY_ATOMS = 3
MAX_HEAVY_ATOMS = 90
MAX_PRECOMPUTED_VERIFY = 256

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
class ContextFilters:
    """Context-local graph filters computable from the provided reaction lines only."""

    context_reaction_count: int
    molecule_freq_cap: int
    frequent_molecules: frozenset[str]


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


def molecule_in_size_window(
    smiles: str,
    *,
    min_heavy_atoms: int = MIN_HEAVY_ATOMS,
    max_heavy_atoms: int = MAX_HEAVY_ATOMS,
) -> bool:
    heavy = heavy_atom_count(smiles)
    return min_heavy_atoms <= heavy <= max_heavy_atoms


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


def is_allowed_molecule_node(
    smiles: str,
    frequent_molecules: set[str] | frozenset[str],
) -> bool:
    return smiles not in frequent_molecules and molecule_in_size_window(smiles)


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


def detect_functional_groups(smiles: str) -> tuple[str, ...]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tuple()
    found: list[str] = []
    for label, patterns in FG_PATTERNS.items():
        if any(mol.HasSubstructMatch(pattern) for pattern in patterns):
            found.append(label)
    return tuple(sorted(found))


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


def build_molecule_graph(
    records: dict[int, ReactionRecord],
    filters: ContextFilters | None = None,
) -> tuple[dict[str, list[MoleculeStep]], dict[str, tuple[str, ...]], ContextFilters]:
    if filters is None:
        filters = context_filters_from_records(records)

    frequent_molecules = filters.frequent_molecules
    all_molecules: set[str] = set()
    for rec in records.values():
        all_molecules.update(rec.reactants)
        all_molecules.update(rec.products)

    group_cache = {
        smiles: detect_functional_groups(smiles)
        for smiles in all_molecules
        if is_allowed_molecule_node(smiles, frequent_molecules)
    }

    graph: dict[str, list[MoleculeStep]] = defaultdict(list)
    for rec in records.values():
        reactants = [smi for smi in rec.reactants if smi in group_cache]
        products = [smi for smi in rec.products if smi in group_cache]
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

    return dict(graph), group_cache, filters


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


def longest_fg_path(
    graph: dict[str, list[MoleculeStep]],
    group_cache: dict[str, tuple[str, ...]],
    source_fg: str,
    target_fg: str,
    path_length: int = PATH_LENGTH,
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

    for _depth in range(path_length):
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
                    len(next_reactions) == path_length
                    and target_fg in next_groups
                    and source_fg not in next_groups
                ):
                    solutions.append((next_reactions, next_molecules))
                if len(next_reactions) < path_length:
                    next_paths.append((nxt, next_reactions, next_molecules))

        paths = next_paths
        if not paths:
            break

    if not solutions:
        return None

    return build_ground_truth_from_solutions(
        solutions=solutions,
        source_fg=source_fg,
        target_fg=target_fg,
        group_cache=group_cache,
        objective="longest",
    )


def ground_truth_fg_path_in_context(
    context_lines: list[str],
    source_fg: str,
    target_fg: str,
    *,
    path_length: int = PATH_LENGTH,
) -> tuple[GroundTruthPath | None, ContextFilters]:
    records = parse_records_from_lines(context_lines)
    graph, group_cache, filters = build_molecule_graph(records)
    context_indices = set(records.keys())

    from task13_fg_chain_ground_truth import hardcoded_chains_for_pair

    candidate_chains = [
        tuple(chain)
        for chain in hardcoded_chains_for_pair(source_fg, target_fg)
        if len(chain) == path_length and set(chain).issubset(context_indices)
    ]

    if len(candidate_chains) > MAX_PRECOMPUTED_VERIFY:
        gt = longest_fg_path(
            graph=graph,
            group_cache=group_cache,
            source_fg=source_fg,
            target_fg=target_fg,
            path_length=path_length,
        )
        return gt, filters

    solutions: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    for reaction_chain in candidate_chains:
        ok, _msg, molecule_chain, _node_groups = verify_predicted_path(
            reaction_chain,
            source_fg,
            target_fg,
            records,
            path_length=path_length,
            filters=filters,
        )
        if ok:
            solutions.append((reaction_chain, molecule_chain))

    if solutions:
        gt = build_ground_truth_from_solutions(
            solutions=solutions,
            source_fg=source_fg,
            target_fg=target_fg,
            group_cache=group_cache,
            objective="longest",
        )
        return gt, filters

    gt = longest_fg_path(
        graph=graph,
        group_cache=group_cache,
        source_fg=source_fg,
        target_fg=target_fg,
        path_length=path_length,
    )
    return gt, filters


def parse_chains(response: str, path_length: int = PATH_LENGTH) -> list[tuple[int, ...]]:
    text = response.strip()
    if not text or text.replace(" ", "") == "-1":
        return []

    chains: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "-1":
            continue
        nums = re.findall(r"\d+", line)
        if len(nums) < path_length:
            continue
        chain = tuple(int(n) for n in nums[:path_length])
        if chain not in seen:
            seen.add(chain)
            chains.append(chain)
    return chains


def parse_response(response: str) -> tuple[int, ...]:
    """Parse a single chain from legacy one-line responses."""
    chains = parse_chains(response)
    return chains[0] if chains else tuple()


def verify_predicted_path(
    reaction_indices: tuple[int, ...],
    source_fg: str,
    target_fg: str,
    records: dict[int, ReactionRecord],
    *,
    path_length: int = PATH_LENGTH,
    filters: ContextFilters | None = None,
) -> tuple[bool, str, tuple[str, ...], tuple[tuple[str, ...], ...]]:
    if filters is None:
        filters = context_filters_from_records(records)
    frequent_molecules = filters.frequent_molecules

    if not reaction_indices:
        return False, "no reaction indices returned", tuple(), tuple()
    if len(reaction_indices) != path_length:
        return False, f"chain must have exactly {path_length} reactions", tuple(), tuple()
    if len(set(reaction_indices)) != len(reaction_indices):
        return False, "reaction indices contain repeats", tuple(), tuple()

    chain_records: list[ReactionRecord] = []
    for reaction_idx in reaction_indices:
        rec = records.get(reaction_idx)
        if rec is None:
            return False, f"reaction index {reaction_idx} not found", tuple(), tuple()
        chain_records.append(rec)

    states: list[tuple[str, tuple[str, ...]]] = []
    for reactant in chain_records[0].reactants:
        if not is_allowed_molecule_node(reactant, frequent_molecules):
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
                if not is_allowed_molecule_node(product, frequent_molecules):
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
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_chain_predictions(
    pred_chains: list[tuple[int, ...]],
    gt: GroundTruthPath,
    records: dict[int, ReactionRecord],
    *,
    path_length: int = PATH_LENGTH,
    filters: ContextFilters | None = None,
) -> dict[str, float | str | int]:
    if filters is None:
        filters = context_filters_from_records(records)

    gt_set = set(gt.accepted_reaction_indices or (gt.reaction_indices,))
    valid_pred: list[tuple[int, ...]] = []
    invalid_reasons: list[str] = []
    for chain in pred_chains:
        ok, reason, _inferred_mols, _inferred_groups = verify_predicted_path(
            reaction_indices=chain,
            source_fg=gt.source_fg,
            target_fg=gt.target_fg,
            records=records,
            path_length=path_length,
            filters=filters,
        )
        if ok:
            valid_pred.append(chain)
        else:
            invalid_reasons.append(f"{chain}: {reason}")

    valid_set = set(valid_pred)
    precision, recall, f1 = precision_recall_f1(valid_set, gt_set)
    is_exact_match = valid_set == gt_set
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "is_exact_match": float(is_exact_match),
        "parsed_chain_count": len(pred_chains),
        "valid_chain_count": len(valid_set),
        "ground_truth_chain_count": len(gt_set),
        "invalid_chain_count": len(pred_chains) - len(valid_set),
        "validity_reason": "; ".join(invalid_reasons[:3]) if invalid_reasons else "ok",
    }


def score_prediction(
    pred_rxns: tuple[int, ...],
    gt: GroundTruthPath,
    records: dict[int, ReactionRecord],
    *,
    path_length: int = PATH_LENGTH,
    filters: ContextFilters | None = None,
) -> dict[str, float | str]:
    if filters is None:
        filters = context_filters_from_records(records)
    valid, reason, inferred_mols, inferred_groups = verify_predicted_path(
        reaction_indices=pred_rxns,
        source_fg=gt.source_fg,
        target_fg=gt.target_fg,
        records=records,
        path_length=path_length,
        filters=filters,
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
    *,
    context_reaction_count: int,
    path_length: int = PATH_LENGTH,
    molecule_freq_cap: int | None = None,
) -> str:
    freq_cap = (
        molecule_freq_cap
        if molecule_freq_cap is not None
        else scaled_molecule_freq_cap(context_reaction_count)
    )
    return f"""
    There is a list of chemical reactions in SMILES format in the provided context, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Each side may contain multiple species separated by dots (.).
    Ignore reagents (the middle field between the two > delimiters).

    Task:
    Find ALL valid reaction chains of exactly {path_length} reactions in the provided context that
    convert a species containing functional group "{source_fg}" into a species containing
    functional group "{target_fg}".

    A valid chain is an ordered sequence of {path_length} distinct reaction indices
    [r_0, r_1, ..., r_{path_length - 1}] such that:
    - For each k from 0 to {path_length - 2}, at least one canonical-SMILES product component of
      reaction r_k is identical to at least one canonical-SMILES reactant component of r_{{k+1}}.
    - At least one reactant component of r_0 contains "{source_fg}" and does not contain "{target_fg}".
    - At least one product component of r_{path_length - 1} contains "{target_fg}" and does not
      contain "{source_fg}".
    - Use exact canonical SMILES equality on dot-separated components for all identity checks.
    - Do not use substructure matching for identity.
    - Do not use the same reaction index twice in one chain.
    - Only use reactions present in the provided context.
    - Exclude any canonical-SMILES species that appears as a reactant or product in more than
      {freq_cap} reactions within the provided context.
    - Exclude species with fewer than {MIN_HEAVY_ATOMS} or more than {MAX_HEAVY_ATOMS} heavy atoms.

    Guidance:
    - Use RDKit for canonicalization, heavy-atom counts, and functional-group detection.
    - Count species appearances across all reactions in the provided context to apply the
      frequency filter.
    - Split multi-component sides on dots (.) and canonicalize each component independently.
    - Skip malformed reactions or species that RDKit cannot parse.

    Output format:
    - Return each chain as a comma-separated list of exactly {path_length} reaction indices,
      one chain per line.
    - Sort chains in lexicographic (ascending) order.
    - No other text, quotes, labels, punctuation, JSON, or formatting.
    - If no valid chain exists, return -1.
    """


RLM_CODE_GUIDANCE = """
    - DO NOT assume/simulate output of code. Wait for code execution and only then return.
    - DO NOT USE `FINAL` for writing a thought/comment. Only use `FINAL` for the final answer.
""".strip()


def build_rlm_question(
    source_fg: str,
    target_fg: str,
    *,
    context_reaction_count: int,
    path_length: int = PATH_LENGTH,
    molecule_freq_cap: int | None = None,
) -> str:
    return (
        f"{build_question(source_fg, target_fg, context_reaction_count=context_reaction_count, path_length=path_length, molecule_freq_cap=molecule_freq_cap)}\n\n"
        f"Guidance:\n{RLM_CODE_GUIDANCE}"
    )
