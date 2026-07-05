"""Truncated synthesis-chain graph helpers for tier4 task16."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from rdkit import Chem, RDLogger

from task11_synthetic_chain_graph import canonicalize_components

RDLogger.DisableLog("rdApp.*")

DATASET_TOTAL_REACTIONS = 122_456
FULL_CHAIN_LENGTH = 5
PREFIX_LENGTH = 4
MIN_HEAVY_ATOMS = 3
MAX_HEAVY_ATOMS = 90
MAX_MOLECULE_FREQ_REFERENCE = 200
MIN_LOCAL_MOLECULE_FREQ = 3
MAX_PRECOMPUTED_VERIFY = 512


@dataclass(frozen=True)
class ReactionRecord:
    index: int
    raw: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]


@dataclass(frozen=True)
class ContextFilters:
    context_reaction_count: int
    molecule_freq_cap: int
    frequent_molecules: frozenset[str]


@dataclass(frozen=True)
class TargetQuestionSpec:
    question_id: str
    target_smiles: str
    target_name: str
    label: str
    description: str


@dataclass(frozen=True)
class GroundTruthPrefixes:
    question_id: str
    target_smiles: str
    reaction_indices: tuple[int, ...]
    accepted_reaction_indices: tuple[tuple[int, ...], ...] = tuple()
    excluded_terminal_indices: frozenset[int] = frozenset()


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


def is_spectator_exempt(smiles: str) -> bool:
    """Salts, counter-ions, and other tiny species skip hub/size filtering."""
    return heavy_atom_count(smiles) < MIN_HEAVY_ATOMS


def is_allowed_molecule_node(
    smiles: str,
    frequent_molecules: set[str] | frozenset[str],
    *,
    target_smiles: str | None = None,
) -> bool:
    if is_spectator_exempt(smiles):
        return True
    if target_smiles:
        target = Chem.CanonSmiles(target_smiles)
        if target and smiles == target:
            return True
    return smiles not in frequent_molecules and molecule_in_size_window(smiles)


def split_reaction_line(indexed_line: str) -> tuple[int, str, str]:
    idx_str, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    if len(parts) < 2:
        raise ValueError(f"Malformed reaction line: {indexed_line[:80]}")
    return int(idx_str), parts[0].strip(), parts[-1].strip()


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


def records_subset(
    full_records: dict[int, ReactionRecord],
    context_lines: list[str],
) -> dict[int, ReactionRecord]:
    """Select pre-parsed records for context lines without re-canonicalizing SMILES."""
    indices: set[int] = set()
    for line in context_lines:
        line = line.strip()
        if not line or " " not in line:
            continue
        idx_str = line.split(" ", 1)[0]
        if idx_str.isdigit():
            indices.add(int(idx_str))
    return {idx: full_records[idx] for idx in indices if idx in full_records}


def organic_components(components: tuple[str, ...] | list[str]) -> set[str]:
    return {smi for smi in components if heavy_atom_count(smi) >= MIN_HEAVY_ATOMS}


def reactions_linked_organic(
    left: ReactionRecord,
    right: ReactionRecord,
) -> bool:
    return bool(organic_components(left.products) & organic_components(right.reactants))


def is_salt_only_step(rec: ReactionRecord) -> bool:
    reactants = organic_components(rec.reactants)
    products = organic_components(rec.products)
    if not reactants or not products:
        return True
    return reactants == products


def build_predecessor_graph(
    records: dict[int, ReactionRecord],
) -> dict[int, set[int]]:
    """Reaction-DAG edges via organic product-to-reactant links only."""
    consumers: dict[str, set[int]] = defaultdict(set)

    for idx, rec in records.items():
        for smi in rec.reactants:
            consumers[smi].add(idx)

    predecessors: dict[int, set[int]] = defaultdict(set)
    for idx, rec in records.items():
        for smi in rec.products:
            if heavy_atom_count(smi) < MIN_HEAVY_ATOMS:
                continue
            for consumer_idx in consumers.get(smi, set()):
                if consumer_idx > idx:
                    predecessors[consumer_idx].add(idx)
    return dict(predecessors)


def backward_chains(
    predecessors: dict[int, set[int]],
    terminal: int,
    *,
    chain_length: int = FULL_CHAIN_LENGTH,
    cap: int = 10_000,
) -> list[tuple[int, ...]]:
    results: list[tuple[int, ...]] = []

    def dfs(chain: list[int]) -> None:
        if len(results) >= cap:
            return
        if len(chain) == chain_length:
            results.append(tuple(chain))
            return
        head = chain[0]
        for pred in sorted(predecessors.get(head, set()), reverse=True):
            if pred < head and pred not in chain:
                dfs([pred, *chain])

    dfs([terminal])
    return results


def is_clean_synthesis_chain(
    chain: tuple[int, ...],
    target_smiles: str,
    records: dict[int, ReactionRecord],
) -> bool:
    """Reject workup routes, ion-only links, and salt/identity steps."""
    target = Chem.CanonSmiles(target_smiles)
    if not target:
        return False
    for pos, reaction_idx in enumerate(chain):
        rec = records[reaction_idx]
        is_terminal = pos == len(chain) - 1
        if target in rec.reactants:
            return False
        if target in rec.products and not is_terminal:
            return False
        if is_salt_only_step(rec):
            return False
    for pos in range(len(chain) - 1):
        left = records[chain[pos]]
        right = records[chain[pos + 1]]
        if not reactions_linked_organic(left, right):
            return False
    terminal_rec = records.get(chain[-1])
    return terminal_rec is not None and target in terminal_rec.products


def mine_full_chains_for_target(
    lines: list[str],
    target_smiles: str,
    *,
    chain_length: int = FULL_CHAIN_LENGTH,
    cap: int = 10_000,
) -> tuple[list[tuple[int, ...]], set[int]]:
    records = parse_records_from_lines(lines)
    predecessors = build_predecessor_graph(records)
    target = Chem.CanonSmiles(target_smiles)
    if not target:
        raise ValueError(f"Invalid target SMILES: {target_smiles}")

    terminals = sorted(
        idx
        for idx, rec in records.items()
        if target in rec.products
    )
    chains: set[tuple[int, ...]] = set()
    for terminal in terminals:
        for chain in backward_chains(
            predecessors,
            terminal,
            chain_length=chain_length,
            cap=cap,
        ):
            if is_clean_synthesis_chain(chain, target_smiles, records):
                chains.add(chain)
            if len(chains) >= cap:
                break
        if len(chains) >= cap:
            break
    return sorted(chains), set(terminals)


def prefixes_from_full_chains(
    full_chains: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted({chain[:PREFIX_LENGTH] for chain in full_chains}))


def terminal_indices_from_full_chains(
    full_chains: tuple[tuple[int, ...], ...],
    prefixes: tuple[tuple[int, ...], ...] | None = None,
) -> frozenset[int]:
    terminals = {chain[-1] for chain in full_chains}
    if prefixes is None:
        return frozenset(terminals)
    prefix_heads = {prefix[-1] for prefix in prefixes}
    return frozenset(terminals - prefix_heads)


def reactions_linked(
    left: ReactionRecord,
    right: ReactionRecord,
) -> bool:
    return reactions_linked_organic(left, right)


def synthesis_spine_molecules(chain_records: list[ReactionRecord]) -> set[str]:
    """Organic intermediates on the product→reactant spine, excluding co-reagents."""
    if not chain_records:
        return set()
    spine: set[str] = set()
    for pos in range(len(chain_records) - 1):
        left = chain_records[pos]
        right = chain_records[pos + 1]
        spine |= organic_components(left.products) & organic_components(right.reactants)
    spine |= organic_components(chain_records[-1].products)
    return spine


def verify_predicted_prefix(
    reaction_indices: tuple[int, ...],
    target_smiles: str,
    records: dict[int, ReactionRecord],
    *,
    full_chains: tuple[tuple[int, ...], ...] | None = None,
    excluded_terminal_indices: set[int] | frozenset[int] | None = None,
    filters: ContextFilters | None = None,
) -> tuple[bool, str]:
    if filters is None:
        filters = context_filters_from_records(records)

    if not reaction_indices:
        return False, "no reaction indices returned"
    if len(reaction_indices) != PREFIX_LENGTH:
        return False, f"chain must have exactly {PREFIX_LENGTH} reactions"
    if len(set(reaction_indices)) != len(reaction_indices):
        return False, "reaction indices contain repeats"
    if list(reaction_indices) != sorted(reaction_indices):
        return False, "reaction indices must be strictly ascending"

    chain_records: list[ReactionRecord] = []
    for reaction_idx in reaction_indices:
        rec = records.get(reaction_idx)
        if rec is None:
            return False, f"reaction index {reaction_idx} not found in context"
        chain_records.append(rec)

    for pos in range(PREFIX_LENGTH - 1):
        if not reactions_linked(chain_records[pos], chain_records[pos + 1]):
            return (
                False,
                f"no product-to-reactant link between reactions "
                f"{chain_records[pos].index} and {chain_records[pos + 1].index}",
            )

    if full_chains is not None:
        matching = [
            chain
            for chain in full_chains
            if chain[:PREFIX_LENGTH] == reaction_indices
        ]
        if not matching:
            return False, "prefix is not a valid truncation of any full-dataset route"
        if excluded_terminal_indices is not None:
            withheld = [
                chain
                for chain in matching
                if chain[-1] in excluded_terminal_indices
            ]
            if withheld and not any(chain[-1] not in records for chain in withheld):
                return False, "terminal reaction for this prefix is present in context"
        return True, "ok"

    target = Chem.CanonSmiles(target_smiles)
    if not target:
        return False, "invalid target SMILES"
    last_products = set(chain_records[-1].products)
    if target not in last_products:
        return (
            False,
            "final prefix reaction does not produce an intermediate linked to target",
        )
    return True, "ok"


def filter_context_lines(
    lines: list[str],
    exclude_indices: set[int] | frozenset[int],
) -> list[str]:
    if not exclude_indices:
        return lines
    filtered: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or " " not in line:
            continue
        idx_str = line.split(" ", 1)[0]
        if idx_str.isdigit() and int(idx_str) in exclude_indices:
            continue
        filtered.append(line)
    return filtered


def ground_truth_prefixes_in_context(
    context_lines: list[str],
    question_id: str,
    target_smiles: str,
    *,
    full_chains: tuple[tuple[int, ...], ...],
    excluded_terminal_indices: set[int] | frozenset[int],
    records: dict[int, ReactionRecord] | None = None,
    limit_to_prefixes: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[GroundTruthPrefixes | None, ContextFilters]:
    from task16_truncated_synthesis_ground_truth import hardcoded_prefixes_for_question

    if records is None:
        records = parse_records_from_lines(context_lines)
    filters = context_filters_from_records(records)
    context_indices = set(records.keys())

    prefix_pool = (
        limit_to_prefixes
        if limit_to_prefixes is not None
        else hardcoded_prefixes_for_question(question_id)
    )
    candidate_prefixes = [
        prefix
        for prefix in prefix_pool
        if set(prefix).issubset(context_indices)
    ]

    accepted: list[tuple[int, ...]] = []
    for prefix in candidate_prefixes:
        ok, _reason = verify_predicted_prefix(
            prefix,
            target_smiles,
            records,
            full_chains=full_chains,
            excluded_terminal_indices=excluded_terminal_indices,
            filters=filters,
        )
        if ok:
            accepted.append(prefix)

    if not accepted:
        return None, filters

    accepted = sorted(set(accepted))
    return (
        GroundTruthPrefixes(
            question_id=question_id,
            target_smiles=target_smiles,
            reaction_indices=accepted[0],
            accepted_reaction_indices=tuple(accepted),
            excluded_terminal_indices=frozenset(excluded_terminal_indices),
        ),
        filters,
    )


def parse_chains(
    response: str,
    *,
    path_length: int = PREFIX_LENGTH,
) -> list[tuple[int, ...]]:
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


def precision_recall_f1(
    predicted: set[tuple[int, ...]],
    ground_truth: set[tuple[int, ...]],
) -> tuple[float, float, float]:
    tp = len(predicted & ground_truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_chain_predictions(
    pred_chains: list[tuple[int, ...]],
    gt: GroundTruthPrefixes,
    records: dict[int, ReactionRecord],
    *,
    full_chains: tuple[tuple[int, ...], ...],
    filters: ContextFilters | None = None,
) -> dict[str, float | str | int]:
    if filters is None:
        filters = context_filters_from_records(records)

    gt_set = set(gt.accepted_reaction_indices or (gt.reaction_indices,))
    valid_pred: list[tuple[int, ...]] = []
    invalid_reasons: list[str] = []
    for chain in pred_chains:
        ok, reason = verify_predicted_prefix(
            chain,
            gt.target_smiles,
            records,
            full_chains=full_chains,
            excluded_terminal_indices=gt.excluded_terminal_indices,
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


def build_question(spec: TargetQuestionSpec) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format in the provided context, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Each side may contain multiple species separated by dots (.).
    Ignore reagents (the middle field between the two > delimiters).

    Task:
    The target molecule for synthesis is:
    {spec.target_name}

    ({spec.label}: {spec.description})

    The final reaction step that forms this target has been withheld from the context.
    Find ALL valid 4-reaction prefixes [r_0, r_1, r_2, r_3] present in the context that
    could complete to this target via one additional reaction not shown in the context.

    A valid prefix is an ordered sequence of exactly {PREFIX_LENGTH} distinct reaction indices
    [r_0, r_1, r_2, r_3] such that:
    - r_0 < r_1 < r_2 < r_3 (strictly ascending reaction indices).
    - For each k in {{0, 1, 2}}, at least one canonical-SMILES product component of reaction
      r_k is identical to at least one canonical-SMILES reactant component of r_{{k+1}}.
    - Use exact canonical SMILES equality on dot-separated components for all identity checks.
    - Do not use substructure matching for identity.
    - Do not use the same reaction index twice in one prefix.
    - Only use reactions present in the provided context.
    - Each prefix must be completable to the target product named above by appending exactly
      one withheld final reaction (identify the target structure from the reaction context).

    Guidance:
    - Use RDKit for canonicalization.
    - Split multi-component sides on dots (.) and canonicalize each component independently.
    - Skip malformed reactions or species that RDKit cannot parse.

    Output format:
    - Return each prefix as a comma-separated list of exactly {PREFIX_LENGTH} reaction indices,
      one prefix per line.
    - Sort prefixes in lexicographic (ascending) order.
    - No other text, quotes, labels, punctuation, JSON, or formatting.
    - If no valid prefix exists, return -1.
    """


RLM_CODE_GUIDANCE = """
- DO NOT assume/simulate output of code. Wait for code execution and only then return.
- DO NOT USE `FINAL` for writing a thought/comment. Only use `FINAL` for the final answer.
- Keep code fast: set a time limit (e.g. check elapsed time in loops) and avoid brute-force
  scans over the full context when it is large; prefer targeted parsing and early exits.
""".strip()

RLM_DOCKER_MEMORY_GUIDANCE = """
- Your Python code runs in a Docker sandbox with a {memory_limit} RAM cap (swap disabled).
  Exceeding this limit OOM-kills the process. Prefer line-by-line or chunked processing over
  loading the entire context into memory at once.
""".strip()


def build_rlm_question(
    spec: TargetQuestionSpec,
    *,
    docker_memory_limit: str | None = None,
) -> str:
    guidance = RLM_CODE_GUIDANCE
    if docker_memory_limit:
        guidance = (
            f"{guidance}\n"
            f"{RLM_DOCKER_MEMORY_GUIDANCE.format(memory_limit=docker_memory_limit)}"
        )
    return (
        f"{build_question(spec)}\n\n"
        f"Guidance:\n{guidance}"
    )
