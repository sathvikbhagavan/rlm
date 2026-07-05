"""Protecting-group install/remove pair helpers for tier4 task14."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATASET_TOTAL_REACTIONS = 122_456
MIN_HEAVY_ATOMS = 3
MAX_HEAVY_ATOMS = 90
FULL_CONTEXT_PAIR_THRESHOLD = int(DATASET_TOTAL_REACTIONS * 0.99)


@dataclass(frozen=True)
class ProtectingGroupSpec:
    label: str
    functional_group: str
    protected_smarts: tuple[str, ...]
    aliases: tuple[str, ...]
    description: str


PROTECTING_GROUPS: tuple[ProtectingGroupSpec, ...] = (
    ProtectingGroupSpec(
        label="Boc_N",
        functional_group="amine",
        protected_smarts=("[NX3][CX3](=O)[OX2][C;X4]([CH3])([CH3])[CH3]",),
        aliases=("Boc", "tert-butyloxycarbonyl", "BOC"),
        description="Boc-protected amines: N-C(=O)-O-tert-butyl carbamates.",
    ),
    ProtectingGroupSpec(
        label="benzyl_O_N",
        functional_group="alcohol_or_amine",
        protected_smarts=("[O,N]Cc1ccccc1",),
        aliases=("Bn", "benzyl"),
        description="Benzyl-protected alcohols or amines: heteroatom-CH2-phenyl.",
    ),
)


@dataclass(frozen=True)
class ReactionRecord:
    index: int
    raw: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]


@dataclass(frozen=True)
class ProtectionEvent:
    reaction_index: int
    pg_label: str
    direction: str
    free_smiles: str
    protected_smiles: str
    scaffold_key: str


@dataclass(frozen=True)
class GroundTruthPair:
    install_index: int
    remove_index: int
    pg_label: str
    functional_group: str
    scaffold_key: str
    install_free_smiles: str
    install_protected_smiles: str
    remove_protected_smiles: str
    remove_free_smiles: str


def compile_pg_patterns() -> dict[str, list[Chem.Mol]]:
    patterns: dict[str, list[Chem.Mol]] = {}
    for spec in PROTECTING_GROUPS:
        compiled = []
        for smarts in spec.protected_smarts:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                raise ValueError(f"Invalid SMARTS for {spec.label}: {smarts}")
            compiled.append(patt)
        patterns[spec.label] = compiled
    return patterns


PG_PATTERNS = compile_pg_patterns()


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


def has_pg(smiles: str, pg_label: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(mol.HasSubstructMatch(pattern) for pattern in PG_PATTERNS[pg_label])


def stripped_scaffold_keys(smiles: str, pg_label: str) -> tuple[str, ...]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tuple()

    keys: set[str] = set()
    for pattern in PG_PATTERNS[pg_label]:
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            atoms_to_remove = sorted(match[1:], reverse=True)
            editable = Chem.RWMol(mol)
            for atom_idx in atoms_to_remove:
                editable.RemoveAtom(atom_idx)
            try:
                stripped = editable.GetMol()
                Chem.SanitizeMol(stripped)
                keys.add(Chem.MolToSmiles(stripped, canonical=True))
            except Exception:
                continue
    return tuple(sorted(keys))


def free_scaffold_key(smiles: str) -> str | None:
    try:
        return Chem.CanonSmiles(smiles)
    except Exception:
        return None


def mine_protection_events(records: dict[int, ReactionRecord]) -> list[ProtectionEvent]:
    events: list[ProtectionEvent] = []
    heavy_cache: dict[str, bool] = {}
    has_pg_cache: dict[tuple[str, str], bool] = {}
    free_key_cache: dict[str, str | None] = {}
    stripped_key_cache: dict[tuple[str, str], tuple[str, ...]] = {}

    def cached_in_size_window(smiles: str) -> bool:
        if smiles not in heavy_cache:
            heavy_cache[smiles] = molecule_in_size_window(smiles)
        return heavy_cache[smiles]

    def cached_has_pg(smiles: str, pg_label: str) -> bool:
        key = (smiles, pg_label)
        if key not in has_pg_cache:
            has_pg_cache[key] = has_pg(smiles, pg_label)
        return has_pg_cache[key]

    def cached_free_key(smiles: str) -> str | None:
        if smiles not in free_key_cache:
            free_key_cache[smiles] = free_scaffold_key(smiles)
        return free_key_cache[smiles]

    def cached_stripped_keys(smiles: str, pg_label: str) -> tuple[str, ...]:
        key = (smiles, pg_label)
        if key not in stripped_key_cache:
            stripped_key_cache[key] = stripped_scaffold_keys(smiles, pg_label)
        return stripped_key_cache[key]

    for rec in records.values():
        reactants = [smi for smi in rec.reactants if cached_in_size_window(smi)]
        products = [smi for smi in rec.products if cached_in_size_window(smi)]
        if not reactants or not products:
            continue

        for spec in PROTECTING_GROUPS:
            reactant_pg = [smi for smi in reactants if cached_has_pg(smi, spec.label)]
            product_pg = [smi for smi in products if cached_has_pg(smi, spec.label)]

            for free_smi in reactants:
                free_key = cached_free_key(free_smi)
                if free_key is None:
                    continue
                for protected_smi in product_pg:
                    if free_key in cached_stripped_keys(protected_smi, spec.label):
                        events.append(
                            ProtectionEvent(
                                reaction_index=rec.index,
                                pg_label=spec.label,
                                direction="install",
                                free_smiles=free_smi,
                                protected_smiles=protected_smi,
                                scaffold_key=free_key,
                            )
                        )

            for protected_smi in reactant_pg:
                protected_keys = cached_stripped_keys(protected_smi, spec.label)
                if not protected_keys:
                    continue
                for free_smi in products:
                    free_key = cached_free_key(free_smi)
                    if free_key is None:
                        continue
                    if free_key in protected_keys:
                        events.append(
                            ProtectionEvent(
                                reaction_index=rec.index,
                                pg_label=spec.label,
                                direction="remove",
                                free_smiles=free_smi,
                                protected_smiles=protected_smi,
                                scaffold_key=free_key,
                            )
                        )
    return sorted(
        set(events),
        key=lambda event: (
            event.pg_label,
            event.scaffold_key,
            event.direction,
            event.reaction_index,
            event.free_smiles,
            event.protected_smiles,
        ),
    )


def build_ground_truth_pairs(
    events: list[ProtectionEvent],
    max_pairs_per_group: int,
) -> list[GroundTruthPair]:
    by_key: dict[tuple[str, str], dict[str, list[ProtectionEvent]]] = defaultdict(
        lambda: {"install": [], "remove": []}
    )
    for event in events:
        by_key[(event.pg_label, event.scaffold_key)][event.direction].append(event)

    spec_by_label = {spec.label: spec for spec in PROTECTING_GROUPS}
    pairs: list[GroundTruthPair] = []

    for (pg_label, scaffold_key), grouped in sorted(by_key.items()):
        installs = sorted(grouped["install"], key=lambda event: event.reaction_index)
        removals = sorted(grouped["remove"], key=lambda event: event.reaction_index)
        selected_pair: GroundTruthPair | None = None
        for install in installs:
            for remove in removals:
                if install.reaction_index >= remove.reaction_index:
                    continue
                selected_pair = GroundTruthPair(
                    install_index=install.reaction_index,
                    remove_index=remove.reaction_index,
                    pg_label=pg_label,
                    functional_group=spec_by_label[pg_label].functional_group,
                    scaffold_key=scaffold_key,
                    install_free_smiles=install.free_smiles,
                    install_protected_smiles=install.protected_smiles,
                    remove_protected_smiles=remove.protected_smiles,
                    remove_free_smiles=remove.free_smiles,
                )
                break
            if selected_pair is not None:
                break
        if selected_pair is not None:
            pairs.append(selected_pair)

    pairs = sorted(pairs, key=lambda pair: (pair.pg_label, pair.install_index, pair.remove_index))
    if max_pairs_per_group <= 0:
        return pairs

    capped_pairs: list[GroundTruthPair] = []
    per_group_counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        if per_group_counts[pair.pg_label] >= max_pairs_per_group:
            continue
        capped_pairs.append(pair)
        per_group_counts[pair.pg_label] += 1
    return capped_pairs


def ground_truth_pairs_in_context(
    context_lines: list[str],
    pg_label: str,
    *,
    max_pairs_per_group: int = 0,
) -> list[GroundTruthPair]:
    records = parse_records_from_lines(context_lines)
    context_indices = set(records.keys())
    events = mine_protection_events(records)
    context_pairs = build_ground_truth_pairs(
        [event for event in events if event.pg_label == pg_label],
        max_pairs_per_group,
    )

    if len(records) < FULL_CONTEXT_PAIR_THRESHOLD:
        return context_pairs

    from task14_protecting_group_ground_truth import hardcoded_pairs_for_label

    context_pair_set = {(pair.install_index, pair.remove_index) for pair in context_pairs}
    verified: list[GroundTruthPair] = []
    for pair in hardcoded_pairs_for_label(pg_label):
        if pair.install_index not in context_indices or pair.remove_index not in context_indices:
            continue
        key = (pair.install_index, pair.remove_index)
        if key in context_pair_set:
            verified.append(pair)
    if verified:
        if max_pairs_per_group > 0:
            return verified[:max_pairs_per_group]
        return verified
    return context_pairs


def parse_response(response: str) -> set[tuple[int, int]]:
    text = response.strip()
    if not text or text.replace(" ", "") == "-1":
        return set()

    pairs: set[tuple[int, int]] = set()
    for line in text.splitlines():
        nums = re.findall(r"\d+", line)
        if len(nums) < 2:
            continue
        install_idx, remove_idx = int(nums[0]), int(nums[1])
        if install_idx != remove_idx:
            pairs.add((install_idx, remove_idx))
    return pairs


def precision_recall_f1(
    predicted: set[tuple[int, int]],
    ground_truth: set[tuple[int, int]],
) -> tuple[float, float, float]:
    tp = len(predicted & ground_truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def summarize_gt_pair(pair: GroundTruthPair) -> dict[str, object]:
    return {
        "install_index": pair.install_index,
        "remove_index": pair.remove_index,
        "pg_label": pair.pg_label,
        "functional_group": pair.functional_group,
        "scaffold_key": pair.scaffold_key,
        "install_free_smiles": pair.install_free_smiles,
        "install_protected_smiles": pair.install_protected_smiles,
        "remove_protected_smiles": pair.remove_protected_smiles,
        "remove_free_smiles": pair.remove_free_smiles,
    }


def count_by_label(items: list[ProtectionEvent] | list[GroundTruthPair]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item.pg_label] += 1
    return dict(sorted(counts.items()))


def build_question(spec: ProtectingGroupSpec, max_pairs: int) -> str:
    pair_limit_instruction = (
        "Return all valid pairs for this protecting group that are present in the provided context."
        if max_pairs <= 0
        else (
            f"Return at most {max_pairs} pairs for this protecting group from the provided context, "
            "choosing the earliest pairs after sorting by install index and then removal index."
        )
    )
    return f"""
    There is a list of chemical reactions in SMILES format in the provided context, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Each side may contain multiple species separated by dots (.).
    Ignore reagents (the middle field between the two > delimiters).

    Task:
    Find ALL protecting-group install/remove reaction pairs (A, B) in the provided context for:
    - label: {spec.label}
    - aliases: {", ".join(spec.aliases)}
    - functional group protected: {spec.functional_group}
    - description: {spec.description}

    A valid pair has:
    - A installs {spec.label} onto a free {spec.functional_group} site.
    - B later removes {spec.label} from the same underlying scaffold.
    - The install reaction index must be smaller than the removal reaction index.
    - Both reactions must appear in the provided context.

    Rules:
    - Use RDKit for canonicalization and SMARTS/substructure checks.
    - Split multi-component sides on dots (.) and canonicalize each component independently.
    - Treat two species as the same scaffold if removing the protecting-group atoms from the
      protected species gives the same canonical free-scaffold SMILES.
    - Exclude species with fewer than {MIN_HEAVY_ATOMS} or more than {MAX_HEAVY_ATOMS} heavy atoms.
    - {pair_limit_instruction}
    - If several valid pairs share the same scaffold and protecting group, prefer the earliest
      install reaction and the earliest later removal reaction.
    - Skip malformed reactions or species that RDKit cannot parse.

    Output format:
    - Return each valid pair on its own line as "install_index,remove_index".
    - Sort pairs by install index, then removal index.
    - No labels, explanations, quotes, JSON, markdown, or other punctuation.
    - If no pair exists for {spec.label} in the context, return -1.
    """


RLM_CODE_GUIDANCE = """
    - DO NOT assume/simulate output of code. Wait for code execution and only then return.
    - DO NOT USE `FINAL` for writing a thought/comment. Only use `FINAL` for the final answer.
""".strip()


def build_rlm_question(spec: ProtectingGroupSpec, max_pairs: int) -> str:
    return f"{build_question(spec, max_pairs)}\n\nGuidance:\n{RLM_CODE_GUIDANCE}"
