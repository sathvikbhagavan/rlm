"""Tier4 task17b: 2- or 3-step sequential synthesis chains via Rxn-INSIGHT SMIRKS."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from task17_smirks_sequential_graph import (
    DATASET_TOTAL_REACTIONS,
    PATT_ALCOHOL,
    PATT_ARYL_AMINE,
    PATT_ARYL_N,
    PATT_BIARYL,
    PATT_BOC_AMINE_LOOSE,
    PATT_ESTER,
    ReactionRecord,
    SmirksTemplate,
    _count,
    _has,
    _mol,
    _template_for_name,
    build_line_mol_cache,
    classify_reactions,
    load_smirks_entries,
    organic_components,
    parse_records_from_lines,
    pick_spine,
    pick_spine2,
    spine_candidates,
)

CHAIN_LENGTHS: tuple[int, ...] = (2, 3)
DATASET_TOTAL_REACTIONS = DATASET_TOTAL_REACTIONS

PATT_AZIDE = "[N;X2]=[N+]=[N-]"
PATT_AZIDE_LOOSE = "N=[N+]=[N-]"
PATT_ARYL_HALIDE = "[c][Cl,Br,I]"
PATT_BORONIC = "[BX3]([OH])[OH]"
PATT_BORONIC_ESTER = "B1OC(C)(C)C(C)(C)O1"


@dataclass(frozen=True)
class QuestionSpec:
    question_id: str
    label: str
    description: str
    chain_length: int
    step_summaries: tuple[str, ...]
    step_template_names: tuple[str, ...]
    step_template_occurrences: tuple[int, ...] = field(default_factory=tuple)
    persistence_summary: str = ""

    def __post_init__(self) -> None:
        if self.chain_length not in CHAIN_LENGTHS:
            raise ValueError(f"chain_length must be one of {CHAIN_LENGTHS}")
        if len(self.step_summaries) != self.chain_length:
            raise ValueError("step_summaries length must match chain_length")
        if len(self.step_template_names) != self.chain_length:
            raise ValueError("step_template_names length must match chain_length")
        occ = self.step_template_occurrences or (0,) * self.chain_length
        if len(occ) != self.chain_length:
            raise ValueError("step_template_occurrences length must match chain_length")
        object.__setattr__(self, "step_template_occurrences", occ)


@dataclass(frozen=True)
class Chain:
    reaction_indices: tuple[int, ...]
    spine_smiles: tuple[str, ...]


PersistenceFn = Callable[[tuple[str, ...]], bool]


def persistence_azide_staudinger(spines: tuple[str, ...]) -> bool:
    """Aryl azide from step 1 is reduced to a primary amine in step 2."""
    if len(spines) != 2:
        return False
    m0, m1 = _mol(spines[0]), _mol(spines[1])
    if not (_has(m0, PATT_AZIDE) or _has(m0, PATT_AZIDE_LOOSE)):
        return False
    if _has(m1, PATT_AZIDE) or _has(m1, PATT_AZIDE_LOOSE):
        return False
    return _count(m1, PATT_ARYL_AMINE) >= _count(m0, PATT_ARYL_AMINE)


def persistence_aryl_brom_negishi(spines: tuple[str, ...]) -> bool:
    """New biaryl C–C bond from Negishi coupling; aryl halide handle consumed."""
    if len(spines) != 2:
        return False
    m0, m1 = _mol(spines[0]), _mol(spines[1])
    if _count(m1, PATT_BIARYL) <= _count(m0, PATT_BIARYL):
        return False
    return _count(m1, PATT_ARYL_HALIDE) < _count(m0, PATT_ARYL_HALIDE)


def persistence_boronic_suzuki(spines: tuple[str, ...]) -> bool:
    """Boronic acid/ester installed in step 1 is consumed in Suzuki biaryl formation."""
    if len(spines) != 2:
        return False
    m0, m1 = _mol(spines[0]), _mol(spines[1])
    if _count(m1, PATT_BIARYL) <= _count(m0, PATT_BIARYL):
        return False
    boron_before = int(_has(m0, PATT_BORONIC) or _has(m0, PATT_BORONIC_ESTER))
    boron_after = int(_has(m1, PATT_BORONIC) or _has(m1, PATT_BORONIC_ESTER))
    return boron_after < boron_before or boron_before == 0


def persistence_boc_deprot_buchwald(spines: tuple[str, ...]) -> bool:
    """Boc removed in step 1; Buchwald installs a new aryl–N bond in step 2."""
    if len(spines) != 2:
        return False
    m0, m1 = _mol(spines[0]), _mol(spines[1])
    if _has(m0, PATT_BOC_AMINE_LOOSE):
        return False
    return _count(m1, PATT_ARYL_N) > _count(m0, PATT_ARYL_N)


def persistence_aryl_brom_suzuki_ester_red(spines: tuple[str, ...]) -> bool:
    """Biaryl from Suzuki persists; ester on linker is reduced to primary alcohol."""
    if len(spines) != 3:
        return False
    m0, m1, m2 = _mol(spines[0]), _mol(spines[1]), _mol(spines[2])
    if _count(m1, PATT_BIARYL) <= _count(m0, PATT_BIARYL):
        return False
    if not _has(m1, PATT_ESTER):
        return False
    if _has(m2, PATT_ESTER):
        return False
    return _has(m2, PATT_ALCOHOL)


PERSISTENCE_BY_QUESTION: dict[str, PersistenceFn] = {
    "azide_staudinger": persistence_azide_staudinger,
    "aryl_brom_negishi": persistence_aryl_brom_negishi,
    "boronic_suzuki": persistence_boronic_suzuki,
    "boc_deprot_buchwald": persistence_boc_deprot_buchwald,
    "aryl_brom_suzuki_ester_red": persistence_aryl_brom_suzuki_ester_red,
}


def build_question_specs() -> list[QuestionSpec]:
    return [
        QuestionSpec(
            question_id="azide_staudinger",
            label="Azide formation → Staudinger reduction",
            description=(
                "Convert a substrate-attached halide to the corresponding organic azide, then "
                "reduce that azide to a primary amine on the same substrate skeleton."
            ),
            chain_length=2,
            step_summaries=(
                "Azide formation: a substrate-attached halide (chloride, bromide, or iodide) is "
                "converted to an organic azide using a sodium-azide source (e.g. NaN₃).",
                "Staudinger reduction: the substrate azide is reduced with a phosphine reagent "
                "to a primary amine at the same carbon.",
            ),
            step_template_names=(
                "Formation of Azides from halogens",
                "Azide to amine reduction (Staudinger)",
            ),
            persistence_summary=(
                "The azide installed in step 1 is consumed in step 2; the product bears a "
                "primary aryl amine where the azide was."
            ),
        ),
        QuestionSpec(
            question_id="aryl_brom_negishi",
            label="Aromatic bromination → Negishi coupling",
            description=(
                "Install an aryl bromide handle, then couple it via Negishi cross-coupling "
                "to form a new biaryl C–C bond while the rest of the substrate persists."
            ),
            chain_length=2,
            step_summaries=(
                "Aromatic bromination: electrophilic bromination converts an aromatic C–H to "
                "C–Br (aryl bromide), typically with NBS or another brominating agent.",
                "Negishi coupling: an aryl halide couples with an organozinc reagent under "
                "Pd catalysis to form a new aryl–aryl C–C bond (not a boronic-acid Suzuki "
                "coupling).",
            ),
            step_template_names=(
                "Aromatic bromination",
                "{Negishi}",
            ),
            persistence_summary=(
                "The substrate gains a biaryl linkage in step 2; the aryl bromide handle from "
                "step 1 is consumed in the coupling."
            ),
        ),
        QuestionSpec(
            question_id="boronic_suzuki",
            label="Boronic acid preparation → Suzuki coupling",
            description=(
                "Prepare a boronic acid (or boronic ester) on the substrate, then use it in "
                "a Suzuki cross-coupling to form a biaryl bond."
            ),
            chain_length=2,
            step_summaries=(
                "Boronic acid preparation: an aryl or vinyl bromide on the substrate is "
                "converted to an aryl boronic acid via lithiation and reaction with a "
                "boronate ester.",
                "Suzuki coupling: an aryl halide couples with an aryl boronic acid (not an "
                "organozinc reagent) under Pd catalysis to form a new aryl–aryl C–C bond.",
            ),
            step_template_names=(
                "Preparation of boronic acids",
                "{Suzuki}",
            ),
            persistence_summary=(
                "The boronic acid/ester from step 1 is consumed in the Suzuki step; the product "
                "contains a new biaryl bond."
            ),
        ),
        QuestionSpec(
            question_id="boc_deprot_buchwald",
            label="Boc deprotection → Buchwald-Hartwig",
            description=(
                "Remove a Boc protecting group to reveal a free amine, then arylate that amine "
                "via Buchwald–Hartwig coupling on the same molecule."
            ),
            chain_length=2,
            step_summaries=(
                "Boc deprotection: a tert-butoxycarbonyl (Boc) carbamate on nitrogen is removed "
                "under acidic conditions, yielding a free amine.",
                "Buchwald–Hartwig N-arylation: an aryl halide (chloride, bromide, or iodide) "
                "couples to an aniline-type amine nitrogen under Pd catalysis, forming a new "
                "aryl–N bond.",
            ),
            step_template_names=(
                "Boc amine deprotection",
                "{Buchwald-Hartwig}",
            ),
            persistence_summary=(
                "The Boc group is absent after step 1; step 2 adds a new aryl–N bond on the deprotected amine."
            ),
        ),
        QuestionSpec(
            question_id="aryl_brom_suzuki_ester_red",
            label="Bromination → Suzuki → ester reduction",
            description=(
                "Install an aryl bromide, couple it by Suzuki to build a biaryl, then reduce "
                "an ester on the linker to a primary alcohol while the biaryl bond remains intact."
            ),
            chain_length=3,
            step_summaries=(
                "Aromatic bromination: electrophilic bromination converts an aromatic C–H to "
                "C–Br (aryl bromide), typically with NBS or another brominating agent.",
                "Suzuki coupling: an aryl halide couples with an aryl boronic acid under Pd "
                "catalysis to form a new aryl–aryl C–C bond.",
                "Ester reduction to primary alcohol: a carboxylic ester on the substrate is "
                "reduced with a hydride reagent (e.g. LiAlH₄/LAH) to the corresponding primary "
                "alcohol.",
            ),
            step_template_names=(
                "Aromatic bromination",
                "{Suzuki}",
                "Reduction of ester to primary alcohol",
            ),
            persistence_summary=(
                "The biaryl bond from step 2 persists through step 3; the ester present after "
                "step 2 is absent in the final product and replaced by a primary alcohol."
            ),
        ),
    ]


BUILTIN_QUESTIONS: tuple[QuestionSpec, ...] = tuple(build_question_specs())
QUESTION_IDS: tuple[str, ...] = tuple(q.question_id for q in BUILTIN_QUESTIONS)
QUESTION_BY_ID: dict[str, QuestionSpec] = {q.question_id: q for q in BUILTIN_QUESTIONS}


def templates_for_question(
    spec: QuestionSpec,
    entries: list[dict[str, str]] | None = None,
) -> tuple[SmirksTemplate, ...]:
    entries = entries or load_smirks_entries()
    return tuple(
        _template_for_name(entries, name, occurrence=occ)
        for name, occ in zip(spec.step_template_names, spec.step_template_occurrences)
    )


def classify_question_steps(
    lines: list[str],
    spec: QuestionSpec,
    entries: list[dict[str, str]] | None = None,
    *,
    mol_cache: dict[int, tuple] | None = None,
) -> tuple[list[dict[int, list[str]]], tuple[SmirksTemplate, ...]]:
    entries = entries or load_smirks_entries()
    templates = templates_for_question(spec, entries)
    hits = [
        classify_reactions(lines, [template], mol_cache=mol_cache)
        for template in templates
    ]
    return hits, templates


def pick_spine_at_step(rec: ReactionRecord, prev_spine: str) -> str | None:
    return pick_spine2(rec, prev_spine)


def build_chain_spines(
    chain: tuple[int, ...],
    records: dict[int, ReactionRecord],
) -> tuple[str, ...] | None:
    spines: list[str] = []
    for i in range(len(chain) - 1):
        rec_a = records[chain[i]]
        rec_b = records[chain[i + 1]]
        link = pick_spine(spine_candidates(rec_a, rec_b))
        if link is None:
            return None
        if i == 0:
            spines.append(link)
        spine_after = pick_spine_at_step(rec_b, link)
        if spine_after is None:
            return None
        spines.append(spine_after)
    return tuple(spines)


def enumerate_chains(
    records: dict[int, ReactionRecord],
    step_index_sets: list[set[int]],
    *,
    question_id: str,
    chain_length: int,
    require_persistence: bool = True,
    max_chains: int = 100_000,
) -> list[Chain]:
    from collections import defaultdict

    persist_fn = PERSISTENCE_BY_QUESTION[question_id]
    chains: list[Chain] = []
    seen: set[tuple[int, ...]] = set()

    reactant_index: list[dict[str, list[int]]] = []
    for step_set in step_index_sets[1:]:
        idx: dict[str, list[int]] = defaultdict(list)
        for r in step_set:
            for smi in organic_components(records[r].reactants):
                idx[smi].append(r)
        reactant_index.append(idx)

    def try_add(chain_indices: list[int]) -> bool:
        key = tuple(chain_indices)
        if key in seen:
            return False
        spines = build_chain_spines(key, records)
        if spines is None:
            return False
        if require_persistence and not persist_fn(spines):
            return False
        seen.add(key)
        chains.append(Chain(reaction_indices=key, spine_smiles=spines))
        return len(chains) >= max_chains

    def extend(chain_indices: list[int], step: int) -> bool:
        if step == chain_length - 1:
            return try_add(chain_indices)
        r_cur = chain_indices[-1]
        rec_cur = records[r_cur]
        idx = reactant_index[step]
        for smi in organic_components(rec_cur.products):
            for r_next in idx.get(smi, ()):
                if r_next in chain_indices:
                    continue
                if r_next not in step_index_sets[step + 1]:
                    continue
                if extend(chain_indices + [r_next], step + 1):
                    return True
        return False

    for r0 in sorted(step_index_sets[0]):
        if chain_length == 2:
            rec0 = records[r0]
            for smi in organic_components(rec0.products):
                for r1 in reactant_index[0].get(smi, ()):
                    if r1 == r0 or r1 not in step_index_sets[1]:
                        continue
                    if try_add([r0, r1]):
                        return chains
        else:
            if extend([r0], 0):
                return chains
    return chains


def verify_chain(
    chain: tuple[int, ...],
    question_id: str,
    records: dict[int, ReactionRecord],
    lines_by_index: dict[int, str],
    *,
    step_hits: list[set[int]],
    chain_length: int,
    require_persistence: bool = True,
) -> tuple[bool, str]:
    if len(chain) != chain_length:
        return False, f"expected {chain_length} reactions"
    if len(set(chain)) != chain_length:
        return False, "duplicate indices"
    for step, idx in enumerate(chain):
        if idx not in step_hits[step]:
            return False, f"step{step + 1} SMIRKS mismatch"
    spines = build_chain_spines(chain, records)
    if spines is None:
        return False, "no canonical spine link"
    if require_persistence and not PERSISTENCE_BY_QUESTION[question_id](spines):
        return False, "persistence check failed"
    _ = lines_by_index
    return True, "ok"


def parse_chain_response(text: str, *, chain_length: int | None = None) -> list[tuple[int, ...]]:
    text = text.strip()
    if not text or text.replace(" ", "") == "-1":
        return []

    chains: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "-1":
            continue
        nums = [int(n) for n in re.findall(r"\d+", line)]
        if not nums:
            continue
        if chain_length is not None:
            if len(nums) < chain_length:
                continue
            if len(nums) % chain_length == 0:
                chunks = [
                    tuple(nums[i : i + chain_length])
                    for i in range(0, len(nums), chain_length)
                ]
            else:
                chunks = [tuple(nums[:chain_length])]
        else:
            for length in (3, 2):
                if len(nums) >= length and len(nums) % length == 0:
                    chunks = [
                        tuple(nums[i : i + length])
                        for i in range(0, len(nums), length)
                    ]
                    break
            else:
                continue
        for chain in chunks:
            if chain not in seen:
                seen.add(chain)
                chains.append(chain)
    return chains


def precision_recall_f1(
    predicted: set[tuple[int, ...]],
    ground_truth: set[tuple[int, ...]],
) -> tuple[float, float, float]:
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def question_prompt(spec: QuestionSpec) -> str:
    step_lines = "\n".join(
        f"- Step {i + 1} (r_{i}): {summary}"
        for i, summary in enumerate(spec.step_summaries)
    )
    indices = ", ".join(f"r_{i}" for i in range(spec.chain_length))
    return f"""
There is a list of chemical reactions in SMILES format in the provided context, separated by newlines.
Each reaction is in one of these forms:
- "index reactants>reagents>products"
- "index reactants>>products"

Each side may contain multiple species separated by dots (.).
Reagents are in the middle field between the two > delimiters when present.

Task:
Find ALL valid {spec.chain_length}-reaction chains [{indices}] in the context where:
{step_lines}
- Consecutive reactions must link: at least one canonical-SMILES product component of r_i must be
  identical to at least one canonical-SMILES reactant component of r_{{i+1}} (exact equality on
  dot-separated components), for each i.
- Do not reuse the same reaction index twice in one chain.
- Only use reactions present in the provided context.

Output format:
- Return each chain as a comma-separated list of exactly {spec.chain_length} reaction indices, one chain per line.
- Sort chains in lexicographic (ascending) order.
- No other text, quotes, labels, punctuation, JSON, or formatting.
- If no valid chain exists, return -1.
""".strip()


def build_rlm_question(spec: QuestionSpec) -> str:
    return question_prompt(spec)


def smirks_documentation(
    spec: QuestionSpec,
    entries: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    entries = entries or load_smirks_entries()
    templates = templates_for_question(spec, entries)
    return {
        "chain_length": spec.chain_length,
        "steps": [
            {
                "step": i + 1,
                "template_name": spec.step_template_names[i],
                "template_occurrence": spec.step_template_occurrences[i],
                "smirks": templates[i].smirks,
            }
            for i in range(spec.chain_length)
        ],
    }
