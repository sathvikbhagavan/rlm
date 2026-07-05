"""Tier4 task17: 2-step sequential synthesis chains via Rxn-INSIGHT SMIRKS."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions

from task11_synthetic_chain_graph import canonicalize_components

RDLogger.DisableLog("rdApp.*")

CHAIN_LENGTH = 2
MIN_HEAVY_ATOMS = 3
DATASET_TOTAL_REACTIONS = 122_456
SMIRKS_PATH = Path(__file__).resolve().parent / "data" / "smirks.json"


@dataclass(frozen=True)
class SmirksTemplate:
    name: str
    smirks: str
    key: str


@dataclass(frozen=True)
class ReactionRecord:
    index: int
    raw: str
    reactants: tuple[str, ...]
    reagents: tuple[str, ...]
    products: tuple[str, ...]


@dataclass(frozen=True)
class QuestionSpec:
    question_id: str
    label: str
    description: str
    step1_summary: str
    step2_summary: str
    step1_template_name: str
    step2_template_name: str
    persistence_summary: str
    step1_template_occurrence: int = 0
    step2_template_occurrence: int = 0


@dataclass(frozen=True)
class Chain:
    reaction_indices: tuple[int, int]
    spine_smiles: tuple[str, str]  # after step1, after step2


# --- SMIRKS loading / matching (task6-style RunReactants) ---


def load_smirks_entries(path: Path = SMIRKS_PATH) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _template_for_name(
    entries: list[dict[str, str]],
    name: str,
    *,
    occurrence: int = 0,
) -> SmirksTemplate:
    seen = 0
    for idx, entry in enumerate(entries):
        if entry["name"] != name:
            continue
        if seen == occurrence:
            return SmirksTemplate(
                name=entry["name"],
                smirks=entry["smirks"],
                key=f"{entry['name']}::{idx}",
            )
        seen += 1
    raise ValueError(f"No SMIRKS template found for name={name!r} occurrence={occurrence}")


def build_reaction_query(smarts: str) -> rdChemReactions.ChemicalReaction:
    query = rdChemReactions.ReactionFromSmarts(smarts)
    if query is None:
        raise ValueError(f"Failed to parse reaction SMARTS: {smarts}")
    return query


def parse_reaction_mols(indexed_line: str) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    _, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    reactant_smiles = [s for s in parts[0].split(".") if s]
    product_smiles = [s for s in parts[-1].split(".") if s]
    reactants = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    products = [Chem.MolFromSmiles(s) for s in product_smiles]
    return [m for m in reactants if m is not None], [m for m in products if m is not None]


def reaction_matches_smirks_cached(
    reactants: list[Chem.Mol],
    products: list[Chem.Mol],
    query_reaction: rdChemReactions.ChemicalReaction,
) -> bool:
    template = query_reaction
    actual_product_smiles = {Chem.MolToSmiles(m) for m in products}
    num_template_reactants = template.GetNumReactantTemplates()
    r_templates = list(template.GetReactants())
    if len(reactants) < num_template_reactants:
        return False
    for perm in permutations(reactants, num_template_reactants):
        if not all(perm[i].HasSubstructMatch(r_templates[i]) for i in range(num_template_reactants)):
            continue
        try:
            product_sets = template.RunReactants(perm)
        except Exception:
            continue
        for prod_set in product_sets:
            generated_smiles: set[str] = set()
            for mol in prod_set:
                try:
                    Chem.SanitizeMol(mol)
                    generated_smiles.add(Chem.MolToSmiles(mol))
                except Exception:
                    continue
            if generated_smiles and generated_smiles.issubset(actual_product_smiles):
                return True
    return False


def reaction_matches_smirks(
    indexed_line: str,
    query_reaction: rdChemReactions.ChemicalReaction,
) -> bool:
    reactants, products = parse_reaction_mols(indexed_line)
    return reaction_matches_smirks_cached(reactants, products, query_reaction)


def build_line_mol_cache(lines: list[str]) -> dict[int, tuple[list[Chem.Mol], list[Chem.Mol]]]:
    cache: dict[int, tuple[list[Chem.Mol], list[Chem.Mol]]] = {}
    for line in lines:
        idx_str, _ = line.split(" ", 1)
        cache[int(idx_str)] = parse_reaction_mols(line)
    return cache


def classify_reactions(
    lines: list[str],
    templates: list[SmirksTemplate],
    *,
    mol_cache: dict[int, tuple[list[Chem.Mol], list[Chem.Mol]]] | None = None,
) -> dict[int, list[str]]:
    """Map reaction index -> list of matching template keys."""
    if mol_cache is None:
        mol_cache = build_line_mol_cache(lines)
    queries: dict[str, rdChemReactions.ChemicalReaction] = {}
    for t in templates:
        q = build_reaction_query(t.smirks)
        q.Initialize()
        queries[t.key] = q
    hits: dict[int, list[str]] = defaultdict(list)
    for idx, mols in mol_cache.items():
        reactants, products = mols
        for template in templates:
            if reaction_matches_smirks_cached(reactants, products, queries[template.key]):
                hits[idx].append(template.key)
    return dict(hits)


# --- Record parsing ---


def split_reaction_line(line: str) -> tuple[int, str, str, str]:
    idx_str, reaction_smiles = line.split(" ", 1)
    parts = reaction_smiles.split(">")
    if len(parts) == 2:
        reactants_raw, products_raw = parts
        reagents_raw = ""
    elif len(parts) == 3:
        reactants_raw, reagents_raw, products_raw = parts
    else:
        raise ValueError(f"Bad reaction line: {line[:80]}")
    return int(idx_str), reactants_raw, reagents_raw, products_raw


def parse_records_from_lines(lines: list[str]) -> dict[int, ReactionRecord]:
    records: dict[int, ReactionRecord] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            idx, reactants_raw, reagents_raw, products_raw = split_reaction_line(line)
        except Exception:
            continue
        reactants = tuple(canonicalize_components(reactants_raw))
        reagents = tuple(canonicalize_components(reagents_raw))
        products = tuple(canonicalize_components(products_raw))
        if not reactants or not products:
            continue
        records[idx] = ReactionRecord(
            index=idx,
            raw=line.split(" ", 1)[1] if " " in line else line,
            reactants=reactants,
            reagents=reagents,
            products=products,
        )
    return records


def organic_components(components: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(s for s in components if heavy_atom_count(s) >= MIN_HEAVY_ATOMS)


def heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _mol(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


def _has(mol: Chem.Mol | None, smarts: str) -> bool:
    patt = Chem.MolFromSmarts(smarts)
    return mol is not None and patt is not None and mol.HasSubstructMatch(patt)


def _count(mol: Chem.Mol | None, smarts: str) -> int:
    patt = Chem.MolFromSmarts(smarts)
    if mol is None or patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def spine_candidates(rec1: ReactionRecord, rec2: ReactionRecord) -> set[str]:
    p1 = set(organic_components(rec1.products))
    r2 = set(organic_components(rec2.reactants))
    return p1 & r2


def pick_spine(candidates: set[str]) -> str | None:
    if not candidates:
        return None
    return max(candidates, key=heavy_atom_count)


# --- Per-question persistence checks on the linking substrate ---


PATT_BOC_AMINE = "[C;H3;D1;+0]-[C;H0;D4;+0](-[C;H3;D1;+0])(-[C;H3;D1;+0])-[O;H0;D2;+0]-[C;H0;D3;+0](=[O;H0;D1;+0])-[#7]"
PATT_BOC_AMINE_LOOSE = "[NX3]C(=O)OC(C)(C)C"
PATT_ARYL_N = "[c]-[NX3]"
PATT_ARYL_AMINE = "[NX3;H2][c]"
PATT_ARYL_AMIDE = "[c][NX3H1][CX3](=O)"
PATT_BIARYL = "[#6;a]-[#6;a]"
PATT_CARBONYL = "[CX3]=[OX1]"
PATT_ALCOHOL = "[CX4][OH]"
PATT_ENONE = "[CX3](=O)[CX3]=[CX3]"
PATT_ENONE_SAT = "[CX3](=O)[CX4][CX4]"
PATT_ESTER = "[CX3](=O)[OX2][#6]"
PATT_ALKENE = "[CX3]=[CX3]"
PATT_ALDEHYDE = "[CX3H1](=O)[#6]"
PATT_SECONDARY_AMINE = "[CX4][NX3H1][#6]"


def persistence_q1(spine1: str, spine2: str) -> bool:
    """Boc persists; new aryl-N bond in step 2."""
    m1, m2 = _mol(spine1), _mol(spine2)
    if not (_has(m1, PATT_BOC_AMINE_LOOSE) and _has(m2, PATT_BOC_AMINE_LOOSE)):
        return False
    return _count(m2, PATT_ARYL_N) > _count(m1, PATT_ARYL_N)


def persistence_q2(spine1: str, spine2: str) -> bool:
    """Aryl amine from nitro reduction is acylated."""
    m1, m2 = _mol(spine1), _mol(spine2)
    if not _has(m1, PATT_ARYL_AMINE):
        return False
    return _has(m2, PATT_ARYL_AMIDE)


def persistence_q3(spine1: str, spine2: str) -> bool:
    """Biaryl persists; ester reduced to primary alcohol."""
    m1, m2 = _mol(spine1), _mol(spine2)
    if _count(m2, PATT_BIARYL) < _count(m1, PATT_BIARYL):
        return False
    if not _has(m1, PATT_ESTER):
        return False
    if _has(m2, PATT_ESTER):
        return False
    return _has(m2, PATT_ALCOHOL)


def persistence_q4(spine1: str, spine2: str) -> bool:
    """Wittig alkene hydrogenated to a single bond."""
    m1, m2 = _mol(spine1), _mol(spine2)
    if not _has(m1, PATT_ALKENE):
        return False
    return _count(m1, PATT_ALKENE) > _count(m2, PATT_ALKENE)


def persistence_q5(spine1: str, spine2: str) -> bool:
    """Carbonyl substrate gains a secondary amine; aldehyde not left unchanged."""
    m1, m2 = _mol(spine1), _mol(spine2)
    if not _has(m1, PATT_CARBONYL):
        return False
    if _count(m2, PATT_SECONDARY_AMINE) <= _count(m1, PATT_SECONDARY_AMINE):
        return False
    if _has(m1, PATT_ALDEHYDE) and _has(m2, PATT_ALDEHYDE):
        return False
    return True


PERSISTENCE_BY_QUESTION = {
    "boc_buchwald": persistence_q1,
    "nitro_reduction_amide": persistence_q2,
    "suzuki_carbonyl_reduction": persistence_q3,
    "wittig_hydrogenation": persistence_q4,
    "alcohol_oxidation_reductive_amination": persistence_q5,
}


# --- Question definitions (one Rxn-INSIGHT SMIRKS template per step) ---


def build_question_specs(entries: list[dict[str, str]] | None = None) -> list[QuestionSpec]:
    _ = entries  # specs use fixed template names from data/smirks.json
    return [
        QuestionSpec(
            question_id="boc_buchwald",
            label="Boc protect → Buchwald-Hartwig",
            description=(
                "Install a Boc (tert-butoxycarbonyl) group on a primary or secondary amine "
                "using Boc₂O, so that a different nitrogen on the same molecule can be "
                "arylated in the next step without competing with the free amine."
            ),
            step1_summary=(
                "Boc protection of an amine: a free amine is converted to a Boc carbamate "
                "on the substrate using Boc₂O (di-tert-butyl dicarbonate)."
            ),
            step2_summary=(
                "Buchwald–Hartwig N-arylation: an aryl halide (chloride, bromide, or iodide) "
                "couples to an aniline-type amine nitrogen on the substrate under Pd "
                "catalysis, forming a new aryl–N bond."
            ),
            step1_template_name="Boc amine protection with Boc anhydride",
            step2_template_name="{Buchwald-Hartwig}",
            persistence_summary=(
                "The Boc carbamate installed in step 1 must still be present on the "
                "substrate entering step 2 and on the final product; step 2 adds a new "
                "aryl–N bond elsewhere on the molecule."
            ),
        ),
        QuestionSpec(
            question_id="nitro_reduction_amide",
            label="Nitro reduction → Schotten-Baumann acylation",
            description=(
                "Reduce an aryl nitro group to a primary aniline, then acylate that "
                "newly revealed nucleophile. The nitro is inert to acylation; the reduction "
                "creates the reactive amine that step 2 consumes."
            ),
            step1_summary=(
                "Nitro reduction: an aryl nitro group on the substrate is reduced to a "
                "primary aryl amine using catalytic hydrogenation, iron, or another "
                "nitro-reducing system."
            ),
            step2_summary=(
                "Schotten–Baumann amide coupling: the substrate aryl primary amine reacts "
                "with an acyl chloride to form an aryl amide."
            ),
            step1_template_name="Reduction of nitro groups to amines",
            step2_template_name="Acyl chloride with primary amine to amide (Schotten-Baumann)",
            persistence_summary=(
                "The aryl primary amine produced in step 1 is the substrate for step 2; "
                "the product bears a new amide bond on that aryl ring."
            ),
        ),
        QuestionSpec(
            question_id="suzuki_carbonyl_reduction",
            label="Suzuki coupling → ester reduction",
            description=(
                "Couple an aryl halide to an aryl boronic acid by Suzuki reaction on a "
                "substrate that already bears an ester, then reduce that ester to a primary "
                "alcohol while the new biaryl bond remains intact."
            ),
            step1_summary=(
                "Suzuki cross-coupling: an aryl halide couples with an aryl boronic acid "
                "(not an organozinc reagent) under Pd catalysis to form a new aryl–aryl "
                "C–C bond on the substrate."
            ),
            step2_summary=(
                "Ester reduction to primary alcohol: a carboxylic ester on the substrate is "
                "reduced with a hydride reagent (e.g. LiAlH₄/LAH) to the corresponding "
                "primary alcohol."
            ),
            step1_template_name="{Suzuki}",
            step2_template_name="Reduction of ester to primary alcohol",
            persistence_summary=(
                "The biaryl bond from step 1 persists; the ester present after step 1 is "
                "absent in the final product and replaced by a primary alcohol."
            ),
        ),
        QuestionSpec(
            question_id="wittig_hydrogenation",
            label="Wittig olefination → alkene hydrogenation",
            description=(
                "Form an alkene on the substrate via Wittig olefination, then hydrogenate "
                "that newly formed C=C double bond under conditions that saturate the alkene."
            ),
            step1_summary=(
                "Wittig reaction: a carbonyl on the substrate reacts with a phosphonium "
                "ylide to form a new C=C double bond (alkene)."
            ),
            step2_summary=(
                "Alkene hydrogenation: a C=C double bond on the substrate is hydrogenated "
                "under catalytic H₂ and Pd to a saturated C–C single bond."
            ),
            step1_template_name="Wittig with Phosphonium",
            step2_template_name="Hydrogenation (double to single)",
            persistence_summary=(
                "The alkene installed in step 1 is the substrate for step 2; the product "
                "has fewer C=C double bonds on that same substrate skeleton."
            ),
        ),
        QuestionSpec(
            question_id="alcohol_oxidation_reductive_amination",
            label="Alcohol oxidation → reductive amination",
            description=(
                "Oxidize an alcohol to a carbonyl (aldehyde or ketone), then use that "
                "carbonyl in a reductive amination with an amine to form a new C–N bond. "
                "The carbonyl is the intermediate that bridges the two steps."
            ),
            step1_summary=(
                "Alcohol oxidation: a primary or secondary alcohol on the substrate is "
                "oxidized to the corresponding aldehyde or ketone."
            ),
            step2_summary=(
                "Reductive amination: a substrate carbonyl condenses with an amine and is "
                "reduced (e.g. with a borohydride or hydride source) to give a secondary "
                "amine with a new C–N bond."
            ),
            step1_template_name="Oxidation or Dehydrogenation of Alcohols to Aldehydes and Ketones",
            step1_template_occurrence=1,
            step2_template_name="{reductive amination}",
            persistence_summary=(
                "The carbonyl formed in step 1 is consumed in step 2; the product "
                "contains a new secondary amine at that carbon."
            ),
        ),
    ]


BUILTIN_QUESTIONS: tuple[QuestionSpec, ...] = tuple(build_question_specs())
QUESTION_IDS: tuple[str, ...] = tuple(q.question_id for q in BUILTIN_QUESTIONS)
QUESTION_BY_ID: dict[str, QuestionSpec] = {q.question_id: q for q in BUILTIN_QUESTIONS}


def templates_for_question(
    spec: QuestionSpec,
    entries: list[dict[str, str]] | None = None,
) -> tuple[SmirksTemplate, SmirksTemplate]:
    entries = entries or load_smirks_entries()
    return (
        _template_for_name(entries, spec.step1_template_name, occurrence=spec.step1_template_occurrence),
        _template_for_name(entries, spec.step2_template_name, occurrence=spec.step2_template_occurrence),
    )


def classify_question_steps(
    lines: list[str],
    spec: QuestionSpec,
    entries: list[dict[str, str]] | None = None,
    *,
    mol_cache: dict[int, tuple[list[Chem.Mol], list[Chem.Mol]]] | None = None,
) -> tuple[dict[int, list[str]], dict[int, list[str]], SmirksTemplate, SmirksTemplate]:
    entries = entries or load_smirks_entries()
    s1_template, s2_template = templates_for_question(spec, entries)
    s1_hits = classify_reactions(lines, [s1_template], mol_cache=mol_cache)
    s2_hits = classify_reactions(lines, [s2_template], mol_cache=mol_cache)
    return s1_hits, s2_hits, s1_template, s2_template


def pick_spine2(rec2: ReactionRecord, spine1: str) -> str | None:
    """Best-effort main organic product from step 2 given the linked reactant."""
    products = organic_components(rec2.products)
    if not products:
        return None
    if spine1 in products:
        return spine1
    ha1 = heavy_atom_count(spine1)
    ranked = sorted(products, key=lambda s: (-heavy_atom_count(s), s))
    for p in ranked:
        if abs(heavy_atom_count(p) - ha1) <= 8:
            return p
    return ranked[0]


def enumerate_chains(
    records: dict[int, ReactionRecord],
    s1_indices: set[int],
    s2_indices: set[int],
    *,
    question_id: str,
    require_persistence: bool = True,
    max_chains: int = 100_000,
) -> list[Chain]:
    """Find chains via product→reactant SMILES index (not |S1|×|S2| brute force)."""
    persist_fn = PERSISTENCE_BY_QUESTION[question_id]
    chains: list[Chain] = []
    seen: set[tuple[int, int]] = set()

    s2_by_reactant: dict[str, list[int]] = defaultdict(list)
    for r2 in s2_indices:
        for smi in organic_components(records[r2].reactants):
            s2_by_reactant[smi].append(r2)

    for r1 in sorted(s1_indices):
        rec1 = records[r1]
        linked: dict[int, str] = {}
        for smi in organic_components(rec1.products):
            for r2 in s2_by_reactant.get(smi, ()):
                if r2 != r1 and r2 not in linked:
                    linked[r2] = smi
        for r2, spine1 in sorted(linked.items()):
            rec2 = records[r2]
            spine2 = pick_spine2(rec2, spine1)
            if spine2 is None:
                continue
            if require_persistence and not persist_fn(spine1, spine2):
                continue
            key = (r1, r2)
            if key in seen:
                continue
            seen.add(key)
            chains.append(Chain(reaction_indices=key, spine_smiles=(spine1, spine2)))
            if len(chains) >= max_chains:
                return chains
    return chains


def verify_chain(
    chain: tuple[int, ...],
    question_id: str,
    records: dict[int, ReactionRecord],
    lines_by_index: dict[int, str],
    *,
    s1_hits: dict[int, list[str]],
    s2_hits: dict[int, list[str]],
    require_persistence: bool = True,
) -> tuple[bool, str]:
    if len(chain) != CHAIN_LENGTH:
        return False, f"expected {CHAIN_LENGTH} reactions"
    if len(set(chain)) != CHAIN_LENGTH:
        return False, "duplicate indices"
    r1, r2 = chain
    if r1 not in s1_hits:
        return False, "step1 SMIRKS mismatch"
    if r2 not in s2_hits:
        return False, "step2 SMIRKS mismatch"
    rec1, rec2 = records[r1], records[r2]
    candidates = spine_candidates(rec1, rec2)
    spine1 = pick_spine(candidates)
    if spine1 is None:
        return False, "no canonical spine link"
    spine2 = pick_spine2(rec2, spine1)
    if spine2 is None:
        return False, "no step2 product"
    if require_persistence and not PERSISTENCE_BY_QUESTION[question_id](spine1, spine2):
        return False, "persistence check failed"
    _ = lines_by_index  # reserved for future template re-check
    return True, "ok"


def parse_chain_response(text: str) -> list[tuple[int, ...]]:
    """Parse model output into 2-reaction chains (one chain per line).

    Tolerates CodeAct-style ``ANSWER:`` prefixes and other non-numeric text on each line
    by extracting the first ``CHAIN_LENGTH`` integers per line (task11-style).
    """
    text = text.strip()
    if not text or text.replace(" ", "") == "-1":
        return []

    chains: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "-1":
            continue
        nums = re.findall(r"\d+", line)
        if len(nums) < CHAIN_LENGTH:
            continue
        if len(nums) % CHAIN_LENGTH == 0:
            chunks = [
                tuple(int(n) for n in nums[i : i + CHAIN_LENGTH])
                for i in range(0, len(nums), CHAIN_LENGTH)
            ]
        else:
            chunks = [tuple(int(n) for n in nums[:CHAIN_LENGTH])]
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


def score_chains(
    predicted: list[tuple[int, ...]],
    ground_truth: list[tuple[int, ...]],
) -> dict[str, float | int]:
    pred_set = {tuple(c) for c in predicted}
    gt_set = {tuple(c) for c in ground_truth}
    precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": len(pred_set),
        "ground_truth_count": len(gt_set),
        "true_positives": len(pred_set & gt_set),
    }


def ground_truth_chains_in_context(
    context_lines: list[str],
    hardcoded_chains: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Return hardcoded chains whose reaction indices all appear in context."""
    context_indices = {
        int(line.split(" ", 1)[0])
        for line in context_lines
        if line.strip() and " " in line and line.split(" ", 1)[0].isdigit()
    }
    return [chain for chain in hardcoded_chains if all(idx in context_indices for idx in chain)]


def build_rlm_question(spec: QuestionSpec) -> str:
    return question_prompt(spec)


def build_question(spec: QuestionSpec) -> str:
    return question_prompt(spec)


def question_prompt(spec: QuestionSpec) -> str:
    return f"""
There is a list of chemical reactions in SMILES format in the provided context, separated by newlines.
Each reaction is in one of these forms:
- "index reactants>reagents>products"
- "index reactants>>products"

Each side may contain multiple species separated by dots (.).
Reagents are in the middle field between the two > delimiters when present.

Task:
Find ALL valid {CHAIN_LENGTH}-reaction chains [r_0, r_1] in the context where:
- Step 1 (r_0): {spec.step1_summary}
- Step 2 (r_1): {spec.step2_summary}
- At least one canonical-SMILES product component of r_0 must be identical to at least one
  canonical-SMILES reactant component of r_1 (exact equality on dot-separated components).
- Do not reuse the same reaction index twice in one chain.
- Only use reactions present in the provided context.

Output format:
- Return each chain as a comma-separated list of exactly {CHAIN_LENGTH} reaction indices, one chain per line.
- Sort chains in lexicographic (ascending) order.
- No other text, quotes, labels, punctuation, JSON, or formatting.
- If no valid chain exists, return -1.
""".strip()


def smirks_documentation(spec: QuestionSpec, entries: list[dict[str, str]] | None = None) -> dict[str, object]:
    entries = entries or load_smirks_entries()
    s1, s2 = templates_for_question(spec, entries)
    return {
        "step1_template_name": spec.step1_template_name,
        "step1_template_occurrence": spec.step1_template_occurrence,
        "step1_smirks": s1.smirks,
        "step2_template_name": spec.step2_template_name,
        "step2_template_occurrence": spec.step2_template_occurrence,
        "step2_smirks": s2.smirks,
    }
