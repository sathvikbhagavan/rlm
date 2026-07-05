"""Hardcoded truncated synthesis ground truth for tier4 task16."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rlm.codeact_helpers import build_context_pipeline

from task16_truncated_synthesis_graph import (
    DATASET_TOTAL_REACTIONS,
    FULL_CHAIN_LENGTH,
    MAX_HEAVY_ATOMS,
    MIN_HEAVY_ATOMS,
    PREFIX_LENGTH,
    ReactionRecord,
    TargetQuestionSpec,
    filter_context_lines,
    ground_truth_prefixes_in_context,
    prefixes_from_full_chains,
    parse_records_from_lines,
    records_subset,
    terminal_indices_from_full_chains,
)

TASK16_TOTAL_REACTIONS = DATASET_TOTAL_REACTIONS
TASK16_FORCED_PREFIX_COUNT = 2
TASK16_MIN_SELECTED_GROUND_TRUTH = PREFIX_LENGTH * TASK16_FORCED_PREFIX_COUNT
TASK16_GROUND_TRUTH_DEFINITION = (
    f"all {PREFIX_LENGTH}-reaction prefixes of length-{FULL_CHAIN_LENGTH} synthesis routes to a "
    "fixed target molecule via reaction-DAG edges (ascending indices, organic product-to-reactant "
    "links only); no target-as-reactant workups or salt/identity steps; final reaction withheld "
    "from context; at finite context sizes, random draws exclude all other mined-prefix "
    f"reactions besides up to {TASK16_FORCED_PREFIX_COUNT} forced support prefixes; only "
    "reactions present in context"
)

DEFAULT_TARGET_QUESTIONS: tuple[TargetQuestionSpec, ...] = (
    TargetQuestionSpec(
        question_id="indole_oxindole",
        target_smiles="CN(C)C(=O)C(=O)c1c[nH]c2cccc(OCc3ccccc3)c12",
        target_name="N,N-dimethyl dicarbonyl indole oxindole benzyl ether",
        label="indole oxindole",
        description=(
            "Oxindole-indole benzyl ether scaffold; the withheld final step is acylation "
            "to install the N,N-dimethyl dicarbonyl group on the target."
        ),
    ),
    TargetQuestionSpec(
        question_id="acrylamide_biaryl",
        target_smiles="C=CC(=O)Nc1cccc(-c2cc(-c3ccncc3)cc3cncnc23)c1",
        target_name="acrylamide-linked biaryl aminopyrimidine",
        label="acrylamide biaryl",
        description=(
            "Acrylamide-linked biaryl aminopyrimidine; the prefix is built through amide "
            "couplings and deprotections, and the withheld final step is acrylamide "
            "coupling to form the acrylamide linker."
        ),
    ),
    TargetQuestionSpec(
        question_id="pyrimidine_piperazine",
        target_smiles=(
            "COc1ncc(-c2cc(N3CC(CC(=O)O)C(F)(F)C3)c3nccn3n2)c(OC)n1"
        ),
        target_name="dimethoxypyrimidine with fluorinated piperazine carboxylate",
        label="pyrimidine piperazine",
        description=(
            "Dimethoxypyrimidine with a fluorinated piperazine carboxylic acid side chain; "
            "the withheld final step is amide coupling to attach the piperazine-acid "
            "substituent to the pyrimidine biaryl core."
        ),
    ),
    TargetQuestionSpec(
        question_id="phthalimide_glutarimide",
        target_smiles=(
            "O=C1CCC(N2C(=O)c3ccc(N4CCC(CN5CCS(=O)(=NCC6CCNCC6)CC5)CC4)cc3C2=O)C(=O)N1"
        ),
        target_name="glutarimide-phthalimide with sulfonimidoyl piperazine side chain",
        label="phthalimide glutarimide",
        description=(
            "Glutarimide-phthalimide core with a sulfonimidoyl piperazine side chain; the "
            "prefix builds the scaffold through amide coupling, piperazine substitution, "
            "and reductive amination, and the withheld final step is Cbz deprotection on "
            "the piperazine nitrogen."
        ),
    ),
    TargetQuestionSpec(
        question_id="pyridine_kinase_like",
        target_smiles=(
            "Cc1ccc(NC(=O)c2cc(C(F)(F)F)ccn2)cc1C1=Cc2cnc(Nc3nccs3)nc2N2CCN=C12"
        ),
        target_name="fused pyrimidine-pyridine nicotinamide with thiazolyl substituent",
        label="pyridine kinase-like",
        description=(
            "Fused pyrimidine-pyridine with thiazolyl and nicotinamide substituents; the "
            "prefix cyclizes and oxidizes the fused heterocycle and couples thiazolylamine, "
            "and the withheld final step is Suzuki biaryl coupling to a "
            "nicotinamide-substituted boronic ester."
        ),
    ),
    TargetQuestionSpec(
        question_id="lactam_dipeptide",
        target_smiles="NC(=O)[C@H](C[C@@H]1CCNC1=O)NC(=O)[C@@H]1CC2(CCN1)CC2",
        target_name="proline-lactam dipeptide with spirocyclic amino acid",
        label="lactam dipeptide",
        description=(
            "Proline-containing lactam dipeptide; the prefix couples amino-acid and lactam "
            "fragments via amide bond formation, and the withheld final step is Boc "
            "deprotection to give the free dipeptide."
        ),
    ),
    TargetQuestionSpec(
        question_id="uracil_piperazine",
        target_smiles="Cn1c(=O)n(C2CCC(=O)NC2=O)c2cccc(C3=CCN(C(=O)OC(C)(C)C)CC3)c21",
        target_name="uracil with piperazinone and Boc-piperazine substituent",
        label="uracil piperazine",
        description=(
            "Uracil heterocycle bearing a piperazinone and an aryl piperazine side chain; "
            "the prefix builds the uracil and piperazinone core, and the withheld final "
            "step is Suzuki coupling to install the aryl piperazine substituent."
        ),
    ),
    TargetQuestionSpec(
        question_id="benzamide_pyrazole",
        target_smiles=(
            "NC(=O)c1ccc(-c2cnn(C(Cc3ccn(C(F)F)n3)c3ccc(B(O)O)cn3)c2)cc1F"
        ),
        target_name="fluorobenzamide biaryl pyrazole with boronic acid handle",
        label="benzamide pyrazole",
        description=(
            "Fluorobenzamide linked to a pyrazole-pyrimidine via a biaryl bond; the prefix "
            "assembles the pyrazole-difluoropyrimidine fragment and forms the "
            "fluorobenzamide, and the withheld final step is Suzuki coupling at the "
            "pyridine boronic ester to complete the biaryl."
        ),
    ),
    TargetQuestionSpec(
        question_id="quinazoline_halide",
        target_smiles=(
            "C[C@@H]1CN(c2nc(Cl)nc3c(F)c(Br)c(Cl)cc23)[C@@H](C)CN1C(=O)OC(C)(C)C"
        ),
        target_name="Boc-protected diamine on a polyhalogenated quinazoline",
        label="quinazoline halide",
        description=(
            "Polyhalogenated quinazoline bearing a Boc-protected diamine substituent; the "
            "prefix chlorinates and functionalizes the quinazoline core, and the withheld "
            "final step is amine coupling to the chlorinated quinazoline."
        ),
    ),
    TargetQuestionSpec(
        question_id="piperidine_scaffold",
        target_smiles=(
            "CC(C)(C)[S@@](=O)N[C@@H]1c2scnc2CC12CCN(c1cnc(Sc3ccnc(C4CC4)c3Cl)c(N)n1)CC2"
        ),
        target_name="spirocyclic piperidine with thioether-linked aminopyrimidine",
        label="piperidine scaffold",
        description=(
            "Spirocyclic piperidine sulfonamide fused to a thioether-linked aminopyrimidine; "
            "the prefix builds the spirocyclic sulfonamide and aryl thioether fragments, "
            "and the withheld final step is piperidine N-arylation onto the "
            "chloroaminopyrimidine core."
        ),
    ),
)

HARDCODED_CHAINS_JSON = Path(__file__).with_name("task16_truncated_hardcoded_chains.json")

HARDCODED_GT_PREFIX_COUNTS: dict[str, int] = {
    'indole_oxindole': 52,
    'acrylamide_biaryl': 31,
    'pyrimidine_piperazine': 31,
    'phthalimide_glutarimide': 17,
    'pyridine_kinase_like': 12,
    'lactam_dipeptide': 10,
    'uracil_piperazine': 10,
    'benzamide_pyrazole': 12,
    'quinazoline_halide': 6,
    'piperidine_scaffold': 5,
}

HARDCODED_GT_FULL_CHAIN_COUNTS: dict[str, int] = {
    'indole_oxindole': 52,
    'acrylamide_biaryl': 31,
    'pyrimidine_piperazine': 62,
    'phthalimide_glutarimide': 17,
    'pyridine_kinase_like': 12,
    'lactam_dipeptide': 20,
    'uracil_piperazine': 10,
    'benzamide_pyrazole': 18,
    'quinazoline_halide': 8,
    'piperidine_scaffold': 10,
}

HARDCODED_GT_EXAMPLE: dict[str, tuple[int, ...]] = {
    'indole_oxindole': (9245, 26791, 81856, 81879),
    'acrylamide_biaryl': (868, 869, 3611, 3638),
    'pyrimidine_piperazine': (868, 869, 3611, 3638),
    'phthalimide_glutarimide': (9245, 26692, 61605, 68652),
    'pyridine_kinase_like': (40943, 40944, 40988, 41391),
    'lactam_dipeptide': (9742, 16689, 16690, 91553),
    'uracil_piperazine': (12346, 26761, 39775, 39776),
    'benzamide_pyrazole': (13443, 64738, 64754, 64755),
    'quinazoline_halide': (18145, 18146, 63271, 72074),
    'piperidine_scaffold': (2269, 89471, 89472, 89473),
}


@dataclass(frozen=True)
class TruncatedSynthesisQuestion:
    question_id: str
    target_smiles: str
    target_name: str
    label: str
    description: str

    @property
    def key(self) -> str:
        return self.question_id


def question_key(question_id: str) -> str:
    return question_id


FIXED_QUESTIONS: list[TruncatedSynthesisQuestion] = [
    TruncatedSynthesisQuestion(
        question_id=spec.question_id,
        target_smiles=spec.target_smiles,
        target_name=spec.target_name,
        label=spec.label,
        description=spec.description,
    )
    for spec in DEFAULT_TARGET_QUESTIONS
]


def target_spec_for_question(question: TruncatedSynthesisQuestion) -> TargetQuestionSpec:
    return TargetQuestionSpec(
        question_id=question.question_id,
        target_smiles=question.target_smiles,
        target_name=question.target_name,
        label=question.label,
        description=question.description,
    )


@dataclass(frozen=True)
class Task16ContextSampling:
    selected_prefixes: tuple[tuple[int, ...], ...]
    support_indices: frozenset[int]
    excluded_terminal_indices: frozenset[int]
    forced_count: int
    selected_prefix_count: int


@dataclass(frozen=True)
class Task16BuiltContext:
    sampling: Task16ContextSampling
    context_lines: list[str]
    context_text: str
    gt: object
    filters: object
    records: dict[int, ReactionRecord]
    support_in_context: int
    context_coverage: float
    context_attempt: int


def _prefixes_for_sampling(question: TruncatedSynthesisQuestion) -> tuple[tuple[int, ...], ...]:
    all_prefixes = hardcoded_prefixes_for_question(question.question_id)
    example = example_prefix_for_question(question)
    ordered: list[tuple[int, ...]] = [example]
    ordered.extend(prefix for prefix in all_prefixes if prefix != example)
    return tuple(ordered)


def chains_for_context_sampling(
    question: TruncatedSynthesisQuestion,
    context_size: int,
    *,
    preferred_prefix: tuple[int, ...] | None = None,
) -> Task16ContextSampling:
    prefix_candidates = _prefixes_for_sampling(question)
    if preferred_prefix is not None:
        prefix_candidates = (preferred_prefix,) + tuple(
            prefix for prefix in prefix_candidates if prefix != preferred_prefix
        )

    all_prefixes = hardcoded_prefixes_for_question(question.question_id)
    full_support = full_support_indices_for_question(question)
    excluded = terminal_indices_for_question(question.question_id)
    if context_size < 0:
        return Task16ContextSampling(
            selected_prefixes=all_prefixes,
            support_indices=frozenset(full_support),
            excluded_terminal_indices=excluded,
            forced_count=len(full_support),
            selected_prefix_count=len(all_prefixes),
        )

    forced_count = tier3_forced_reaction_count(len(full_support), context_size)
    target_prefix_count = min(
        max(1, math.ceil(forced_count / PREFIX_LENGTH)),
        len(all_prefixes),
        TASK16_FORCED_PREFIX_COUNT,
    )

    selected_prefixes: list[tuple[int, ...]] = []
    support_acc: set[int] = set()

    def try_add_prefix(prefix: tuple[int, ...], *, require_disjoint: bool) -> bool:
        nonlocal support_acc
        if prefix in selected_prefixes:
            return False
        if require_disjoint and set(prefix) & support_acc:
            return False
        candidate_support = support_acc | set(prefix)
        pipeline_forced = tier3_forced_reaction_count(len(candidate_support), context_size)
        if len(candidate_support) > pipeline_forced:
            return False
        selected_prefixes.append(prefix)
        support_acc = candidate_support
        return True

    for prefix in prefix_candidates:
        if len(selected_prefixes) >= target_prefix_count:
            break
        try_add_prefix(prefix, require_disjoint=True)

    if len(selected_prefixes) < target_prefix_count:
        for prefix in prefix_candidates:
            if len(selected_prefixes) >= target_prefix_count:
                break
            try_add_prefix(prefix, require_disjoint=False)

    if not selected_prefixes:
        fallback = preferred_prefix or prefix_candidates[0]
        selected_prefixes = [fallback]
        support_acc = set(fallback)

    return Task16ContextSampling(
        selected_prefixes=tuple(selected_prefixes),
        support_indices=frozenset(support_acc),
        excluded_terminal_indices=excluded,
        forced_count=forced_count,
        selected_prefix_count=len(selected_prefixes),
    )


def random_pool_excluded_indices(
    question: TruncatedSynthesisQuestion,
    support_indices: set[int] | frozenset[int],
) -> frozenset[int]:
    """Withheld terminals plus GT reactions from prefixes other than the forced support set."""
    terminals = terminal_indices_for_question(question.question_id)
    full_support = full_support_indices_for_question(question)
    other_gt = full_support - set(support_indices)
    return frozenset(terminals | other_gt)


def build_task16_eval_context(
    *,
    question: TruncatedSynthesisQuestion,
    lines: list[str],
    context_size: int,
    sample_index: int,
    seed: int,
    pipeline_name: str = "random",
    min_scored_prefixes: int = TASK16_FORCED_PREFIX_COUNT,
    max_attempts: int = 25,
    full_records: dict[int, ReactionRecord] | None = None,
) -> Task16BuiltContext:
    """Build context with the forced GT prefixes injected into the sample."""
    full_chains = hardcoded_full_chains_for_question(question.question_id)
    excluded = terminal_indices_for_question(question.question_id)

    if context_size < 0:
        context_lines = filter_context_lines(
            [line for line in lines if line.strip()],
            excluded,
        )
        if full_records is None:
            raise ValueError(
                "full_records is required when context_size < 0 "
                "(parse the dataset once and reuse across questions)."
            )
        context_records = records_subset(full_records, context_lines)
        example_prefix = example_prefix_for_question(question)
        sampling = chains_for_context_sampling(
            question,
            context_size,
            preferred_prefix=example_prefix,
        )
        gt, filters = ground_truth_prefixes_in_context(
            context_lines,
            question.question_id,
            question.target_smiles,
            full_chains=full_chains,
            excluded_terminal_indices=excluded,
            records=context_records,
        )
        scored = len(gt.accepted_reaction_indices) if gt else 0
        if scored < min_scored_prefixes:
            raise ValueError(
                f"No scorable prefixes for {question.question_id} at context_size=-1 "
                f"(scored={scored})"
            )
        support_indices = set(sampling.support_indices)
        support_in_context = len(support_indices & set(context_records.keys()))
        context_coverage = len(context_lines) / len(lines) if lines else 0.0
        return Task16BuiltContext(
            sampling=sampling,
            context_lines=context_lines,
            context_text="\n".join(context_lines),
            gt=gt,
            filters=filters,
            records=context_records,
            support_in_context=support_in_context,
            context_coverage=context_coverage,
            context_attempt=0,
        )

    example_prefix = example_prefix_for_question(question)
    sampling = chains_for_context_sampling(
        question,
        context_size,
        preferred_prefix=example_prefix,
    )
    support_indices = set(sampling.support_indices)
    required_prefixes = min(min_scored_prefixes, sampling.selected_prefix_count)
    sampling_excluded = random_pool_excluded_indices(question, support_indices)

    last_context_lines: list[str] = []
    last_gt = None
    last_filters = None
    last_records: dict[int, ReactionRecord] | None = None
    last_support_in_context = 0

    for attempt in range(max_attempts):
        attempt_seed = seed + sample_index + attempt * 997
        pipeline = build_context_pipeline(
            name=pipeline_name,
            lines=lines,
            rng=random.Random(attempt_seed),
            min_selected_ground_truth=max(
                TASK16_MIN_SELECTED_GROUND_TRUTH,
                len(support_indices),
            ),
        )
        sample_context = pipeline.build_context(
            context_size=context_size,
            correct_indices=support_indices,
            query=f"truncated_synthesis_{question.question_id}",
            excluded_indices=sampling_excluded,
        )
        context_lines = filter_context_lines(
            [line for line in sample_context.splitlines() if line.strip()],
            sampling_excluded,
        )
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in context_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        support_in_context = len(support_indices & context_indices)

        if full_records is not None:
            context_records = records_subset(full_records, context_lines)
        else:
            context_records = None

        gt, filters = ground_truth_prefixes_in_context(
            context_lines,
            question.question_id,
            question.target_smiles,
            full_chains=full_chains,
            excluded_terminal_indices=excluded,
            records=context_records,
            limit_to_prefixes=sampling.selected_prefixes,
        )
        scored = len(gt.accepted_reaction_indices) if gt else 0
        last_context_lines = context_lines
        last_gt = gt
        last_filters = filters
        last_records = context_records or (
            records_subset(full_records, context_lines)
            if full_records is not None
            else None
        )
        last_support_in_context = support_in_context

        if scored >= required_prefixes and support_in_context == len(support_indices):
            if last_records is None:
                last_records = parse_records_from_lines(context_lines)
            context_coverage = len(context_lines) / len(lines) if lines else 0.0
            return Task16BuiltContext(
                sampling=sampling,
                context_lines=context_lines,
                context_text="\n".join(context_lines),
                gt=gt,
                filters=filters,
                records=last_records,
                support_in_context=support_in_context,
                context_coverage=context_coverage,
                context_attempt=attempt,
            )

    raise ValueError(
        f"Could not build context with >={required_prefixes} scorable "
        f"{PREFIX_LENGTH}-reaction prefix(es) for {question.question_id} "
        f"(context_size={context_size}, selected_prefixes={sampling.selected_prefix_count}, "
        f"last_support_in_context={last_support_in_context}, last_scored="
        f"{len(last_gt.accepted_reaction_indices) if last_gt else 0})"
    )


def tier3_forced_reaction_count(
    answer_count: int,
    context_size: int,
    *,
    dataset_size: int = TASK16_TOTAL_REACTIONS,
    min_selected_ground_truth: int = TASK16_MIN_SELECTED_GROUND_TRUTH,
) -> int:
    if context_size < 0:
        top_k = dataset_size
    else:
        top_k = min(context_size, dataset_size)
    if top_k == 0 or answer_count == 0:
        return 0

    ratio_scaled_floor = int((answer_count / dataset_size) * top_k)
    half_cap = top_k // 2
    return min(
        answer_count,
        half_cap,
        max(min_selected_ground_truth, ratio_scaled_floor),
    )


@lru_cache(maxsize=1)
def _load_mined_payload() -> dict[str, dict[str, object]]:
    with HARDCODED_CHAINS_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hardcoded_full_chains_for_question(question_id: str) -> tuple[tuple[int, ...], ...]:
    payload = _load_mined_payload()[question_key(question_id)]
    return tuple(tuple(chain) for chain in payload["full_chains"])


def hardcoded_prefixes_for_question(question_id: str) -> tuple[tuple[int, ...], ...]:
    payload = _load_mined_payload()[question_key(question_id)]
    return tuple(tuple(chain) for chain in payload["prefix_chains"])


def terminal_indices_for_question(question_id: str) -> frozenset[int]:
    """All unique withheld 5th-step reaction indices for a question."""
    payload = _load_mined_payload()[question_key(question_id)]
    return frozenset(int(idx) for idx in payload["terminal_indices"])


def full_support_indices_for_question(question: TruncatedSynthesisQuestion) -> set[int]:
    return {
        idx
        for prefix in hardcoded_prefixes_for_question(question.question_id)
        for idx in prefix
    }


def example_prefix_for_question(question: TruncatedSynthesisQuestion) -> tuple[int, ...]:
    prefix = HARDCODED_GT_EXAMPLE.get(question.key)
    if prefix is None:
        raise KeyError(f"No hardcoded example prefix for {question.key}")
    if len(prefix) != PREFIX_LENGTH:
        raise ValueError(
            f"Expected prefix length {PREFIX_LENGTH}, got {len(prefix)} for {question.key}"
        )
    return prefix


def full_dataset_prefix_count(question: TruncatedSynthesisQuestion) -> int:
    return HARDCODED_GT_PREFIX_COUNTS[question.key]


def full_dataset_full_chain_count(question: TruncatedSynthesisQuestion) -> int:
    return HARDCODED_GT_FULL_CHAIN_COUNTS[question.key]


def print_task16_startup_banner() -> None:
    for question in FIXED_QUESTIONS:
        example = example_prefix_for_question(question)
        print(
            f"Ground truth [truncated synthesis {question.label}] "
            f"target={question.target_name} "
            f"full_dataset_prefixes={full_dataset_prefix_count(question)} "
            f"full_dataset_chains={full_dataset_full_chain_count(question)} "
            f"example_prefix={list(example)} "
            f"support_indices={len(full_support_indices_for_question(question))}"
        )


def print_task16_sample_context(
    *,
    sample_index: int,
    question: TruncatedSynthesisQuestion,
    gt,
    sampling: Task16ContextSampling,
    full_support_indices: set[int],
    support_indices: set[int],
    support_in_context: int,
    context_size: int,
    context_line_count: int,
    context_coverage: float,
    filters,
) -> None:
    print(
        f"\nQuestion {sample_index + 1}/{len(FIXED_QUESTIONS)}: "
        f"{question.label}, "
        f"gt_len={len(gt.reaction_indices)}, "
        f"gt_prefixes={len(gt.accepted_reaction_indices)}"
    )
    print(
        f"[CONTEXT] requested_size={context_size} actual_size={context_line_count} "
        f"selected_prefixes={sampling.selected_prefix_count}/{full_dataset_prefix_count(question)} "
        f"excluded_terminals={len(sampling.excluded_terminal_indices)} "
        f"forced_count={sampling.forced_count} "
        f"support_in_context={support_in_context}/{len(support_indices)} "
        f"full_support={len(full_support_indices)} "
        f"in_context_prefixes={len(gt.accepted_reaction_indices)}/{sampling.selected_prefix_count} "
        f"molecule_freq_cap={filters.molecule_freq_cap} "
        f"frequent_molecules={len(filters.frequent_molecules)} "
        f"coverage={context_coverage:.4f}"
    )


def format_task16_chains(chains: list[tuple[int, ...]], limit: int = 20) -> str:
    shown = [",".join(str(x) for x in chain) for chain in chains[:limit]]
    if len(chains) > limit:
        shown.append(f"... (+{len(chains) - limit} more)")
    return " | ".join(shown)


def print_task16_sample_metrics(
    *,
    sample_index: int,
    question: TruncatedSynthesisQuestion,
    response: str,
    parsed_chains: list[tuple[int, ...]],
    gt_chains: list[tuple[int, ...]],
    scores: dict[str, object],
) -> None:
    print(f"Response [sample={sample_index}]: {response[:500]}{'…' if len(response) > 500 else ''}")
    print(
        f"Predicted [{question.label}]: "
        f"{scores['valid_chain_count']}/{scores['parsed_chain_count']} valid prefixes"
    )
    print(f"Ground truth [{question.label}]: {len(gt_chains)} prefixes")
    print(
        f"Metrics [sample={sample_index}] -> precision={scores['precision']:.4f} "
        f"recall={scores['recall']:.4f} f1={scores['f1']:.4f} "
        f"exact_match={bool(scores['is_exact_match'])}"
    )


def build_task16_wandb_sample_log(
    *,
    sample_index: int,
    question: TruncatedSynthesisQuestion,
    gt,
    sampling: Task16ContextSampling,
    full_support_indices: set[int],
    support_indices: set[int],
    support_in_context: int,
    filters,
    parsed_chains: list[tuple[int, ...]],
    gt_chains: list[tuple[int, ...]],
    scores: dict[str, object],
    response: str,
    context_size: int,
    context_coverage: float,
    context_line_count: int,
    completion_prompt_char_count: int,
    final_input_tokens: int,
    final_output_tokens: int,
    final_total_tokens: int,
    iterations: int,
    sample_cost_usd: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "sample_idx": sample_index,
        f"sample/{sample_index}/prefix_length": PREFIX_LENGTH,
        f"sample/{sample_index}/full_chain_length": FULL_CHAIN_LENGTH,
        f"sample/{sample_index}/question_id": question.question_id,
        f"sample/{sample_index}/target_smiles": question.target_smiles,
        f"sample/{sample_index}/ground_truth_count": len(gt_chains),
        f"sample/{sample_index}/ground_truth_full_count": full_dataset_prefix_count(question),
        f"sample/{sample_index}/ground_truth_chains": format_task16_chains(gt_chains),
        f"sample/{sample_index}/molecule_freq_cap": filters.molecule_freq_cap,
        f"sample/{sample_index}/frequent_molecule_count": len(filters.frequent_molecules),
        f"sample/{sample_index}/response_parsed_count": scores["parsed_chain_count"],
        f"sample/{sample_index}/response_valid_count": scores["valid_chain_count"],
        f"sample/{sample_index}/response_parsed_chains": format_task16_chains(parsed_chains),
        f"sample/{sample_index}/selected_prefix_count": sampling.selected_prefix_count,
        f"sample/{sample_index}/excluded_terminal_count": len(sampling.excluded_terminal_indices),
        f"sample/{sample_index}/forced_reaction_count": sampling.forced_count,
        f"sample/{sample_index}/support_indices_in_context": support_in_context,
        f"sample/{sample_index}/support_indices_selected_count": len(support_indices),
        f"sample/{sample_index}/support_indices_full_count": len(full_support_indices),
        f"sample/{sample_index}/response_char_count": len(response),
        f"sample/{sample_index}/validity_reason": scores["validity_reason"],
        f"sample/{sample_index}/precision": scores["precision"],
        f"sample/{sample_index}/recall": scores["recall"],
        f"sample/{sample_index}/f1": scores["f1"],
        f"sample/{sample_index}/is_exact_match": scores["is_exact_match"],
        f"sample/{sample_index}/invalid_chain_count": scores["invalid_chain_count"],
        f"sample/{sample_index}/completion_prompt_char_count": completion_prompt_char_count,
        f"sample/{sample_index}/context_size": context_size,
        f"sample/{sample_index}/context_coverage": context_coverage,
        f"sample/{sample_index}/retrieved_line_count": context_line_count,
        f"sample/{sample_index}/final_total_input_tokens": final_input_tokens,
        f"sample/{sample_index}/final_total_output_tokens": final_output_tokens,
        f"sample/{sample_index}/final_total_tokens": final_total_tokens,
        f"sample/{sample_index}/iterations": iterations,
    }
    if sample_cost_usd is not None:
        payload[f"sample/{sample_index}/final_total_cost_usd"] = sample_cost_usd
    return payload


def print_task16_run_summary(
    *,
    total: int,
    exact_match_count: int,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Queries evaluated: {total}")
    print(f"Exact match: {exact_match_count}/{total}")
    print(f"Exact match accuracy: {exact_match_count / total if total else 0.0:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


def update_task16_run_summary(
    run,
    *,
    total: int,
    exact_match_count: int,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
    total_input_tokens: int,
    total_output_tokens: int,
    samples_with_cost: int,
    total_cost_usd: float,
) -> None:
    run.summary["queries_evaluated"] = total
    run.summary["exact_match_correct"] = exact_match_count
    run.summary["exact_match_accuracy"] = exact_match_count / total if total else 0.0
    run.summary["macro_precision"] = macro_precision
    run.summary["macro_recall"] = macro_recall
    run.summary["macro_f1"] = macro_f1
    run.summary["avg_total_input_tokens_per_sample"] = (
        total_input_tokens / total if total else 0.0
    )
    run.summary["avg_total_output_tokens_per_sample"] = (
        total_output_tokens / total if total else 0.0
    )
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
