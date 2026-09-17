"""Prompt and context controls for the Task-16 prospective decomposition study.

This module is deliberately separate from the submitted Task-16 prompt.  The
legacy prompt is retained verbatim for reproducibility; new conditions use a
common prompt body and a stricter context that removes every exact occurrence
of the target on a reaction product side.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from rdkit import Chem
from task16_truncated_synthesis_graph import (
    PREFIX_LENGTH,
    RLM_CODE_GUIDANCE,
    RLM_DOCKER_MEMORY_GUIDANCE,
    ReactionRecord,
    TargetQuestionSpec,
)


class PromptCondition(StrEnum):
    LEGACY = "legacy"
    NAME_ONLY = "name_only"
    STRUCTURE_ONLY = "structure_only"
    STRUCTURE_PLUS_CLASS = "structure_plus_class"


INITIAL_QUESTION_IDS: tuple[str, ...] = (
    "pyrimidine_piperazine",
    "lactam_dipeptide",
    "benzamide_pyrazole",
)


@dataclass(frozen=True)
class FinalTransformation:
    summary: str
    accepted_terminal_indices: tuple[int, ...]
    supporting_target_product_indices: tuple[int, ...]
    provenance: str
    review_status: str = "requires-author-chemist-confirmation-before-scientific-launch"


# These labels were checked against the listed cleaned-USPTO records during the
# 2026-09-17 engineering audit.  They are intentionally narrow descriptions,
# not automatically selected Rxn-INSIGHT labels.  An author chemist must confirm
# them before STRUCTURE_PLUS_CLASS results are treated as scientific evidence.
FINAL_TRANSFORMATIONS: Mapping[str, FinalTransformation] = {
    "pyrimidine_piperazine": FinalTransformation(
        summary=(
            "aryl carbon-nitrogen coupling (N-arylation), replacing an aryl bromide "
            "with the cyclic secondary amine"
        ),
        accepted_terminal_indices=(11072, 11737),
        supporting_target_product_indices=(11072, 11737, 12317),
        provenance="cleaned USPTO target-producing records 11072, 11737, and 12317",
    ),
    "lactam_dipeptide": FinalTransformation(
        summary="Boc deprotection of the spirocyclic amine",
        accepted_terminal_indices=(91554, 91619),
        supporting_target_product_indices=(91554, 91619),
        provenance="cleaned USPTO target-producing records 91554 and 91619",
    ),
    "benzamide_pyrazole": FinalTransformation(
        summary=("palladium-catalyzed aryl bromide borylation to form the aryl boronic acid"),
        accepted_terminal_indices=(64756, 64761),
        supporting_target_product_indices=(64756, 64761),
        provenance="cleaned USPTO target-producing records 64756 and 64761",
    ),
}


def parse_prompt_condition(value: str | PromptCondition) -> PromptCondition:
    if isinstance(value, PromptCondition):
        return value
    try:
        return PromptCondition(value.strip().lower())
    except ValueError as exc:
        choices = ", ".join(x.value for x in PromptCondition)
        raise ValueError(f"Unknown Task-16 prompt condition {value!r}; choose {choices}") from exc


def selected_question_ids(value: str | None = None) -> tuple[str, ...]:
    raw = value if value is not None else os.environ.get("RXNHAYSTACK_TASK16_QUESTION_IDS", "")
    if not raw.strip():
        return tuple()
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(values) != len(set(values)):
        raise ValueError("RXNHAYSTACK_TASK16_QUESTION_IDS contains duplicates")
    return values


def transformation_for(question_id: str) -> FinalTransformation:
    try:
        return FINAL_TRANSFORMATIONS[question_id]
    except KeyError as exc:
        raise ValueError(
            f"No reviewed final-transformation candidate is recorded for {question_id!r}"
        ) from exc


def _target_block(spec: TargetQuestionSpec, condition: PromptCondition) -> str:
    if condition == PromptCondition.NAME_ONLY:
        return f"The target molecule is identified only by this name:\n{spec.target_name}"
    if condition == PromptCondition.STRUCTURE_ONLY:
        return f"The target molecule is specified by this canonical SMILES:\n{spec.target_smiles}"
    if condition == PromptCondition.STRUCTURE_PLUS_CLASS:
        transformation = transformation_for(spec.question_id)
        return (
            "The target molecule is specified by this canonical SMILES:\n"
            f"{spec.target_smiles}\n\n"
            "The withheld final reaction has this transformation class:\n"
            f"{transformation.summary}"
        )
    raise ValueError(
        "The legacy prompt is built by task16_truncated_synthesis_graph.build_question"
    )


def build_prospective_question(
    spec: TargetQuestionSpec,
    condition: str | PromptCondition,
    *,
    docker_memory_limit: str | None = None,
) -> str:
    """Build one of the three controlled prompts (never the legacy prompt)."""
    condition = parse_prompt_condition(condition)
    if condition == PromptCondition.LEGACY:
        raise ValueError("Use build_question() for the frozen legacy prompt")
    target_block = _target_block(spec, condition)
    question = f"""
    There is a list of chemical reactions in SMILES format in the provided context, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Each side may contain multiple species separated by dots (.).
    Ignore reagents (the middle field between the two > delimiters).

    Task:
    {target_block}

    Every reaction that produces the exact target has been withheld from the context.
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
    - Each prefix must be completable to the target specified above by appending exactly one
      withheld final reaction.

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
    guidance = RLM_CODE_GUIDANCE
    if docker_memory_limit:
        guidance = (
            f"{guidance}\n{RLM_DOCKER_MEMORY_GUIDANCE.format(memory_limit=docker_memory_limit)}"
        )
    return f"{question}\n\nGuidance:\n{guidance}"


def exact_target_product_indices(
    records: Mapping[int, ReactionRecord],
    target_smiles: str,
) -> frozenset[int]:
    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        raise ValueError(f"Invalid target SMILES: {target_smiles}")
    target = Chem.MolToSmiles(target_mol)
    return frozenset(idx for idx, record in records.items() if target in record.products)


def prediction_artifact_path(environ: Mapping[str, str] | None = None) -> Path | None:
    env = os.environ if environ is None else environ
    metrics_path = env.get("RXNHAYSTACK_METRICS_PATH")
    if not metrics_path:
        return None
    return Path(metrics_path).resolve().parent / "task16-predictions.json"


def write_prediction_artifact(
    rows: Iterable[Mapping[str, object]],
    *,
    prompt_condition: str,
    excluded_all_target_products: bool,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    path = prediction_artifact_path(environ)
    if path is None:
        return None
    env = os.environ if environ is None else environ
    payload = {
        "schema_version": 1,
        "run_id": env.get("RXNHAYSTACK_RUN_ID"),
        "task": "tier4/task16",
        "model": env.get("RXNHAYSTACK_MODEL"),
        "prompting_method": "rlm",
        "prompt_condition": prompt_condition,
        "excluded_all_target_products": excluded_all_target_products,
        "questions": list(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    return path


def transformation_manifest() -> dict[str, dict[str, object]]:
    return {key: asdict(value) for key, value in FINAL_TRANSFORMATIONS.items()}
