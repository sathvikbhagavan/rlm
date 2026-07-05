"""Hardcoded protecting-group pair ground truth for tier4 task14."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from task14_protecting_group_graph import (
    DATASET_TOTAL_REACTIONS,
    GroundTruthPair,
    MAX_HEAVY_ATOMS,
    MIN_HEAVY_ATOMS,
    PROTECTING_GROUPS,
    ProtectingGroupSpec,
)

TASK14_TOTAL_REACTIONS = DATASET_TOTAL_REACTIONS
TASK14_MIN_SELECTED_GROUND_TRUTH = 2
TASK14_GROUND_TRUTH_DEFINITION = (
    "install/remove reaction pairs for a protecting group on the same canonical scaffold, "
    f"with install index < removal index, RDKit SMARTS PG detection, heavy-atom window "
    f"[{MIN_HEAVY_ATOMS}, {MAX_HEAVY_ATOMS}], only reactions present in context"
)


HARDCODED_PAIRS_JSON = Path(__file__).with_name("task14_pg_hardcoded_pairs.json")

HARDCODED_GT_PAIR_COUNTS: dict[str, int] = {
    'Boc_N': 28,
    'benzyl_O_N': 1,
}

FIXED_QUESTIONS: list[ProtectingGroupSpec] = list(PROTECTING_GROUPS)


@dataclass(frozen=True)
class Task14ContextSampling:
    selected_pairs: tuple[GroundTruthPair, ...]
    support_indices: frozenset[int]
    forced_count: int
    selected_pair_count: int


def tier3_forced_reaction_count(
    answer_count: int,
    context_size: int,
    *,
    dataset_size: int = TASK14_TOTAL_REACTIONS,
    min_selected_ground_truth: int = TASK14_MIN_SELECTED_GROUND_TRUTH,
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
    with HARDCODED_PAIRS_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pair_from_payload(payload: dict[str, object]) -> GroundTruthPair:
    return GroundTruthPair(
        install_index=int(payload["install_index"]),
        remove_index=int(payload["remove_index"]),
        pg_label=str(payload["pg_label"]),
        functional_group=str(payload["functional_group"]),
        scaffold_key=str(payload["scaffold_key"]),
        install_free_smiles=str(payload["install_free_smiles"]),
        install_protected_smiles=str(payload["install_protected_smiles"]),
        remove_protected_smiles=str(payload["remove_protected_smiles"]),
        remove_free_smiles=str(payload["remove_free_smiles"]),
    )


def hardcoded_pairs_for_label(pg_label: str) -> tuple[GroundTruthPair, ...]:
    payload = _load_mined_payload()[pg_label]
    return tuple(_pair_from_payload(pair) for pair in payload["pairs"])


def full_support_indices_for_question(spec: ProtectingGroupSpec) -> set[int]:
    support: set[int] = set()
    for pair in hardcoded_pairs_for_label(spec.label):
        support.add(pair.install_index)
        support.add(pair.remove_index)
    return support


def pairs_for_context_sampling(
    spec: ProtectingGroupSpec,
    context_size: int,
) -> Task14ContextSampling:
    all_pairs = hardcoded_pairs_for_label(spec.label)
    full_support = full_support_indices_for_question(spec)
    forced_count = tier3_forced_reaction_count(len(full_support), context_size)
    target_pair_count = min(
        max(1, math.ceil(forced_count / 2)),
        len(all_pairs),
    )

    selected_pairs: list[GroundTruthPair] = []
    support_acc: set[int] = set()
    for pair in all_pairs:
        if len(selected_pairs) >= target_pair_count:
            break
        candidate_support = support_acc | {pair.install_index, pair.remove_index}
        pipeline_forced = tier3_forced_reaction_count(len(candidate_support), context_size)
        if len(candidate_support) > pipeline_forced:
            continue
        selected_pairs.append(pair)
        support_acc = candidate_support

    if not selected_pairs and all_pairs:
        first = all_pairs[0]
        selected_pairs = [first]
        support_acc = {first.install_index, first.remove_index}

    selected = tuple(selected_pairs)
    return Task14ContextSampling(
        selected_pairs=selected,
        support_indices=frozenset(support_acc),
        forced_count=forced_count,
        selected_pair_count=len(selected),
    )


def full_dataset_pair_count(spec: ProtectingGroupSpec) -> int:
    return HARDCODED_GT_PAIR_COUNTS.get(spec.label, len(hardcoded_pairs_for_label(spec.label)))


def format_task14_pairs(pairs: list[GroundTruthPair] | list[tuple[int, int]], limit: int = 20) -> str:
    if pairs and isinstance(pairs[0], GroundTruthPair):
        shown = [f"{pair.install_index},{pair.remove_index}" for pair in pairs[:limit]]  # type: ignore[index]
    else:
        shown = [f"{a},{b}" for a, b in pairs[:limit]]  # type: ignore[index]
    if len(pairs) > limit:
        shown.append(f"... (+{len(pairs) - limit} more)")
    return " | ".join(shown)


def print_task14_startup_banner(*, max_pairs_per_group: int) -> None:
    for spec in FIXED_QUESTIONS:
        count = full_dataset_pair_count(spec)
        support = len(full_support_indices_for_question(spec))
        print(
            f"Ground truth [{spec.label}] full_dataset_pairs={count} "
            f"support_indices={support} max_pairs_per_group={max_pairs_per_group}"
        )


def print_task14_sample_context(
    *,
    sample_index: int,
    spec: ProtectingGroupSpec,
    gt_pairs: list[GroundTruthPair],
    sampling: Task14ContextSampling,
    full_support_indices: set[int],
    support_indices: set[int],
    support_in_context: int,
    context_size: int,
    context_line_count: int,
    context_coverage: float,
) -> None:
    print(
        f"\nQuestion {sample_index + 1}: pg_label={spec.label}, "
        f"gt_pairs={len(gt_pairs)}"
    )
    print(
        f"[CONTEXT] requested_size={context_size} actual_size={context_line_count} "
        f"selected_pairs={sampling.selected_pair_count}/{full_dataset_pair_count(spec)} "
        f"forced_count={sampling.forced_count} "
        f"support_in_context={support_in_context}/{len(support_indices)} "
        f"full_support={len(full_support_indices)} "
        f"in_context_pairs={len(gt_pairs)}/{sampling.selected_pair_count} "
        f"coverage={context_coverage:.4f}"
    )


def print_task14_sample_metrics(
    *,
    sample_index: int,
    spec: ProtectingGroupSpec,
    response: str,
    predicted: set[tuple[int, int]],
    gt_set: set[tuple[int, int]],
    precision: float,
    recall: float,
    f1: float,
    exact_set_match: bool,
) -> None:
    print(f"Response [sample={sample_index}]: {response[:500]}{'…' if len(response) > 500 else ''}")
    print(f"Predicted [{spec.label}]: {len(predicted)} pairs")
    print(f"Ground truth [{spec.label}]: {len(gt_set)} pairs")
    print(
        f"Metrics [sample={sample_index}] -> exact_set_match={int(exact_set_match)} "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
    )


def build_task14_wandb_sample_log(
    *,
    sample_index: int,
    spec: ProtectingGroupSpec,
    gt_pairs: list[GroundTruthPair],
    sampling: Task14ContextSampling,
    full_support_indices: set[int],
    support_indices: set[int],
    support_in_context: int,
    predicted: set[tuple[int, int]],
    precision: float,
    recall: float,
    f1: float,
    exact_set_match: float,
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
    gt_set = {(pair.install_index, pair.remove_index) for pair in gt_pairs}
    payload: dict[str, object] = {
        "sample_idx": sample_index,
        f"sample/{sample_index}/pg_label": spec.label,
        f"sample/{sample_index}/functional_group": spec.functional_group,
        f"sample/{sample_index}/ground_truth_count": len(gt_pairs),
        f"sample/{sample_index}/ground_truth_full_count": full_dataset_pair_count(spec),
        f"sample/{sample_index}/ground_truth_pairs": format_task14_pairs(gt_pairs),
        f"sample/{sample_index}/pred_pair_count": len(predicted),
        f"sample/{sample_index}/pred_pairs": format_task14_pairs(sorted(predicted)),
        f"sample/{sample_index}/selected_pair_count": sampling.selected_pair_count,
        f"sample/{sample_index}/forced_reaction_count": sampling.forced_count,
        f"sample/{sample_index}/support_indices_in_context": support_in_context,
        f"sample/{sample_index}/support_indices_selected_count": len(support_indices),
        f"sample/{sample_index}/support_indices_full_count": len(full_support_indices),
        f"sample/{sample_index}/response_char_count": len(response),
        f"sample/{sample_index}/precision": precision,
        f"sample/{sample_index}/recall": recall,
        f"sample/{sample_index}/f1": f1,
        f"sample/{sample_index}/exact_set_match": exact_set_match,
        f"sample/{sample_index}/completion_prompt_char_count": completion_prompt_char_count,
        f"sample/{sample_index}/context_size": context_size,
        f"sample/{sample_index}/context_coverage": context_coverage,
        f"sample/{sample_index}/retrieved_line_count": context_line_count,
        f"sample/{sample_index}/final_total_input_tokens": final_input_tokens,
        f"sample/{sample_index}/final_total_output_tokens": final_output_tokens,
        f"sample/{sample_index}/final_total_tokens": final_total_tokens,
        f"sample/{sample_index}/iterations": iterations,
        f"sample/{sample_index}/gt_pair_keys": format_task14_pairs(sorted(gt_set)),
    }
    if sample_cost_usd is not None:
        payload[f"sample/{sample_index}/final_total_cost_usd"] = sample_cost_usd
    return payload


def print_task14_run_summary(
    *,
    total: int,
    exact_match_count: int,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Protecting-group questions evaluated: {total}")
    print(f"Exact set match: {exact_match_count}/{total}")
    print(f"Exact set match accuracy: {exact_match_count / total if total else 0.0:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


def update_task14_run_summary(
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
    run.summary["questions_evaluated"] = total
    run.summary["exact_set_match_correct"] = exact_match_count
    run.summary["exact_set_match_accuracy"] = exact_match_count / total if total else 0.0
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
