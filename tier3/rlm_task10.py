import argparse
import json
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass

import wandb
from rlm import RLM
from rlm.codeact_helpers import parse_indices, precision_recall_f1
from rlm.tracing import init_tracing, using_tracing_attributes

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
NEGATIVE_REACTIONS_PATH = "/home/bhagavan/rlms/rlm/tier3/negative_reactions.jsonl"
BACKEND = "openrouter"
MODEL_NAME = "x-ai/grok-4-fast"
ENABLE_TRACING = True
SEED = 42
NEGATIVES_PER_QUESTION = 10

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}

NEGATIVE_TYPE_LABELS: dict[str, str] = {
    "structurally_impossible": "Structurally impossible reactions",
    "conservation_violation": "Conservation-violating reactions",
    "product_swapping": "Product-swapped reactions",
    "handcrafted_wrong_reagent_or_product": "Hand-crafted wrong reagent/product reactions",
}

NEGATIVE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "structurally_impossible": (
        "Flag any reaction whose product-side molecule is invalid: SMILES cannot be parsed, "
        "sanitization fails, or chemistry is impossible (e.g., valence violations). "
        "Do not flag a reaction only because it is uncommon; it must be structurally invalid."
    ),
    "conservation_violation": (
        "Flag reactions where at least one element symbol present in products is absent from "
        "the combined reactants+reagents side. Compare element identity only (not atom counts), "
        "because stoichiometric imbalance is common in USPTO data."
    ),
    "product_swapping": (
        "Flag reactions whose reactants/reagents and products are semantically mismatched, "
        "as if the product side belongs to a different source reaction. The molecules may be "
        "valid individually, but the transformation is inconsistent as a pair."
    ),
    "handcrafted_wrong_reagent_or_product": (
        "Flag manually corrupted reactions such as irrelevant reagent injection, product "
        "replacement with an inappropriate reagent/reactant, or other edits that break "
        "reaction plausibility without necessarily breaking SMILES syntax."
    ),
}


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task10",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RLM task 10 with injected incorrect reaction equations."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for backend (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--negatives-per-question",
        type=int,
        default=NEGATIVES_PER_QUESTION,
        help=(
            "Number of injected negative reactions for each question "
            f"(default: {NEGATIVES_PER_QUESTION})."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DATASET_PATH,
        help=f"Path to valid reaction dataset (default: {DATASET_PATH}).",
    )
    parser.add_argument(
        "--negative-reactions-path",
        type=str,
        default=NEGATIVE_REACTIONS_PATH,
        help=f"Path to generated negative reactions JSONL (default: {NEGATIVE_REACTIONS_PATH}).",
    )
    return parser.parse_args()


def load_raw_reactions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


@dataclass
class NegativeRecord:
    source_index_1: int | None
    corrupted_reaction: str


def load_negatives_by_type(path: str) -> dict[str, list[NegativeRecord]]:
    grouped: dict[str, list[NegativeRecord]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            ctype = payload.get("corruption_type")
            source_index_1 = payload.get("source_index_1")
            rxn = payload.get("corrupted_reaction")
            if isinstance(ctype, str) and isinstance(rxn, str) and ctype in NEGATIVE_TYPE_LABELS:
                grouped[ctype].append(
                    NegativeRecord(
                        source_index_1=source_index_1 if isinstance(source_index_1, int) else None,
                        corrupted_reaction=rxn,
                    )
                )
    return grouped


def build_question(reaction_label: str, reaction_description: str) -> str:
    return f"""
    Context: You are given a list of chemical reactions in SMILES format, separated by newlines.
    Each reaction line is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Task:
    Return all indices that correspond to:
    - {reaction_label}

    Description:
    - {reaction_description}

    Guidance:
    - Use RdKit for all analysis and counting.
    - Only identify reactions matching the target corruption type above.
    - Ignore valid reactions.
    - The index to return is the integer at the start of each line in this context.
    - DO NOT assume/simulate output of the code. Wait for the code to get executed and only then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.

    Output format:
    - Return ONLY the matching reaction INDICES.
    - Format must be a comma-separated list of integers in ascending order (e.g., 3,8,21).
    - No other text, quotes, labels, punctuation, or formatting.
    - If no matching reaction exists, return -1.
    """


def build_context(
    dataset_reactions: list[str],
    negative_pool: list[NegativeRecord],
    negatives_per_question: int,
) -> tuple[str, list[int], int]:
    if len(negative_pool) < negatives_per_question:
        raise ValueError("Not enough negatives for requested negatives_per_question.")
    selected = negative_pool[:negatives_per_question]

    selected_indices = {
        r.source_index_1
        for r in selected
        if isinstance(r.source_index_1, int) and r.source_index_1 >= 0
    }
    all_detected_indices = {
        r.source_index_1
        for r in negative_pool
        if isinstance(r.source_index_1, int) and r.source_index_1 >= 0
    }
    excluded_indices = all_detected_indices - selected_indices

    selected_overrides = {
        r.source_index_1: r.corrupted_reaction
        for r in selected
        if isinstance(r.source_index_1, int) and r.source_index_1 >= 0
    }

    context_lines: list[str] = []
    for idx, base_reaction in enumerate(dataset_reactions):
        if idx in excluded_indices:
            continue
        reaction = selected_overrides.get(idx, base_reaction)
        context_lines.append(f"{idx} {reaction}")

    gt_indices = sorted(selected_indices)
    return "\n".join(context_lines), gt_indices, len(context_lines)


def main(
    model_name: str,
    negatives_per_question: int,
    dataset_path: str,
    negative_reactions_path: str,
) -> None:
    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    valid_reactions = load_raw_reactions(dataset_path)
    negatives_by_type = load_negatives_by_type(negative_reactions_path)
    reaction_keys = list(NEGATIVE_TYPE_LABELS.keys())

    for reaction_key in reaction_keys:
        available = len(negatives_by_type.get(reaction_key, []))
        if available < negatives_per_question:
            raise ValueError(
                f"Not enough negatives for {reaction_key}: "
                f"required={negatives_per_question}, available={available}"
            )

    run = wandb.init(
        project="RLMs-Task10",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": dataset_path,
            "negative_reactions_path": negative_reactions_path,
            "num_questions": len(reaction_keys),
            "valid_context_size": "full_dataset",
            "negatives_per_question": negatives_per_question,
            "seed": SEED,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Find injected incorrect reactions by corruption type.",
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    exact_match_count = 0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, reaction_key in enumerate(reaction_keys):
        reaction_label = NEGATIVE_TYPE_LABELS[reaction_key]
        reaction_description = NEGATIVE_TYPE_DESCRIPTIONS[reaction_key]
        question = build_question(
            reaction_label=reaction_label,
            reaction_description=reaction_description,
        )
        context, gt_indices, context_size = build_context(
            dataset_reactions=valid_reactions,
            negative_pool=negatives_by_type[reaction_key],
            negatives_per_question=negatives_per_question,
        )
        gt_set = set(gt_indices)

        completion_kwargs = {"prompt": context, "root_prompt": question}
        print(f"Question {i + 1}/{len(reaction_keys)} type={reaction_key}")
        print(
            f"Injected negatives in context: {len(gt_indices)} / {context_size} "
            f"(target={reaction_key})"
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(reaction_keys),
                "task": "injected_negative_detection",
                "corruption_type": reaction_key,
            },
            tags=["run_rlms", "sample", "task10_negative_injection"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed = parse_indices(response)
        pred_set = set(parsed)
        precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        is_exact_match = pred_set == gt_set
        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        print(f"Response [{reaction_key}]: {response}")
        print(f"Predicted [{reaction_key}] count: {len(parsed)}")
        print(f"Ground truth [{reaction_key}] count: {len(gt_indices)}")
        print(
            f"Metrics [{reaction_key}] -> precision={precision:.4f} "
            f"recall={recall:.4f} f1={f1:.4f} exact_match={is_exact_match}"
        )

        for metric in iteration_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                    f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                    f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                }
            )

        if iteration_metrics:
            last_metric = iteration_metrics[-1]
            wandb.log(
                {
                    "sample_idx": i,
                    f"sample/{i}/corruption_type": reaction_key,
                    f"sample/{i}/final_total_input_tokens": last_metric["total_input_tokens"],
                    f"sample/{i}/final_total_output_tokens": last_metric["total_output_tokens"],
                    f"sample/{i}/final_total_tokens": last_metric["total_tokens"],
                    f"sample/{i}/iterations": len(iteration_metrics),
                    f"sample/{i}/response_raw": response,
                    f"sample/{i}/response_parsed_indices": ",".join(str(x) for x in parsed),
                    f"sample/{i}/response_parsed_count": len(parsed),
                    f"sample/{i}/ground_truth_indices": ",".join(str(x) for x in gt_indices),
                    f"sample/{i}/ground_truth_count": len(gt_indices),
                    f"sample/{i}/precision": precision,
                    f"sample/{i}/recall": recall,
                    f"sample/{i}/f1": f1,
                    f"sample/{i}/is_exact_match": int(is_exact_match),
                    f"sample/{i}/completion_root_prompt": question,
                    f"sample/{i}/completion_prompt_char_count": len(context),
                    f"sample/{i}/context_line_count": context_size,
                    **(
                        {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                        if sample_cost_usd is not None
                        else {}
                    ),
                }
            )

    total = len(reaction_keys)
    exact_match_accuracy = (exact_match_count / total) if total else 0.0
    macro_precision = (macro_precision / total) if total else 0.0
    macro_recall = (macro_recall / total) if total else 0.0
    macro_f1 = (macro_f1 / total) if total else 0.0
    print(f"Exact match: {exact_match_count}/{total}")
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    run.summary["exact_match_correct"] = exact_match_count
    run.summary["total"] = total
    run.summary["exact_match_accuracy"] = exact_match_accuracy
    run.summary["macro_precision"] = macro_precision
    run.summary["macro_recall"] = macro_recall
    run.summary["macro_f1"] = macro_f1
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        negatives_per_question=args.negatives_per_question,
        dataset_path=args.dataset_path,
        negative_reactions_path=args.negative_reactions_path,
    )
