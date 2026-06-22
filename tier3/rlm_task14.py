import argparse
import os
import random
import uuid

import wandb
from rlm import RLM
from rlm.codeact_helpers import build_context_pipeline, parse_indices, precision_recall_f1
from rlm.tracing import init_tracing, using_tracing_attributes
from task14_hardcoded_ground_truth import (
    TASK14_HARDCODED_GROUND_TRUTH_INDICES,
    TASK14_POSITIVE_REACTIONS,
    TASK14_SKIPPED_REACTIONS,
    TASK14_TASK11_POSITIVE_REACTIONS,
    TASK14_TASK12_POSITIVE_REACTIONS,
    TASK14_TOTAL_REACTIONS,
    TASK14_VALID_REACTIONS,
)

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": False,
    "max_depth": 2,
}

TASK_LABEL = "Reactions that form at least one C-N bond and break at least one C-O bond"
TASK_DESCRIPTION = (
    "A reaction matches when both conditions hold in the same reaction. "
    "C-N bond formation: the total number of carbon-nitrogen connections in the "
    "products is greater than the total number of carbon-nitrogen connections in "
    "the reactants. Count connectivity only: single, double, triple, and aromatic "
    "C-N bonds each count as one C-N connection. "
    "C-O bond breaking: the reactants contain more instances of at least one "
    "carbon-oxygen bond type than the products. Bond type matters: single, double, "
    "triple, and aromatic C-O bonds are distinct types. Ignore the reagent field."
)


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task14_tier3",
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
        description="Run RLM task 14 C-N forming and C-O breaking index evaluation."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for backend (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=CONTEXT_SIZE,
        help=(
            "Number of retrieved reactions to include in context "
            f"(default: {CONTEXT_SIZE}; use -1 for all lines)."
        ),
    )
    return parser.parse_args()


def load_lines(dataset_path: str) -> list[str]:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        raw_lines = [line.strip() for line in handle if line.strip()]
    return [f"{i} {line}" for i, line in enumerate(raw_lines)]


def build_question() -> str:
    return f"""
    Context: You are given a large string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Task:
    Return all reaction indices that match this reaction type:
    - {TASK_LABEL}

    Description:
    - {TASK_DESCRIPTION}

    Guidance:
    - Use RDKit for all parsing and bond analysis.
    - Convert reactants and products to RDKit molecules; do not count by string matching.
    - Ignore reagents in the middle field.
    - For each molecule, iterate through bonds with RDKit.
    - C-N formation: a C-N connection is any bond where one endpoint atom is carbon and the other endpoint atom is nitrogen; count connectivity, not bond order; a reaction forms at least one C-N bond when product-side C-N connections minus reactant-side C-N connections is greater than zero.
    - C-O breaking: a C-O bond is any bond where one endpoint atom is carbon and the other endpoint atom is oxygen; when comparing sides, treat bonds of different order or aromaticity as different bond types; count how many times each C-O bond type appears on each side; a reaction breaks at least one C-O bond when at least one C-O bond type has a higher count among reactants than among products.
    - A reaction matches only when both the C-N formation and C-O breaking criteria are satisfied in the same reaction.
    - Skip malformed reactions.
    - DO NOT assume/simulate output of the code. Wait for the code to get executed and only then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If no matching reaction exists, return -1.
    """


def main(model_name: str, context_size: int) -> None:
    lines = load_lines(DATASET_PATH)
    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=random.Random(SEED),
    )
    question = build_question()
    gt_indices = TASK14_HARDCODED_GROUND_TRUTH_INDICES
    gt_set = set(gt_indices)

    print(
        "Ground truth [cn_forming_and_co_breaking] "
        f"count={TASK14_POSITIVE_REACTIONS} "
        f"valid={TASK14_VALID_REACTIONS} "
        f"skipped={TASK14_SKIPPED_REACTIONS} "
        f"(task11={TASK14_TASK11_POSITIVE_REACTIONS} "
        f"task12={TASK14_TASK12_POSITIVE_REACTIONS})"
    )

    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    run = wandb.init(
        project="RLMs-Task14_tier3",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "num_questions": 1,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_label": TASK_LABEL,
            "task_description": TASK_DESCRIPTION,
            "ground_truth_count": TASK14_POSITIVE_REACTIONS,
            "ground_truth_task11_positive_reactions": TASK14_TASK11_POSITIVE_REACTIONS,
            "ground_truth_task12_positive_reactions": TASK14_TASK12_POSITIVE_REACTIONS,
            "ground_truth_total_reactions": TASK14_TOTAL_REACTIONS,
            "ground_truth_valid_reactions": TASK14_VALID_REACTIONS,
            "ground_truth_skipped_reactions": TASK14_SKIPPED_REACTIONS,
            "ground_truth_definition": (
                "C-N forming and C-O breaking both hold in the same reaction"
            ),
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    sample_context = context_pipeline.build_context(
        context_size=context_size,
        correct_indices=gt_set,
        query="cn_forming_and_co_breaking",
    )
    context_lines = [line for line in sample_context.splitlines() if line.strip()]
    context_indices = {
        int(line.split(" ", 1)[0])
        for line in context_lines
        if " " in line and line.split(" ", 1)[0].isdigit()
    }
    ground_truth_in_context_set = gt_set & context_indices
    ground_truth_count = len(ground_truth_in_context_set)
    print(
        f"[CONTEXT] requested_size={context_size} actual_size={len(context_lines)} "
        f"ground_truth_in_context={ground_truth_count}/{len(gt_set)}"
    )
    completion_kwargs = {"prompt": sample_context, "root_prompt": question}
    print("Question 1/1 task=cn_forming_and_co_breaking")

    with using_tracing_attributes(
        session_id=run_session_id,
        metadata={
            "sample_index": 0,
            "sample_count": 1,
            "task": "cn_forming_and_co_breaking",
            "ground_truth_definition": "cn_forming_and_co_breaking",
        },
        tags=["run_rlms", "sample", "task14_cn_forming_and_co_breaking"],
    ):
        completion = rlm.completion(**completion_kwargs)
        response = completion.response

    iteration_metrics = rlm.get_last_iteration_metrics()
    parsed_indices = parse_indices(response)
    pred_set = set(parsed_indices)
    precision, recall, f1 = precision_recall_f1(pred_set, ground_truth_in_context_set)
    predicted_count = len(pred_set)
    count_error = abs(predicted_count - ground_truth_count)
    count_exact = int(predicted_count == ground_truth_count)
    sample_cost_usd = completion.usage_summary.total_cost
    is_exact_match = pred_set == ground_truth_in_context_set

    print(f"Predicted [cn_forming_and_co_breaking] count: {predicted_count}")
    print(f"Ground truth [cn_forming_and_co_breaking] count: {ground_truth_count}")
    print(
        "Metrics [cn_forming_and_co_breaking] -> "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"exact_match={is_exact_match} count_error={count_error} count_exact={count_exact}"
    )

    for metric in iteration_metrics:
        wandb.log(
            {
                "sample_iteration": metric["iteration"],
                "sample/0/iteration_input_tokens": metric["iteration_input_tokens"],
                "sample/0/iteration_output_tokens": metric["iteration_output_tokens"],
                "sample/0/iteration_total_tokens": metric["iteration_total_tokens"],
            }
        )

    final_input_tokens = 0
    final_output_tokens = 0
    final_total_tokens = 0
    if iteration_metrics:
        last_metric = iteration_metrics[-1]
        final_input_tokens = int(last_metric["total_input_tokens"])
        final_output_tokens = int(last_metric["total_output_tokens"])
        final_total_tokens = int(last_metric["total_tokens"])
        wandb.log(
            {
                "sample_idx": 0,
                "sample/0/reaction_key": "cn_forming_and_co_breaking",
                "sample/0/final_total_input_tokens": final_input_tokens,
                "sample/0/final_total_output_tokens": final_output_tokens,
                "sample/0/final_total_tokens": final_total_tokens,
                "sample/0/iterations": len(iteration_metrics),
                "sample/0/precision": precision,
                "sample/0/recall": recall,
                "sample/0/f1": f1,
                "sample/0/is_exact_match": int(is_exact_match),
                "sample/0/predicted_count": predicted_count,
                "sample/0/ground_truth_count": ground_truth_count,
                "sample/0/ground_truth_full_count": len(gt_set),
                "sample/0/count_error": count_error,
                "sample/0/completion_prompt_char_count": len(sample_context),
                "sample/0/context_size": context_size,
                **(
                    {"sample/0/final_total_cost_usd": sample_cost_usd}
                    if sample_cost_usd is not None
                    else {}
                ),
            }
        )

    run.summary["exact_match_correct"] = int(is_exact_match)
    run.summary["total"] = 1
    run.summary["exact_match_accuracy"] = float(is_exact_match)
    run.summary["macro_precision"] = precision
    run.summary["macro_recall"] = recall
    run.summary["macro_f1"] = f1
    run.summary["avg_total_input_tokens_per_sample"] = final_input_tokens
    run.summary["avg_total_output_tokens_per_sample"] = final_output_tokens
    run.summary["ground_truth/cn_forming_and_co_breaking/count"] = ground_truth_count
    run.summary["ground_truth/cn_forming_and_co_breaking/full_count"] = len(gt_set)
    run.summary["ground_truth/task11_positive_reactions"] = TASK14_TASK11_POSITIVE_REACTIONS
    run.summary["ground_truth/task12_positive_reactions"] = TASK14_TASK12_POSITIVE_REACTIONS
    run.summary["ground_truth/total_reactions"] = TASK14_TOTAL_REACTIONS
    run.summary["ground_truth/valid_reactions"] = TASK14_VALID_REACTIONS
    run.summary["ground_truth/skipped_reactions"] = TASK14_SKIPPED_REACTIONS
    run.summary["predicted_count"] = predicted_count
    run.summary["count_error"] = count_error
    run.summary["count_exact"] = count_exact
    run.summary["precision"] = precision
    run.summary["recall"] = recall
    run.summary["f1"] = f1
    run.summary["samples_with_cost"] = int(sample_cost_usd is not None)
    if sample_cost_usd is not None:
        run.summary["total_cost_usd"] = sample_cost_usd
        run.summary["avg_cost_per_sample_usd"] = sample_cost_usd
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        context_size=args.context_size,
    )
