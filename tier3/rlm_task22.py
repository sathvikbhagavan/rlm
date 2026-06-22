import argparse
import random
import uuid

import wandb
from rlm import RLM
from rlm.codeact_helpers import build_context_pipeline, load_lines, parse_indices, precision_recall_f1
from rlm.tracing import init_tracing, using_tracing_attributes
from task22_hardcoded_ground_truth import (
    TASK22_GROUND_TRUTH_DEFINITION,
    TASK22_HARDCODED_GROUND_TRUTH_INDICES,
    TASK22_POSITIVE_REACTIONS,
    TASK22_SKIPPED_REACTIONS,
    TASK22_TOTAL_REACTIONS,
    TASK22_VALID_REACTIONS,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"
MIN_SELECTED_GROUND_TRUTH = 5

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}

REACTION_KEY = "hatu_pf6_or_t3p_reagent"
TASK_LABEL = "Reactions that use HATU+PF6 or T3P"
TASK_DESCRIPTION = (
    "A reaction matches when the reagent slot contains HATU+PF6 or T3P. "
    "Only the middle reagent slot counts; appearances in reactants or products alone "
    "do not match."
)
TASK_EVALUATION_GUIDANCE = """
    Matching rule:
    - Inspect only the reagent slot, i.e. the middle field in reactants>reagents>products.
    - Count a reaction if the reagent slot contains HATU+PF6 or T3P.
    - HATU+PF6 may appear as separate dot-delimited salt fragments in the reagent slot.
    - Do not count PF6 alone, HATU paired with a different counterion, or T3P outside the reagent slot.
    - Multi-component reagent slots are separated by dots (.); check each component.
"""


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task22_tier3",
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
        description="Run RLM task 22 HATU+PF6 or T3P reagent index evaluation."
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
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
    - Use RDKit for parsing molecules and comparing reagent identities.
    - Do not use string matching alone; salts may be represented as dot-separated components.
{TASK_EVALUATION_GUIDANCE}
    - Skip malformed reactions and unparseable reagent components.
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
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )
    question = build_question()
    gt_set = set(TASK22_HARDCODED_GROUND_TRUTH_INDICES)

    print(
        "Ground truth [hatu_pf6_or_t3p_reagent] "
        f"count={TASK22_POSITIVE_REACTIONS} "
        f"valid={TASK22_VALID_REACTIONS} "
        f"skipped={TASK22_SKIPPED_REACTIONS} "
        f"definition={TASK22_GROUND_TRUTH_DEFINITION}"
    )

    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    run = wandb.init(
        project="RLMs-Task22_tier3",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "min_selected_ground_truth": MIN_SELECTED_GROUND_TRUTH,
            "num_questions": 1,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_label": TASK_LABEL,
            "task_description": TASK_DESCRIPTION,
            "ground_truth_count": TASK22_POSITIVE_REACTIONS,
            "ground_truth_total_reactions": TASK22_TOTAL_REACTIONS,
            "ground_truth_valid_reactions": TASK22_VALID_REACTIONS,
            "ground_truth_skipped_reactions": TASK22_SKIPPED_REACTIONS,
            "ground_truth_definition": TASK22_GROUND_TRUTH_DEFINITION,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    sample_context = context_pipeline.build_context(
        context_size=context_size,
        correct_indices=gt_set,
        query=REACTION_KEY,
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

    print("Question 1/1 task=hatu_pf6_or_t3p_reagent")
    with using_tracing_attributes(
        session_id=run_session_id,
        metadata={
            "sample_index": 0,
            "sample_count": 1,
            "task": REACTION_KEY,
            "ground_truth_definition": TASK22_GROUND_TRUTH_DEFINITION,
        },
        tags=["run_rlms", "sample", "task22_hatu_pf6_or_t3p_reagent"],
    ):
        completion = rlm.completion(prompt=sample_context, root_prompt=question)
        response = completion.response

    iteration_metrics = rlm.get_last_iteration_metrics()
    pred_set = set(parse_indices(response))
    precision, recall, f1 = precision_recall_f1(pred_set, ground_truth_in_context_set)
    predicted_count = len(pred_set)
    count_error = abs(predicted_count - ground_truth_count)
    count_exact = int(predicted_count == ground_truth_count)
    sample_cost_usd = completion.usage_summary.total_cost
    is_exact_match = pred_set == ground_truth_in_context_set

    print(f"Predicted [hatu_pf6_or_t3p_reagent] count: {predicted_count}")
    print(f"Ground truth [hatu_pf6_or_t3p_reagent] count: {ground_truth_count}")
    print(
        "Metrics [hatu_pf6_or_t3p_reagent] -> "
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
            "sample/0/reaction_key": REACTION_KEY,
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
            "sample/0/count_exact": count_exact,
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
    run.summary["ground_truth/hatu_pf6_or_t3p_reagent/count"] = ground_truth_count
    run.summary["ground_truth/hatu_pf6_or_t3p_reagent/full_count"] = len(gt_set)
    run.summary["ground_truth/definition"] = TASK22_GROUND_TRUTH_DEFINITION
    run.summary["ground_truth/total_reactions"] = TASK22_TOTAL_REACTIONS
    run.summary["ground_truth/valid_reactions"] = TASK22_VALID_REACTIONS
    run.summary["ground_truth/skipped_reactions"] = TASK22_SKIPPED_REACTIONS
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
    main(model_name=args.model_name, context_size=args.context_size)
