import argparse
import os
import random
import uuid

import wandb
from rlm import RLM
from rlm.codeact_helpers import (
    build_context_pipeline,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from task10b_hardcoded_ground_truth import (
    TASK10B_GROUND_TRUTH_DEFINITION,
    TASK10B_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION,
    TASK10B_POSITIVE_REACTIONS_BY_KEY,
    TASK10B_SKIPPED_REACTIONS,
    TASK10B_TOTAL_REACTIONS,
    TASK10B_VALID_REACTIONS,
)
from task10b_prompt_config import (
    REACTION_KEYS,
    TASK_DESCRIPTIONS,
    TASK_LABELS,
    build_task10b_question,
)

# os.environ["WANDB_MODE"] = "disabled"

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


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task10b",
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
        description="Run RLM task 10b mechanism-family evaluation."
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


def build_question(reaction_key: str) -> str:
    return build_task10b_question(reaction_key, allow_code=True)


def main(model_name: str, context_size: int) -> None:
    lines = load_lines(DATASET_PATH)
    gt_indices_by_reaction = TASK10B_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION

    for reaction_key in REACTION_KEYS:
        print(
            f"Ground truth [{reaction_key}] "
            f"count={TASK10B_POSITIVE_REACTIONS_BY_KEY[reaction_key]} "
            f"valid={TASK10B_VALID_REACTIONS} "
            f"skipped={TASK10B_SKIPPED_REACTIONS} "
            f"definition={TASK10B_GROUND_TRUTH_DEFINITION}"
        )

    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=random.Random(SEED),
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )

    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    run = wandb.init(
        project="RLMs-Task10b",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "min_selected_ground_truth": MIN_SELECTED_GROUND_TRUTH,
            "num_questions": len(REACTION_KEYS),
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_keys": list(REACTION_KEYS),
            "task_labels": TASK_LABELS,
            "task_descriptions": TASK_DESCRIPTIONS,
            "ground_truth_counts": TASK10B_POSITIVE_REACTIONS_BY_KEY,
            "ground_truth_total_reactions": TASK10B_TOTAL_REACTIONS,
            "ground_truth_valid_reactions": TASK10B_VALID_REACTIONS,
            "ground_truth_skipped_reactions": TASK10B_SKIPPED_REACTIONS,
            "ground_truth_definition": TASK10B_GROUND_TRUTH_DEFINITION,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    sample_results = []
    total_cost_usd = 0.0
    samples_with_cost = 0

    for sample_idx, reaction_key in enumerate(REACTION_KEYS):
        gt_indices = gt_indices_by_reaction[reaction_key]
        gt_set = set(gt_indices)
        question = build_question(reaction_key)

        sample_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=gt_set,
            query=reaction_key,
        )
        context_lines = [line for line in sample_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in context_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        ground_truth_in_context_set = gt_set & context_indices
        ground_truth_count = len(ground_truth_in_context_set)
        context_coverage = len(context_lines) / len(lines) if lines else 0.0
        print(
            f"[CONTEXT] task={reaction_key} requested_size={context_size} "
            f"actual_size={len(context_lines)} "
            f"ground_truth_in_context={ground_truth_count}/{len(gt_set)} "
            f"coverage={context_coverage:.4f}"
        )
        completion_kwargs = {"prompt": sample_context, "root_prompt": question}
        print(f"Question {sample_idx + 1}/{len(REACTION_KEYS)} task={reaction_key}")

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": sample_idx,
                "sample_count": len(REACTION_KEYS),
                "task": reaction_key,
                "ground_truth_definition": TASK10B_GROUND_TRUTH_DEFINITION,
            },
            tags=["run_rlms", "sample", f"task10b_{reaction_key}"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed_indices = parse_indices(response)
        pred_set = set(parsed_indices)
        precision, recall, f1 = precision_recall_f1(
            pred_set, ground_truth_in_context_set
        )
        predicted_count = len(pred_set)
        count_error = abs(predicted_count - ground_truth_count)
        count_exact = int(predicted_count == ground_truth_count)
        sample_cost_usd = completion.usage_summary.total_cost
        is_exact_match = pred_set == ground_truth_in_context_set

        print(f"Predicted [{reaction_key}] count: {predicted_count}")
        print(f"Ground truth [{reaction_key}] count: {ground_truth_count}")
        print(
            f"Metrics [{reaction_key}] -> "
            f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
            f"exact_match={is_exact_match} count_error={count_error} "
            f"count_exact={count_exact}"
        )

        for metric in iteration_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    f"sample/{sample_idx}/iteration_input_tokens": metric[
                        "iteration_input_tokens"
                    ],
                    f"sample/{sample_idx}/iteration_output_tokens": metric[
                        "iteration_output_tokens"
                    ],
                    f"sample/{sample_idx}/iteration_total_tokens": metric[
                        "iteration_total_tokens"
                    ],
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

        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        sample_results.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "is_exact_match": int(is_exact_match),
                "final_input_tokens": final_input_tokens,
                "final_output_tokens": final_output_tokens,
            }
        )

        wandb.log(
            {
                "sample_idx": sample_idx,
                f"sample/{sample_idx}/reaction_key": reaction_key,
                f"sample/{sample_idx}/final_total_input_tokens": final_input_tokens,
                f"sample/{sample_idx}/final_total_output_tokens": final_output_tokens,
                f"sample/{sample_idx}/final_total_tokens": final_total_tokens,
                f"sample/{sample_idx}/iterations": len(iteration_metrics),
                f"sample/{sample_idx}/precision": precision,
                f"sample/{sample_idx}/recall": recall,
                f"sample/{sample_idx}/f1": f1,
                f"sample/{sample_idx}/is_exact_match": int(is_exact_match),
                f"sample/{sample_idx}/predicted_count": predicted_count,
                f"sample/{sample_idx}/ground_truth_count": ground_truth_count,
                f"sample/{sample_idx}/ground_truth_full_count": len(gt_set),
                f"sample/{sample_idx}/count_error": count_error,
                f"sample/{sample_idx}/count_exact": count_exact,
                f"sample/{sample_idx}/completion_prompt_char_count": len(sample_context),
                f"sample/{sample_idx}/context_size": context_size,
                f"sample/{sample_idx}/context_coverage": context_coverage,
                f"sample/{sample_idx}/retrieved_line_count": len(context_lines),
                **(
                    {f"sample/{sample_idx}/final_total_cost_usd": sample_cost_usd}
                    if sample_cost_usd is not None
                    else {}
                ),
            }
        )

    total_samples = len(sample_results)
    exact_match_correct = sum(result["is_exact_match"] for result in sample_results)
    run.summary["exact_match_correct"] = exact_match_correct
    run.summary["total"] = total_samples
    run.summary["exact_match_accuracy"] = exact_match_correct / total_samples
    run.summary["macro_precision"] = (
        sum(result["precision"] for result in sample_results) / total_samples
    )
    run.summary["macro_recall"] = (
        sum(result["recall"] for result in sample_results) / total_samples
    )
    run.summary["macro_f1"] = (
        sum(result["f1"] for result in sample_results) / total_samples
    )
    run.summary["avg_total_input_tokens_per_sample"] = (
        sum(result["final_input_tokens"] for result in sample_results) / total_samples
    )
    run.summary["avg_total_output_tokens_per_sample"] = (
        sum(result["final_output_tokens"] for result in sample_results) / total_samples
    )
    for reaction_key in REACTION_KEYS:
        run.summary[f"ground_truth/{reaction_key}/count"] = (
            TASK10B_POSITIVE_REACTIONS_BY_KEY[reaction_key]
        )
    run.summary["ground_truth/definition"] = TASK10B_GROUND_TRUTH_DEFINITION
    run.summary["ground_truth/total_reactions"] = TASK10B_TOTAL_REACTIONS
    run.summary["ground_truth/valid_reactions"] = TASK10B_VALID_REACTIONS
    run.summary["ground_truth/skipped_reactions"] = TASK10B_SKIPPED_REACTIONS
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        context_size=args.context_size,
    )
