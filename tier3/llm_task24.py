import argparse
import asyncio
import os
import random
import uuid

import wandb
from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter
from task24_hardcoded_ground_truth import (
    TASK24_GROUND_TRUTH_DEFINITION,
    TASK24_HARDCODED_GROUND_TRUTH_INDICES,
    TASK24_POSITIVE_REACTIONS,
    TASK24_SKIPPED_REACTIONS,
    TASK24_TOTAL_REACTIONS,
    TASK24_VALID_REACTIONS,
)

from rlm.codeact_helpers import (
    build_context_pipeline,
    extract_response_text,
    extract_usage_metrics,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"
MIN_SELECTED_GROUND_TRUTH = 5
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 30_000

REACTION_KEY = "e_double_bond_product"
TASK_LABEL = (
    "Reactions that produce a product containing an E-configured (trans) C=C double bond"
)
TASK_DESCRIPTION = (
    "A reaction matches when at least one product SMILES contains a C=C double bond "
    "explicitly annotated as E (trans) configuration. Unannotated double bonds are not counted."
)


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="LLM-Task24_tier3",
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
        description="Run LLM task 24 E-configured C=C double bond in product index evaluation."
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
    There is a list of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Return all reaction indices that match this reaction type:
      - {TASK_LABEL}

    Description:
      - {TASK_DESCRIPTION}

    Guidance:
    - Examine product molecules only; reactants and reagents are not considered.
    - E means CIP-based E geometry on a non-aromatic C=C double bond with explicit stereo annotation in the SMILES.
    - Do not count Z-configured C=C double bonds or C=C double bonds without explicit E/Z annotation.
    - Do not count aromatic ring bonds.
    - Count a reaction if at least one product contains an E-configured C=C double bond.
    - Multi-component product slots are separated by dots (.); check each component.
    - Skip malformed reactions.

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If no matching reaction exists, report: -1
    """


async def main(model_name: str, context_size: int) -> None:
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=random.Random(SEED),
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )
    run_session_id = f"llm-task24-{uuid.uuid4()}"
    question = build_question()
    full_gt_set = set(TASK24_HARDCODED_GROUND_TRUTH_INDICES)

    print(
        "Ground truth [e_double_bond_product] "
        f"count={TASK24_POSITIVE_REACTIONS} "
        f"valid={TASK24_VALID_REACTIONS} "
        f"skipped={TASK24_SKIPPED_REACTIONS} "
        f"definition={TASK24_GROUND_TRUTH_DEFINITION}"
    )

    llm = OpenRouter(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
    )

    run = wandb.init(
        project="LLM-Task24_tier3",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "min_selected_ground_truth": MIN_SELECTED_GROUND_TRUTH,
            "reasoning_effort": REASONING_EFFORT,
            "num_questions": 1,
            "task_label": TASK_LABEL,
            "task_description": TASK_DESCRIPTION,
            "ground_truth_count": TASK24_POSITIVE_REACTIONS,
            "ground_truth_total_reactions": TASK24_TOTAL_REACTIONS,
            "ground_truth_valid_reactions": TASK24_VALID_REACTIONS,
            "ground_truth_skipped_reactions": TASK24_SKIPPED_REACTIONS,
            "ground_truth_definition": TASK24_GROUND_TRUTH_DEFINITION,
            "mode": "llm_baseline_no_tools",
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    retrieved_context = context_pipeline.build_context(
        context_size=context_size,
        correct_indices=full_gt_set,
        query=REACTION_KEY,
    )
    retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
    retrieved_indices = {
        int(line.split(" ", 1)[0])
        for line in retrieved_lines
        if " " in line and line.split(" ", 1)[0].isdigit()
    }
    ground_truth_in_context_set = full_gt_set & retrieved_indices
    ground_truth_count = len(ground_truth_in_context_set)
    print(
        f"[CONTEXT] requested_size={context_size} actual_size={len(retrieved_lines)} "
        f"ground_truth_in_context={ground_truth_count}/{len(full_gt_set)}"
    )
    context_has_ground_truth = bool(ground_truth_in_context_set)
    context_coverage = len(retrieved_lines) / len(lines) if lines else 0.0
    completion_prompt = f"""
    You are given a subset of chemical reactions in SMILES format and a question.
    <context>
    {retrieved_context}
    </context>
    <question>
    {question}
    </question>
    """
    print("Question 1/1 task=e_double_bond_product")

    with using_tracing_attributes(
        session_id=run_session_id,
        metadata={
            "sample_index": 0,
            "sample_count": 1,
            "task": REACTION_KEY,
            "agent": "llm_baseline",
        },
        tags=["llm-baseline", "sample", "task24_e_double_bond_product"],
    ):
        response = await llm.achat([ChatMessage(role="user", content=completion_prompt)])

    response_text = extract_response_text(response)
    pred_set = set(parse_indices(response_text))
    precision, recall, f1 = precision_recall_f1(pred_set, ground_truth_in_context_set)
    predicted_count = len(pred_set)
    count_error = abs(predicted_count - ground_truth_count)
    count_exact = int(predicted_count == ground_truth_count)
    is_exact_match = pred_set == ground_truth_in_context_set

    print(f"Predicted [e_double_bond_product] count: {predicted_count}")
    print(f"Ground truth [e_double_bond_product] count: {ground_truth_count}")
    print(
        "Metrics [e_double_bond_product] -> "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"exact_match={is_exact_match} count_error={count_error} count_exact={count_exact}"
    )

    usage_metrics = extract_usage_metrics(response)
    prompt_tokens = int(usage_metrics.get("prompt_tokens", 0))
    completion_tokens = int(usage_metrics.get("completion_tokens", 0))
    total_tokens = int(usage_metrics.get("total_tokens", 0))
    sample_cost = float(usage_metrics["cost_usd"]) if "cost_usd" in usage_metrics else None
    if total_tokens == 0:
        prompt_tokens = count_tokens(
            [{"role": "user", "content": completion_prompt}],
            model_name,
        )
        completion_tokens = count_tokens(
            [{"role": "assistant", "content": response_text}],
            model_name,
        )
        total_tokens = prompt_tokens + completion_tokens

    wandb.log(
        {
            "sample_iteration": 1,
            "sample/0/iteration_input_tokens": prompt_tokens,
            "sample/0/iteration_output_tokens": completion_tokens,
            "sample/0/iteration_total_tokens": total_tokens,
            **({"sample/0/iteration_cost_usd": sample_cost} if sample_cost is not None else {}),
        }
    )
    wandb.log(
        {
            "sample_idx": 0,
            "sample/0/reaction_key": REACTION_KEY,
            "sample/0/final_total_input_tokens": prompt_tokens,
            "sample/0/final_total_output_tokens": completion_tokens,
            "sample/0/final_total_tokens": total_tokens,
            "sample/0/iterations": 1,
            "sample/0/precision": precision,
            "sample/0/recall": recall,
            "sample/0/f1": f1,
            "sample/0/is_exact_match": int(is_exact_match),
            "sample/0/predicted_count": predicted_count,
            "sample/0/ground_truth_count": ground_truth_count,
            "sample/0/ground_truth_full_count": len(full_gt_set),
            "sample/0/count_error": count_error,
            "sample/0/count_exact": count_exact,
            "sample/0/completion_prompt_char_count": len(completion_prompt),
            "sample/0/context_char_count": len(retrieved_context),
            "sample/0/retrieved_line_count": len(retrieved_lines),
            "sample/0/context_size": context_size,
            "sample/0/context_coverage": context_coverage,
            "sample/0/context_has_ground_truth": int(context_has_ground_truth),
            **({"sample/0/final_total_cost_usd": sample_cost} if sample_cost is not None else {}),
        }
    )

    run.summary["exact_match_correct"] = int(is_exact_match)
    run.summary["total"] = 1
    run.summary["exact_match_accuracy"] = float(is_exact_match)
    run.summary["macro_precision"] = precision
    run.summary["macro_recall"] = recall
    run.summary["macro_f1"] = f1
    run.summary["retrieval_hits"] = int(context_has_ground_truth)
    run.summary["retrieval_hit_rate"] = float(context_has_ground_truth)
    run.summary["avg_total_input_tokens_per_sample"] = prompt_tokens
    run.summary["avg_total_output_tokens_per_sample"] = completion_tokens
    run.summary["ground_truth/e_double_bond_product/count"] = ground_truth_count
    run.summary["ground_truth/e_double_bond_product/full_count"] = len(full_gt_set)
    run.summary["ground_truth/definition"] = TASK24_GROUND_TRUTH_DEFINITION
    run.summary["ground_truth/total_reactions"] = TASK24_TOTAL_REACTIONS
    run.summary["ground_truth/valid_reactions"] = TASK24_VALID_REACTIONS
    run.summary["ground_truth/skipped_reactions"] = TASK24_SKIPPED_REACTIONS
    run.summary["predicted_count"] = predicted_count
    run.summary["count_error"] = count_error
    run.summary["count_exact"] = count_exact
    run.summary["samples_with_cost"] = int(sample_cost is not None)
    if sample_cost is not None:
        run.summary["total_cost_usd"] = sample_cost
        run.summary["avg_cost_per_sample_usd"] = sample_cost
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(model_name=args.model_name, context_size=args.context_size))
