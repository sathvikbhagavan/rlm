import argparse
import asyncio
import os
import random
import uuid

import wandb
from llama_index.core.workflow import Context
from llama_index.llms.openrouter import OpenRouter

from rlm.codeact_core import (
    CodeActAgent,
    INDEX_CODEACT_SYSTEM_PROMPT,
    INDEX_FORCE_LOOP_MESSAGE,
    INDEX_OBSERVATION_FOLLOWUP,
    make_simple_code_executor,
    run_agent_verbose,
)
from rlm.codeact_helpers import (
    build_context_pipeline,
    extract_response_text,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import get_tracer, init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens
from task15_hardcoded_ground_truth import (
    TASK15_HARDCODED_GROUND_TRUTH_INDICES,
    TASK15_POSITIVE_REACTIONS,
    TASK15_SKIPPED_REACTIONS,
    TASK15_TOTAL_REACTIONS,
    TASK15_VALID_REACTIONS,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 900.0
SEED = 42
CONTEXT_SIZE = 500
CONTEXT_PIPELINE_NAME = "random"
MIN_SELECTED_GROUND_TRUTH = 5
MAX_OUTPUT_TOKENS = 30_000
REASONING_EFFORT = "high"
MAX_ITERATIONS = 8
LLM_TIMEOUT_RETRIES = 2
LLM_TIMEOUT_RETRY_BACKOFF_S = 2.0
LLM_REQUEST_TIMEOUT_S = 300.0
CODE_EXECUTION_TIMEOUT_S = 300.0
# os.environ["WANDB_MODE"] = "disabled"

REACTION_KEY = "exactly_one_cc_formed"
TASK_LABEL = "Reactions that form exactly one C-C bond with no other carbon-carbon bond changes"
TASK_DESCRIPTION = (
    "A reaction forms exactly one C-C bond with no other carbon-carbon bond changes "
    "when exactly one carbon-carbon bond is formed and zero carbon-carbon bonds are "
    "broken. Bond type matters: single, double, triple, and aromatic C-C bonds are "
    "distinct types. Ignore the reagent field."
)


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="CodeAct-Task15_tier3",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAct task 15 evaluation.")
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
    - Use RDKit for all parsing and bond analysis.
    - Convert reactants and products to RDKit molecules; do not count by string matching.
    - Ignore reagents in the middle field.
    - For each molecule, iterate through bonds with RDKit.
    - A C-C bond is any bond where both endpoint atoms are carbon.
    - When comparing reactants and products, treat bonds of different order or aromaticity as different bond types.
    - Count how many C-C bonds of each type are formed and how many are broken.
    - A reaction matches when exactly one C-C bond is formed in total and zero C-C bonds are broken, with no other carbon-carbon bond changes.
    - Skip malformed reactions.
    - If you copy SMILES text into a Python string literal, handle backslashes safely:
      use a raw string (for example, r\"\"\"...\"\"\") or escape backslashes as "\\\\".

    Output format:
    - Return ONLY the matching reaction INDICES.
    - Format must be a comma-separated list of integers in ascending order (e.g., 3,8,21).
    - No other text, quotes, labels, punctuation, or formatting.
    - If no matching reaction exists, return -1.
    """


def build_code_executor():
    return make_simple_code_executor(
        extra_globals={
            "np": __import__("numpy"),
            "rdkit": __import__("rdkit"),
        },
    )


async def main(model_name: str, context_size: int) -> None:
    maybe_init_tracing()
    tracer = get_tracer("codeact-task15")
    lines = load_lines(DATASET_PATH)
    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=random.Random(SEED),
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )
    run_session_id = f"codeact-task15-{uuid.uuid4()}"
    question = build_question()
    full_gt_set = set(TASK15_HARDCODED_GROUND_TRUTH_INDICES)

    print(
        "Ground truth [exactly_one_cc_formed] "
        f"count={TASK15_POSITIVE_REACTIONS} "
        f"valid={TASK15_VALID_REACTIONS} "
        f"skipped={TASK15_SKIPPED_REACTIONS}"
    )

    run = wandb.init(
        project="CodeAct-Task15_tier3",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "min_selected_ground_truth": MIN_SELECTED_GROUND_TRUTH,
            "reasoning_effort": REASONING_EFFORT,
            "llm_timeout_retries": LLM_TIMEOUT_RETRIES,
            "llm_timeout_retry_backoff_s": LLM_TIMEOUT_RETRY_BACKOFF_S,
            "llm_request_timeout_s": LLM_REQUEST_TIMEOUT_S,
            "code_execution_timeout_s": CODE_EXECUTION_TIMEOUT_S,
            "task_label": TASK_LABEL,
            "task_description": TASK_DESCRIPTION,
            "ground_truth_count": TASK15_POSITIVE_REACTIONS,
            "ground_truth_total_reactions": TASK15_TOTAL_REACTIONS,
            "ground_truth_valid_reactions": TASK15_VALID_REACTIONS,
            "ground_truth_skipped_reactions": TASK15_SKIPPED_REACTIONS,
            "ground_truth_definition": (
                "bond-change signature: exactly one C-C bond formed and zero C-C bonds broken"
            ),
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
    print("Question 1/1 task=exactly_one_cc_formed")

    executor = build_code_executor()
    agent = CodeActAgent(
        code_execute_fn=executor.execute,
        llm=OpenRouter(
            model=model_name,
            api_key=OPENROUTER_API_KEY,
            max_tokens=MAX_OUTPUT_TOKENS,
            reasoning_effort=REASONING_EFFORT,
            additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
        ),
        system_prompt=INDEX_CODEACT_SYSTEM_PROMPT,
        max_iterations=MAX_ITERATIONS,
        force_loop_message=INDEX_FORCE_LOOP_MESSAGE,
        observation_followup=INDEX_OBSERVATION_FOLLOWUP,
        timeout=WORKFLOW_TIMEOUT_S,
        llm_timeout_retries=LLM_TIMEOUT_RETRIES,
        llm_timeout_retry_backoff_s=LLM_TIMEOUT_RETRY_BACKOFF_S,
        llm_request_timeout_s=LLM_REQUEST_TIMEOUT_S,
        code_execution_timeout_s=CODE_EXECUTION_TIMEOUT_S,
    )
    ctx = Context(agent)

    with tracer.start_as_current_span("codeact_task15_sample_0") as sample_span:
        sample_span.set_attributes(
            {
                "sample.index": 0,
                "sample.count": 1,
                "reaction.key": REACTION_KEY,
                "agent.name": "codeact",
            }
        )
        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": 0,
                "sample_count": 1,
                "reaction_key": REACTION_KEY,
                "agent": "codeact",
            },
            tags=["codeact", "sample", "task15_exactly_one_cc_formed"],
        ):
            response = await run_agent_verbose(agent, ctx, completion_prompt)

    response_text = extract_response_text(response)
    llm_turn_metrics = await ctx.store.get("llm_turn_metrics", default=[])
    if not llm_turn_metrics:
        estimated_prompt_tokens = count_tokens(
            [{"role": "user", "content": completion_prompt}],
            model_name,
        )
        estimated_completion_tokens = count_tokens(
            [{"role": "assistant", "content": response_text}],
            model_name,
        )
        llm_turn_metrics = [
            {
                "iteration": 1,
                "iteration_input_tokens": estimated_prompt_tokens,
                "iteration_output_tokens": estimated_completion_tokens,
                "iteration_total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
            }
        ]

    parsed_indices = parse_indices(response_text)
    pred_set = set(parsed_indices)
    precision, recall, f1 = precision_recall_f1(pred_set, ground_truth_in_context_set)
    predicted_count = len(pred_set)
    count_error = abs(predicted_count - ground_truth_count)
    count_exact = int(predicted_count == ground_truth_count)
    is_exact_match = pred_set == ground_truth_in_context_set

    print(f"Predicted [exactly_one_cc_formed] count: {predicted_count}")
    print(f"Ground truth [exactly_one_cc_formed] count: {ground_truth_count}")
    print(
        "Metrics [exactly_one_cc_formed] -> "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"exact_match={is_exact_match} count_error={count_error} count_exact={count_exact}"
    )

    for metric in llm_turn_metrics:
        wandb.log(
            {
                "sample_iteration": metric["iteration"],
                "sample/0/iteration_input_tokens": metric["iteration_input_tokens"],
                "sample/0/iteration_output_tokens": metric["iteration_output_tokens"],
                "sample/0/iteration_total_tokens": metric["iteration_total_tokens"],
                **(
                    {"sample/0/iteration_cost_usd": metric["iteration_cost_usd"]}
                    if "iteration_cost_usd" in metric
                    else {}
                ),
            }
        )

    final_input_tokens = sum(int(m.get("iteration_input_tokens", 0)) for m in llm_turn_metrics)
    final_output_tokens = sum(int(m.get("iteration_output_tokens", 0)) for m in llm_turn_metrics)
    final_total_tokens = sum(int(m.get("iteration_total_tokens", 0)) for m in llm_turn_metrics)
    final_cost = sum(float(m.get("iteration_cost_usd", 0.0)) for m in llm_turn_metrics)
    has_cost = any("iteration_cost_usd" in m for m in llm_turn_metrics)

    wandb.log(
        {
            "sample_idx": 0,
            "sample/0/reaction_key": REACTION_KEY,
            "sample/0/final_total_input_tokens": final_input_tokens,
            "sample/0/final_total_output_tokens": final_output_tokens,
            "sample/0/final_total_tokens": final_total_tokens,
            "sample/0/iterations": len(llm_turn_metrics),
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
            **({"sample/0/final_total_cost_usd": final_cost} if has_cost else {}),
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
    run.summary["predicted_count"] = predicted_count
    run.summary["count_error"] = count_error
    run.summary["count_exact"] = count_exact
    run.summary["ground_truth/exactly_one_cc_formed/count"] = ground_truth_count
    run.summary["ground_truth/exactly_one_cc_formed/full_count"] = len(full_gt_set)
    run.summary["ground_truth/total_reactions"] = TASK15_TOTAL_REACTIONS
    run.summary["ground_truth/valid_reactions"] = TASK15_VALID_REACTIONS
    run.summary["ground_truth/skipped_reactions"] = TASK15_SKIPPED_REACTIONS
    run.summary["samples_with_cost"] = int(has_cost)
    if has_cost:
        run.summary["total_cost_usd"] = final_cost
        run.summary["avg_cost_per_sample_usd"] = final_cost
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            model_name=args.model_name,
            context_size=args.context_size,
        )
    )
