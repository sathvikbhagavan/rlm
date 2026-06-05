import argparse
import asyncio
import os
import random
import uuid

import wandb
from llama_index.core.workflow import Context
from llama_index.llms.openrouter import OpenRouter
from task3_hardcoded_ground_truth import (
    TASK3_HARDCODED_GROUND_TRUTH_INDICES,
    TASK3_THRESHOLDS,
)

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


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 900.0
THRESHOLDS = TASK3_THRESHOLDS
SEED = 42
CONTEXT_SIZE = 500
CONTEXT_PIPELINE_NAME = "random"
MAX_OUTPUT_TOKENS = 30_000
MAX_ITERATIONS = 8
REASONING_EFFORT = "high"
LLM_TIMEOUT_RETRIES = 2
LLM_TIMEOUT_RETRY_BACKOFF_S = 2.0
LLM_REQUEST_TIMEOUT_S = 300.0
# os.environ["WANDB_MODE"] = "disabled"


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="CodeAct-Task3",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def build_question(threshold: int) -> str:
    return f"""
    Above is a list of chemical reactions in SMILES format. Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Find all the indices of the reactions that satisfy:
      max over all pairs [rings(product_component) - rings(reactant_component)] == {threshold}

    Guidance:
    - You may use RDKit functions if needed (for example, Chem.MolFromSmiles and rdMolDescriptors.CalcNumRings).
    - Split each side by dot (.) to get components and compute ring count for each valid component.
    - For each reaction, compute ALL pairwise deltas:
      rings(product_component) - rings(reactant_component)
      and use the maximum of those deltas.
    - Ignore reagents (middle field).
    - For each side (reactants/products), ignore invalid or empty dot-separated molecules.
    - Skip a reaction only if reactant side or product side has no valid molecules left after filtering.
    - If you copy SMILES text into a Python string literal, handle backslashes safely:
      use a raw string (for example, r\"\"\"...\"\"\") or escape backslashes as "\\\\".

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If no matching reaction exists, report: -1
"""


def build_code_executor(lines: list[str]) -> object:
    return make_simple_code_executor(
        extra_globals={
            "np": __import__("numpy"),
            "rdkit": __import__("rdkit"),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAct task 3 evaluation.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for OpenRouter (default: {MODEL_NAME}).",
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


async def main(model_name: str, context_size: int) -> None:
    maybe_init_tracing()
    tracer = get_tracer("codeact-task3")
    lines = load_lines(DATASET_PATH)
    rng = random.Random(SEED)
    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=rng,
    )
    run_session_id = f"codeact-task3-{uuid.uuid4()}"

    run = wandb.init(
        project="CodeAct-Task3",
        config={
            "MODEL_NAME": model_name,
            "thresholds": THRESHOLDS,
            "dataset_path": DATASET_PATH,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "reasoning_effort": REASONING_EFFORT,
            "llm_timeout_retries": LLM_TIMEOUT_RETRIES,
            "llm_timeout_retry_backoff_s": LLM_TIMEOUT_RETRY_BACKOFF_S,
            "llm_request_timeout_s": LLM_REQUEST_TIMEOUT_S,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    ground_truth_indices_by_threshold: dict[int, set[int]] = {
        threshold: set(TASK3_HARDCODED_GROUND_TRUTH_INDICES.get(threshold, []))
        for threshold in THRESHOLDS
    }

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    total_input_tokens_sum = 0
    total_output_tokens_sum = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, threshold in enumerate(THRESHOLDS):
        print(f"Question {i + 1}/{len(THRESHOLDS)} for X={threshold}")
        question = build_question(threshold)
        ground_truth_index_set = ground_truth_indices_by_threshold[threshold]
        retrieved_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=ground_truth_index_set,
            query=str(threshold),
        )
        retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
        retrieved_indices = {
            int(line.split(" ", 1)[0])
            for line in retrieved_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        ground_truth_in_context_set = ground_truth_index_set & retrieved_indices
        gt_in_context_count = len(ground_truth_in_context_set)
        print(
            f"[CONTEXT] requested_size={context_size} actual_size={len(retrieved_lines)} "
            f"ground_truth_in_context={gt_in_context_count}/{len(ground_truth_index_set)}"
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
        executor = build_code_executor(lines=retrieved_lines)
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
        )
        ctx = Context(agent)

        with tracer.start_as_current_span(f"codeact_task3_sample_{i}") as sample_span:
            sample_span.set_attributes(
                {
                    "sample.index": i,
                    "sample.count": len(THRESHOLDS),
                    "sample.threshold": threshold,
                    "agent.name": "codeact",
                }
            )
            with using_tracing_attributes(
                session_id=run_session_id,
                metadata={
                    "sample_index": i,
                    "sample_count": len(THRESHOLDS),
                    "threshold": threshold,
                    "agent": "codeact",
                },
                tags=["codeact", "sample", "delta_rings"],
            ):
                # print(f"Prompt: {completion_prompt!r}")
                response = await run_agent_verbose(agent, ctx, completion_prompt)

        response_text = extract_response_text(response)
        # print(f"Raw response: {response_text!r}")
        # print("-" * 60)
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

        predicted_index_set = set(parse_indices(response_text))
        precision, recall, f1 = precision_recall_f1(predicted_index_set, ground_truth_in_context_set)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        if f1 < 1.0:
            print(f"Mismatch for X={threshold}")
            print(f"Predicted indices: {sorted(predicted_index_set)}")
            print(f"Ground truth indices (in context): {sorted(ground_truth_in_context_set)}")
            print(
                f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
            )
            print("--------------------------------")
        else:
            print(f"F1 is 1.0 for X={threshold}")

        for metric in llm_turn_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                    f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                    f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                    **(
                        {f"sample/{i}/iteration_cost_usd": metric["iteration_cost_usd"]}
                        if "iteration_cost_usd" in metric
                        else {}
                    ),
                }
            )

        final_input_tokens = sum(
            int(metric.get("iteration_input_tokens", 0)) for metric in llm_turn_metrics
        )
        final_output_tokens = sum(
            int(metric.get("iteration_output_tokens", 0)) for metric in llm_turn_metrics
        )
        final_total_tokens = sum(
            int(metric.get("iteration_total_tokens", 0)) for metric in llm_turn_metrics
        )
        total_input_tokens_sum += final_input_tokens
        total_output_tokens_sum += final_output_tokens
        final_cost = sum(float(metric.get("iteration_cost_usd", 0.0)) for metric in llm_turn_metrics)
        has_cost = any("iteration_cost_usd" in metric for metric in llm_turn_metrics)
        if has_cost:
            total_cost_usd += final_cost
            samples_with_cost += 1

        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/threshold_x": threshold,
                f"sample/{i}/final_total_input_tokens": final_input_tokens,
                f"sample/{i}/final_total_output_tokens": final_output_tokens,
                f"sample/{i}/final_total_tokens": final_total_tokens,
                f"sample/{i}/iterations": len(llm_turn_metrics),
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/ground_truth_count": len(ground_truth_in_context_set),
                f"sample/{i}/predicted_count": len(predicted_index_set),
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/context_size": context_size,
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_coverage": context_coverage,
                **({f"sample/{i}/final_total_cost_usd": final_cost} if has_cost else {}),
            }
        )
        wandb.log(
            {
                "running_precision": precision_sum / (i + 1),
                "running_recall": recall_sum / (i + 1),
                "running_f1": f1_sum / (i + 1),
                "running_context_coverage": context_coverage,
            }
        )

    total = len(THRESHOLDS)
    avg_precision = (precision_sum / total) if total else 0.0
    avg_recall = (recall_sum / total) if total else 0.0
    avg_f1 = (f1_sum / total) if total else 0.0
    avg_total_input_tokens = (total_input_tokens_sum / total) if total else 0.0
    avg_total_output_tokens = (total_output_tokens_sum / total) if total else 0.0
    print(f"Macro Precision: {avg_precision:.4f}")
    print(f"Macro Recall: {avg_recall:.4f}")
    print(f"Macro F1: {avg_f1:.4f}")
    print(f"Avg total input tokens/sample: {avg_total_input_tokens:.2f}")
    print(f"Avg total output tokens/sample: {avg_total_output_tokens:.2f}")

    run.summary["total"] = total
    run.summary["macro_precision"] = avg_precision
    run.summary["macro_recall"] = avg_recall
    run.summary["macro_f1"] = avg_f1
    run.summary["avg_total_input_tokens_per_sample"] = avg_total_input_tokens
    run.summary["avg_total_output_tokens_per_sample"] = avg_total_output_tokens
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(model_name=args.model_name, context_size=args.context_size))
