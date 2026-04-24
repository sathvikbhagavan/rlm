import argparse
import asyncio
import random
import os
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
    build_retriever,
    extract_response_text,
    load_lines,
    parse_indices,
)
from rlm.tracing import get_tracer, init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 900.0
NUM_QUESTIONS = 20
SEED = 42
CONTEXT_SIZE = 100
RETRIEVER_NAME = "random"
MAX_OUTPUT_TOKENS = 40_000
MAX_ITERATIONS = 8
REASONING_EFFORT = "medium"
# os.environ["WANDB_MODE"] = "disabled"


def extract_product(indexed_line: str) -> str:
    _, reaction_smiles = indexed_line.split(" ", 1)
    return reaction_smiles.split(">")[-1].strip()


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="CodeAct-Task1",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def build_question(product: str) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format loaded into a variable `lines`.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Find the index/indices of the reaction for the following PRODUCT
    (and not the reactants/reagents): {product}

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If the product is not found, report: -1
"""


def build_code_executor(lines: list[str]):
    return make_simple_code_executor(
        extra_locals={
            "lines": lines,
        },
        extra_globals={
            "np": __import__("numpy"),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAct task 1 evaluation.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--num-questions", type=int, default=NUM_QUESTIONS)
    parser.add_argument(
        "--context-size",
        type=int,
        default=CONTEXT_SIZE,
        help="Number of retrieved reactions to include in context; use -1 for all lines.",
    )
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH)
    return parser.parse_args()


async def main(
    model_name: str,
    num_questions: int,
    context_size: int,
    dataset_path: str,
) -> None:
    maybe_init_tracing()
    tracer = get_tracer("codeact-task1")
    lines = load_lines(dataset_path=dataset_path)
    rng = random.Random(SEED)
    retriever = build_retriever(name=RETRIEVER_NAME, lines=lines, rng=rng)
    sampled_indices = rng.sample(range(len(lines)), k=min(num_questions, len(lines)))
    run_session_id = f"codeact-task1-{uuid.uuid4()}"

    run = wandb.init(
        project="CodeAct-Task1",
        config={
            "MODEL_NAME": model_name,
            "NUM_QUESTIONS": num_questions,
            "dataset_path": dataset_path,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": SEED,
            "context_size": context_size,
            "retriever_name": RETRIEVER_NAME,
            "reasoning_effort": REASONING_EFFORT,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    correct = 0
    retrieval_hits = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, target_index in enumerate(sampled_indices):
        print(f"Question {i + 1}/{len(sampled_indices)}")
        target_product = extract_product(lines[target_index])
        question = build_question(target_product)
        retrieved_context = retriever.build_context(
            query=target_product,
            target_index=target_index,
            k=context_size,
        )
        context_has_ground_truth = str(target_index) in retrieved_context
        retrieval_hits += int(context_has_ground_truth)
        if not context_has_ground_truth:
            print(f"[WARNING] Ground truth missing from retrieved context for target_index={target_index}")
        retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
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
        )
        ctx = Context(agent)

        with tracer.start_as_current_span(f"codeact_task1_sample_{i}") as sample_span:
            sample_span.set_attributes(
                {
                    "sample.index": i,
                    "sample.count": len(sampled_indices),
                    "target.index": target_index,
                    "agent.name": "codeact",
                }
            )
            with using_tracing_attributes(
                session_id=run_session_id,
                metadata={
                    "sample_index": i,
                    "sample_count": len(sampled_indices),
                    "target_index": target_index,
                    "agent": "codeact",
                },
                tags=["codeact", "sample"],
            ):
                print(f"Prompt: {completion_prompt!r}")
                response = await run_agent_verbose(agent, ctx, completion_prompt)

        response_text = extract_response_text(response)
        print(f"Raw response: {response_text!r}")
        print("-" * 60)
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
        is_correct = target_index in parsed_indices
        if is_correct:
            correct += 1
        else:
            print(f"Error: {target_index} not in {parsed_indices}")
            print(f"Line in context: {lines[target_index]}")
            print(f"Product: {target_product}")
            print(f"Raw response: {response_text!r}")
            print("-" * 60)

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
        final_cost = sum(float(metric.get("iteration_cost_usd", 0.0)) for metric in llm_turn_metrics)
        has_cost = any("iteration_cost_usd" in metric for metric in llm_turn_metrics)
        if has_cost:
            total_cost_usd += final_cost
            samples_with_cost += 1

        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/final_total_input_tokens": final_input_tokens,
                f"sample/{i}/final_total_output_tokens": final_output_tokens,
                f"sample/{i}/final_total_tokens": final_total_tokens,
                f"sample/{i}/iterations": len(llm_turn_metrics),
                f"sample/{i}/is_correct": int(is_correct),
                f"sample/{i}/target_index": target_index,
                f"sample/{i}/target_product": target_product,
                f"sample/{i}/response_raw": response_text,
                f"sample/{i}/response_parsed": ",".join(str(x) for x in parsed_indices),
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/context_size": context_size,
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_coverage": context_coverage,
                f"sample/{i}/context_has_ground_truth": int(context_has_ground_truth),
                **({f"sample/{i}/final_total_cost_usd": final_cost} if has_cost else {}),
            }
        )
        wandb.log(
            {
                "running_accuracy": correct / (i + 1),
                "running_retrieval_hit_rate": retrieval_hits / (i + 1),
            }
        )

    total = len(sampled_indices)
    accuracy = (correct / total) if total else 0.0
    retrieval_hit_rate = (retrieval_hits / total) if total else 0.0
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Retrieval hit-rate (ground truth in context): {retrieval_hit_rate:.4f}")

    run.summary["correct"] = correct
    run.summary["total"] = total
    run.summary["accuracy"] = accuracy
    run.summary["retrieval_hits"] = retrieval_hits
    run.summary["retrieval_hit_rate"] = retrieval_hit_rate
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            model_name=args.model_name,
            num_questions=args.num_questions,
            context_size=args.context_size,
            dataset_path=args.dataset_path,
        )
    )
