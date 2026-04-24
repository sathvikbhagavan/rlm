import asyncio
import os
import random
import uuid

import wandb
from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter

from rlm.codeact_helpers import (
    build_retriever,
    extract_response_text,
    extract_usage_metrics,
    load_lines,
    parse_indices,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/workspace/datasets/reactionSmilesFigShareUSPTO2023.txt"
MODEL_NAME = "x-ai/grok-4.1-fast"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = False
NUM_QUESTIONS = 2
SEED = 42
CONTEXT_SIZE = 100
RETRIEVER_NAME = "random"
MAX_OUTPUT_TOKENS = 10_000
os.environ["WANDB_MODE"] = "disabled"


def extract_product(indexed_line: str) -> str:
    _, reaction_smiles = indexed_line.split(" ", 1)
    return reaction_smiles.split(">")[-1].strip()


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="LLM-Task1",
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
    There is a list of chemical reactions in SMILES format.
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


async def main() -> None:
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    rng = random.Random(SEED)
    retriever = build_retriever(name=RETRIEVER_NAME, lines=lines, rng=rng)
    sampled_indices = rng.sample(range(len(lines)), k=min(NUM_QUESTIONS, len(lines)))
    run_session_id = f"llm-task1-{uuid.uuid4()}"

    llm = OpenRouter(
        model=MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
        additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
    )

    run = wandb.init(
        project="LLM-Task1",
        config={
            "MODEL_NAME": MODEL_NAME,
            "NUM_QUESTIONS": NUM_QUESTIONS,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": CONTEXT_SIZE,
            "retriever_name": RETRIEVER_NAME,
            "mode": "llm_baseline_no_tools",
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
            k=CONTEXT_SIZE,
        )
        context_has_ground_truth = str(target_index) in retrieved_context
        retrieval_hits += int(context_has_ground_truth)

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

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(sampled_indices),
                "target_index": target_index,
                "agent": "llm_baseline",
            },
            tags=["llm-baseline", "sample"],
        ):
            response = await llm.achat([ChatMessage(role="user", content=completion_prompt)])

        response_text = extract_response_text(response)
        parsed_indices = parse_indices(response_text)
        is_correct = target_index in parsed_indices
        if is_correct:
            correct += 1

        usage_metrics = extract_usage_metrics(response)
        prompt_tokens = int(usage_metrics.get("prompt_tokens", 0))
        completion_tokens = int(usage_metrics.get("completion_tokens", 0))
        total_tokens = int(usage_metrics.get("total_tokens", 0))
        sample_cost = (
            float(usage_metrics["cost_usd"]) if "cost_usd" in usage_metrics else None
        )
        if total_tokens == 0:
            prompt_tokens = count_tokens([{"role": "user", "content": completion_prompt}], MODEL_NAME)
            completion_tokens = count_tokens(
                [{"role": "assistant", "content": response_text}],
                MODEL_NAME,
            )
            total_tokens = prompt_tokens + completion_tokens
        if sample_cost is not None:
            total_cost_usd += sample_cost
            samples_with_cost += 1

        wandb.log(
            {
                "sample_iteration": 1,
                f"sample/{i}/iteration_input_tokens": prompt_tokens,
                f"sample/{i}/iteration_output_tokens": completion_tokens,
                f"sample/{i}/iteration_total_tokens": total_tokens,
                **({f"sample/{i}/iteration_cost_usd": sample_cost} if sample_cost is not None else {}),
            }
        )
        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/final_total_input_tokens": prompt_tokens,
                f"sample/{i}/final_total_output_tokens": completion_tokens,
                f"sample/{i}/final_total_tokens": total_tokens,
                f"sample/{i}/iterations": 1,
                f"sample/{i}/is_correct": int(is_correct),
                f"sample/{i}/target_index": target_index,
                f"sample/{i}/target_product": target_product,
                f"sample/{i}/response_raw": response_text,
                f"sample/{i}/response_parsed": ",".join(str(x) for x in parsed_indices),
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/context_size": CONTEXT_SIZE,
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_coverage": context_coverage,
                f"sample/{i}/context_has_ground_truth": int(context_has_ground_truth),
                **({f"sample/{i}/final_total_cost_usd": sample_cost} if sample_cost is not None else {}),
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
    asyncio.run(main())
