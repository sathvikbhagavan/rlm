import argparse
import asyncio
import os
import random
import uuid

from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter
from task17b_smirks_sequential_graph import (
    build_rlm_question,
    parse_chain_response,
    precision_recall_f1,
)
from task17b_ground_truth import (
    FIXED_QUESTIONS,
    TASK17B_GROUND_TRUTH_DEFINITION,
    TASK17B_TOTAL_REACTIONS,
    build_task17b_wandb_sample_log,
    chains_for_context_sampling,
    full_dataset_chain_count,
    full_support_indices_for_question,
    ground_truth_chains_in_context,
    min_selected_ground_truth_for_spec,
    random_pool_excluded_indices,
    print_task17b_run_summary,
    print_task17b_sample_context,
    print_task17b_sample_metrics,
    print_task17b_startup_banner,
    update_task17b_run_summary,
)

import wandb
from rlm.codeact_helpers import (
    build_context_pipeline,
    extract_response_text,
    extract_usage_metrics,
    load_lines,
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
MAX_CHAINS_PER_QUESTION = 0
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 30_000


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="LLM-Task17b",
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
        description="Run LLM task 17b — SMIRKS sequential 2- or 3-reaction chains."
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    parser.add_argument("--max-chains-per-question", type=int, default=MAX_CHAINS_PER_QUESTION)
    return parser.parse_args()


def parse_predictions(response: str, chain_length: int) -> list[tuple[int, ...]]:
    return [
        chain
        for chain in parse_chain_response(response, chain_length=chain_length)
        if len(chain) == chain_length
    ]


async def main(model_name: str, context_size: int, max_chains_per_question: int) -> None:
    if max_chains_per_question < 0:
        raise ValueError("--max-chains-per-question must be non-negative.")

    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    evaluated_specs = [
        spec for spec in FIXED_QUESTIONS if full_dataset_chain_count(spec) > 0
    ]
    if not evaluated_specs:
        raise ValueError("No task17b questions have non-empty ground truth.")

    print_task17b_startup_banner(max_chains_per_question=max_chains_per_question)
    run_session_id = f"llm-task17b-{uuid.uuid4()}"

    llm = OpenRouter(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
    )

    run = wandb.init(
        project="LLM-Task17b",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "reasoning_effort": REASONING_EFFORT,
            "num_questions": len(evaluated_specs),
            "question_ids": [spec.question_id for spec in evaluated_specs],
            "chain_lengths": [spec.chain_length for spec in evaluated_specs],
            "max_chains_per_question": max_chains_per_question,
            "task_description": "SMIRKS sequential 2- or 3-reaction chain discovery.",
            "ground_truth_definition": TASK17B_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK17B_TOTAL_REACTIONS,
            "mode": "llm_baseline_no_tools",
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    exact_match_count = 0
    total_cost_usd = 0.0
    samples_with_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    samples_run = 0

    for i, spec in enumerate(evaluated_specs):
        full_support_indices = full_support_indices_for_question(spec)
        sampling = chains_for_context_sampling(spec, context_size)
        support_indices = set(sampling.support_indices)
        sampling_excluded = random_pool_excluded_indices(
            spec, support_indices, context_size=context_size
        )
        min_selected = min_selected_ground_truth_for_spec(spec)

        context_pipeline = build_context_pipeline(
            name=CONTEXT_PIPELINE_NAME,
            lines=lines,
            rng=random.Random(SEED + i),
            min_selected_ground_truth=min_selected,
        )
        retrieved_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=support_indices,
            query=f"task17b_{spec.question_id}",
            excluded_indices=sampling_excluded,
        )
        retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in retrieved_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        support_in_context = len(support_indices & context_indices)
        context_coverage = len(retrieved_lines) / len(lines) if lines else 0.0

        gt_chains = ground_truth_chains_in_context(
            retrieved_lines,
            spec,
            max_chains_per_question=max_chains_per_question,
        )
        if not gt_chains:
            raise ValueError(
                f"No ground-truth chains in context for question_id={spec.question_id}"
            )

        gt_set = {tuple(chain) for chain in gt_chains}
        prompt_question = build_rlm_question(spec)

        completion_prompt = f"""
        You are given a subset of chemical reactions in SMILES format and a question.
        <context>
        {retrieved_context}
        </context>
        <question>
        {prompt_question}
        </question>
        """

        print_task17b_sample_context(
            sample_index=i,
            spec=spec,
            gt_chains=gt_chains,
            sampling=sampling,
            full_support_indices=full_support_indices,
            support_indices=support_indices,
            support_in_context=support_in_context,
            context_size=context_size,
            context_line_count=len(retrieved_lines),
            context_coverage=context_coverage,
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(evaluated_specs),
                "question_id": spec.question_id,
                "label": spec.label,
                "chain_length": spec.chain_length,
                "agent": "llm_baseline",
            },
            tags=["llm-baseline", "sample", "task17b_SMIRKS_SEQUENTIAL_CHAINS"],
        ):
            response = await llm.achat([ChatMessage(role="user", content=completion_prompt)])

        response_text = extract_response_text(response)
        predicted_list = parse_predictions(response_text, spec.chain_length)
        predicted = {tuple(chain) for chain in predicted_list}
        precision, recall, f1 = precision_recall_f1(predicted=predicted, ground_truth=gt_set)
        is_exact_match = predicted == gt_set
        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        samples_run += 1

        print_task17b_sample_metrics(
            sample_index=i,
            spec=spec,
            response=response_text,
            predicted=predicted,
            gt_set=gt_set,
            precision=precision,
            recall=recall,
            f1=f1,
            exact_set_match=is_exact_match,
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
        total_input_tokens += prompt_tokens
        total_output_tokens += completion_tokens
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
            build_task17b_wandb_sample_log(
                sample_index=i,
                spec=spec,
                gt_chains=gt_chains,
                sampling=sampling,
                full_support_indices=full_support_indices,
                support_indices=support_indices,
                support_in_context=support_in_context,
                predicted=predicted,
                precision=precision,
                recall=recall,
                f1=f1,
                exact_set_match=float(is_exact_match),
                response=response_text,
                context_size=context_size,
                context_coverage=context_coverage,
                context_line_count=len(retrieved_lines),
                completion_prompt_char_count=len(completion_prompt),
                final_input_tokens=prompt_tokens,
                final_output_tokens=completion_tokens,
                final_total_tokens=total_tokens,
                iterations=1,
                sample_cost_usd=sample_cost,
            )
        )
        wandb.log(
            {
                "running_exact_set_match_accuracy": exact_match_count / samples_run,
                "running_macro_precision": macro_precision / samples_run,
                "running_macro_recall": macro_recall / samples_run,
                "running_macro_f1": macro_f1 / samples_run,
            }
        )

    total = samples_run
    macro_precision = macro_precision / total if total else 0.0
    macro_recall = macro_recall / total if total else 0.0
    macro_f1 = macro_f1 / total if total else 0.0

    print_task17b_run_summary(
        total=total,
        exact_match_count=exact_match_count,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
    )

    update_task17b_run_summary(
        run,
        total=total,
        exact_match_count=exact_match_count,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        samples_with_cost=samples_with_cost,
        total_cost_usd=total_cost_usd,
    )
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            model_name=args.model_name,
            context_size=args.context_size,
            max_chains_per_question=args.max_chains_per_question,
        )
    )
