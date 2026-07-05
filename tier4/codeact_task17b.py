import argparse
import asyncio
import os
import random
import uuid

from llama_index.core.workflow import Context
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
from rlm.codeact_core import (
    INDEX_CODEACT_SYSTEM_PROMPT,
    INDEX_FORCE_LOOP_MESSAGE,
    INDEX_OBSERVATION_FOLLOWUP,
    CodeActAgent,
    make_simple_code_executor,
    run_agent_verbose,
)
from rlm.codeact_helpers import build_context_pipeline, extract_response_text, load_lines
from rlm.tracing import get_tracer, init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 900.0
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"
MAX_CHAINS_PER_QUESTION = 0
MAX_OUTPUT_TOKENS = 30_000
REASONING_EFFORT = "high"
MAX_ITERATIONS = 8
LLM_TIMEOUT_RETRIES = 2
LLM_TIMEOUT_RETRY_BACKOFF_S = 2.0
LLM_REQUEST_TIMEOUT_S = 300.0
CODE_EXECUTION_TIMEOUT_S = 300.0


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="CodeAct-Task17b",
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
        description="Run CodeAct task 17b — SMIRKS sequential 2- or 3-reaction chains."
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    parser.add_argument("--max-chains-per-question", type=int, default=MAX_CHAINS_PER_QUESTION)
    return parser.parse_args()


def build_code_executor():
    return make_simple_code_executor(
        extra_globals={
            "np": __import__("numpy"),
            "rdkit": __import__("rdkit"),
        },
    )


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
    tracer = get_tracer("codeact-task17b")
    lines = load_lines(DATASET_PATH)
    evaluated_specs = [
        spec for spec in FIXED_QUESTIONS if full_dataset_chain_count(spec) > 0
    ]
    if not evaluated_specs:
        raise ValueError("No task17b questions have non-empty ground truth.")

    print_task17b_startup_banner(max_chains_per_question=max_chains_per_question)
    run_session_id = f"codeact-task17b-{uuid.uuid4()}"

    run = wandb.init(
        project="CodeAct-Task17b",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
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

        with tracer.start_as_current_span(f"codeact_task17b_sample_{i}") as sample_span:
            sample_span.set_attributes(
                {
                    "sample.index": i,
                    "sample.count": len(evaluated_specs),
                    "question_id": spec.question_id,
                    "chain_length": spec.chain_length,
                    "agent.name": "codeact",
                }
            )
            with using_tracing_attributes(
                session_id=run_session_id,
                metadata={
                    "sample_index": i,
                    "sample_count": len(evaluated_specs),
                    "question_id": spec.question_id,
                    "label": spec.label,
                    "chain_length": spec.chain_length,
                    "agent": "codeact",
                },
                tags=["codeact", "sample", "task17b_SMIRKS_SEQUENTIAL_CHAINS"],
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

        final_input_tokens = sum(int(m.get("iteration_input_tokens", 0)) for m in llm_turn_metrics)
        final_output_tokens = sum(int(m.get("iteration_output_tokens", 0)) for m in llm_turn_metrics)
        final_total_tokens = sum(int(m.get("iteration_total_tokens", 0)) for m in llm_turn_metrics)
        final_cost = sum(float(m.get("iteration_cost_usd", 0.0)) for m in llm_turn_metrics)
        has_cost = any("iteration_cost_usd" in m for m in llm_turn_metrics)
        if has_cost:
            total_cost_usd += final_cost
            samples_with_cost += 1
        total_input_tokens += final_input_tokens
        total_output_tokens += final_output_tokens

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
                final_input_tokens=final_input_tokens,
                final_output_tokens=final_output_tokens,
                final_total_tokens=final_total_tokens,
                iterations=len(llm_turn_metrics),
                sample_cost_usd=final_cost if has_cost else None,
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
