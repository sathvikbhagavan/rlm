import argparse
import asyncio
import os
import random
import uuid

from llama_index.core.workflow import Context
from llama_index.llms.openrouter import OpenRouter
from task15_ring_chain_graph import (
    MAX_HEAVY_ATOMS,
    MIN_HEAVY_ATOMS,
    PATH_LENGTH,
    build_question,
    ground_truth_ring_path_in_context,
    parse_records_from_lines,
    parse_response,
    score_prediction,
)
from task15_ring_chain_ground_truth import (
    FIXED_QUESTIONS,
    TASK15_GROUND_TRUTH_DEFINITION,
    TASK15_MIN_SELECTED_GROUND_TRUTH,
    TASK15_TOTAL_REACTIONS,
    build_task15_wandb_sample_log,
    chains_for_context_sampling,
    full_dataset_chain_count,
    full_support_indices_for_question,
    print_task15_run_summary,
    print_task15_sample_context,
    print_task15_sample_metrics,
    print_task15_startup_banner,
    ring_spec_for_question,
    update_task15_run_summary,
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

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 900.0
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"
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
        project_name="CodeAct-Task15",
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
        description="Run CodeAct task 15 — ring-construction chains from acyclic precursors."
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
    evaluated_questions = [
        question for question in FIXED_QUESTIONS if full_dataset_chain_count(question) > 0
    ]
    if not evaluated_questions:
        raise ValueError("No ring-system questions have non-empty ground truth.")

    print_task15_startup_banner()
    run_session_id = f"codeact-task15-{uuid.uuid4()}"

    run = wandb.init(
        project="CodeAct-Task15",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "reasoning_effort": REASONING_EFFORT,
            "llm_timeout_retries": LLM_TIMEOUT_RETRIES,
            "llm_timeout_retry_backoff_s": LLM_TIMEOUT_RETRY_BACKOFF_S,
            "llm_request_timeout_s": LLM_REQUEST_TIMEOUT_S,
            "code_execution_timeout_s": CODE_EXECUTION_TIMEOUT_S,
            "num_questions": len(evaluated_questions),
            "ring_questions": [q.ring_system for q in evaluated_questions],
            "path_length": PATH_LENGTH,
            "min_selected_ground_truth": TASK15_MIN_SELECTED_GROUND_TRUTH,
            "min_heavy_atoms": MIN_HEAVY_ATOMS,
            "max_heavy_atoms": MAX_HEAVY_ATOMS,
            "task_description": "Ring-construction chains from acyclic precursors via RDKit SMARTS.",
            "ground_truth_definition": TASK15_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK15_TOTAL_REACTIONS,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    macro_accuracy = 0.0
    macro_valid_path = 0.0
    macro_objective_length = 0.0
    macro_reaction_f1 = 0.0
    index_match_count = 0
    total_cost_usd = 0.0
    samples_with_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    samples_run = 0

    for i, question in enumerate(evaluated_questions):
        spec = ring_spec_for_question(question)
        full_support_indices = full_support_indices_for_question(question)
        sampling = chains_for_context_sampling(question, context_size)
        support_indices = set(sampling.support_indices)

        context_pipeline = build_context_pipeline(
            name=CONTEXT_PIPELINE_NAME,
            lines=lines,
            rng=random.Random(SEED + i),
            min_selected_ground_truth=max(
                TASK15_MIN_SELECTED_GROUND_TRUTH,
                len(support_indices),
            ),
        )
        retrieved_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=support_indices,
            query=f"ring_chain_{question.ring_system}",
        )
        retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in retrieved_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        support_in_context = len(support_indices & context_indices)
        context_coverage = len(retrieved_lines) / len(lines) if lines else 0.0

        gt, filters = ground_truth_ring_path_in_context(
            retrieved_lines,
            question.ring_system,
        )
        if gt is None:
            raise ValueError(
                f"No {PATH_LENGTH}-reaction ground-truth path in context for "
                f"ring_system={question.ring_system}"
            )

        prompt_question = build_question(
            spec=spec,
            context_reaction_count=filters.context_reaction_count,
            molecule_freq_cap=filters.molecule_freq_cap,
        )

        completion_prompt = f"""
        You are given a subset of chemical reactions in SMILES format and a question.
        <context>
        {retrieved_context}
        </context>
        <question>
        {prompt_question}
        </question>
        """

        print_task15_sample_context(
            sample_index=i,
            question=question,
            gt=gt,
            sampling=sampling,
            full_support_indices=full_support_indices,
            support_indices=support_indices,
            support_in_context=support_in_context,
            context_size=context_size,
            context_line_count=len(retrieved_lines),
            context_coverage=context_coverage,
            filters=filters,
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

        with tracer.start_as_current_span(f"codeact_task15_sample_{i}") as sample_span:
            sample_span.set_attributes(
                {
                    "sample.index": i,
                    "sample.count": len(evaluated_questions),
                    "ring_system": question.ring_system,
                    "path_length": PATH_LENGTH,
                    "agent.name": "codeact",
                }
            )
            with using_tracing_attributes(
                session_id=run_session_id,
                metadata={
                    "sample_index": i,
                    "sample_count": len(evaluated_questions),
                    "ring_system": question.ring_system,
                    "path_length": PATH_LENGTH,
                    "agent": "codeact",
                },
                tags=["codeact", "sample", "task15_RING_CONSTRUCTION_CHAIN"],
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

        records = parse_records_from_lines(retrieved_lines)
        pred_rxns = parse_response(response_text)
        scores = score_prediction(
            pred_rxns=pred_rxns,
            gt=gt,
            records=records,
            filters=filters,
            min_path_reactions=PATH_LENGTH,
        )
        macro_accuracy += float(scores["is_correct"])
        macro_valid_path += float(scores["valid_path"])
        macro_objective_length += float(scores["objective_length_match"])
        macro_reaction_f1 += float(scores["reaction_f1"])
        index_match_count += int(scores["index_match"])
        samples_run += 1

        print_task15_sample_metrics(
            sample_index=i,
            question=question,
            response=response_text,
            pred_rxns=pred_rxns,
            scores=scores,
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
            build_task15_wandb_sample_log(
                sample_index=i,
                question=question,
                gt=gt,
                sampling=sampling,
                full_support_indices=full_support_indices,
                support_indices=support_indices,
                support_in_context=support_in_context,
                filters=filters,
                pred_rxns=pred_rxns,
                scores=scores,
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
                "running_macro_accuracy": macro_accuracy / samples_run,
                "running_macro_valid_path": macro_valid_path / samples_run,
                "running_macro_objective_length": macro_objective_length / samples_run,
                "running_macro_reaction_f1": macro_reaction_f1 / samples_run,
            }
        )

    total = samples_run
    macro_accuracy = macro_accuracy / total if total else 0.0
    macro_valid_path = macro_valid_path / total if total else 0.0
    macro_objective_length = macro_objective_length / total if total else 0.0
    macro_reaction_f1 = macro_reaction_f1 / total if total else 0.0

    print_task15_run_summary(
        total=total,
        index_match_count=index_match_count,
        macro_accuracy=macro_accuracy,
        macro_valid_path=macro_valid_path,
        macro_objective_length=macro_objective_length,
        macro_reaction_f1=macro_reaction_f1,
    )

    update_task15_run_summary(
        run,
        total=total,
        index_match_count=index_match_count,
        macro_accuracy=macro_accuracy,
        macro_valid_path=macro_valid_path,
        macro_objective_length=macro_objective_length,
        macro_reaction_f1=macro_reaction_f1,
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
        )
    )
