import argparse
import asyncio
import os
import random
import uuid

from llama_index.core.llms import ChatMessage
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
from rlm.codeact_helpers import (
    build_context_pipeline,
    extract_response_text,
    extract_usage_metrics,
    load_lines,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 30_000


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="LLM-Task15",
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
        description="Run LLM task 15 — ring-construction chains from acyclic precursors."
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


async def main(model_name: str, context_size: int) -> None:
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    evaluated_questions = [
        question for question in FIXED_QUESTIONS if full_dataset_chain_count(question) > 0
    ]
    if not evaluated_questions:
        raise ValueError("No ring-system questions have non-empty ground truth.")

    print_task15_startup_banner()
    run_session_id = f"llm-task15-{uuid.uuid4()}"

    llm = OpenRouter(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
    )

    run = wandb.init(
        project="LLM-Task15",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "reasoning_effort": REASONING_EFFORT,
            "num_questions": len(evaluated_questions),
            "ring_questions": [q.ring_system for q in evaluated_questions],
            "path_length": PATH_LENGTH,
            "min_selected_ground_truth": TASK15_MIN_SELECTED_GROUND_TRUTH,
            "min_heavy_atoms": MIN_HEAVY_ATOMS,
            "max_heavy_atoms": MAX_HEAVY_ATOMS,
            "task_description": "Ring-construction chains from acyclic precursors via RDKit SMARTS.",
            "ground_truth_definition": TASK15_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK15_TOTAL_REACTIONS,
            "mode": "llm_baseline_no_tools",
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

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(evaluated_questions),
                "ring_system": question.ring_system,
                "path_length": PATH_LENGTH,
                "agent": "llm_baseline",
            },
            tags=["llm-baseline", "sample", "task15_RING_CONSTRUCTION_CHAIN"],
        ):
            response = await llm.achat([ChatMessage(role="user", content=completion_prompt)])

        response_text = extract_response_text(response)
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
                final_input_tokens=prompt_tokens,
                final_output_tokens=completion_tokens,
                final_total_tokens=total_tokens,
                iterations=1,
                sample_cost_usd=sample_cost,
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
