import argparse
import asyncio
import os
import uuid

from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter
from task16_truncated_synthesis_graph import (
    FULL_CHAIN_LENGTH,
    MAX_HEAVY_ATOMS,
    MIN_HEAVY_ATOMS,
    PREFIX_LENGTH,
    build_question,
    parse_chains,
    parse_records_from_lines,
    score_chain_predictions,
)
from task16_truncated_synthesis_ground_truth import (
    FIXED_QUESTIONS,
    TASK16_FORCED_PREFIX_COUNT,
    TASK16_GROUND_TRUTH_DEFINITION,
    TASK16_MIN_SELECTED_GROUND_TRUTH,
    TASK16_TOTAL_REACTIONS,
    build_task16_eval_context,
    build_task16_wandb_sample_log,
    full_support_indices_for_question,
    hardcoded_full_chains_for_question,
    print_task16_run_summary,
    print_task16_sample_context,
    print_task16_sample_metrics,
    print_task16_startup_banner,
    target_spec_for_question,
    update_task16_run_summary,
)

import wandb
from rlm.codeact_helpers import (
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
        project_name="LLM-Task16",
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
        description="Run LLM task 16 — truncated synthesis route prefixes."
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
    print("Parsing full dataset records (one-time)...")
    full_records = parse_records_from_lines(lines)
    print(f"Parsed {len(full_records)} reactions.")
    print_task16_startup_banner()
    run_session_id = f"llm-task16-{uuid.uuid4()}"

    llm = OpenRouter(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
    )

    run = wandb.init(
        project="LLM-Task16",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "reasoning_effort": REASONING_EFFORT,
            "num_questions": len(FIXED_QUESTIONS),
            "target_questions": [q.question_id for q in FIXED_QUESTIONS],
            "prefix_length": PREFIX_LENGTH,
            "full_chain_length": FULL_CHAIN_LENGTH,
            "forced_prefix_count": TASK16_FORCED_PREFIX_COUNT,
            "min_selected_ground_truth": TASK16_MIN_SELECTED_GROUND_TRUTH,
            "min_heavy_atoms": MIN_HEAVY_ATOMS,
            "max_heavy_atoms": MAX_HEAVY_ATOMS,
            "task_description": "Truncated synthesis prefixes with withheld final reaction.",
            "ground_truth_definition": TASK16_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK16_TOTAL_REACTIONS,
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

    for i, question in enumerate(FIXED_QUESTIONS):
        spec = target_spec_for_question(question)
        full_chains = hardcoded_full_chains_for_question(question.question_id)
        full_support_indices = full_support_indices_for_question(question)
        built = build_task16_eval_context(
            question=question,
            lines=lines,
            context_size=context_size,
            sample_index=i,
            seed=SEED,
            pipeline_name=CONTEXT_PIPELINE_NAME,
            full_records=full_records,
        )
        sampling = built.sampling
        support_indices = set(sampling.support_indices)
        excluded_terminals = sampling.excluded_terminal_indices
        retrieved_lines = built.context_lines
        retrieved_context = built.context_text
        gt = built.gt
        filters = built.filters
        support_in_context = built.support_in_context
        context_coverage = built.context_coverage

        gt_chains = sorted(gt.accepted_reaction_indices or (gt.reaction_indices,))
        records = built.records
        prompt_question = build_question(spec=spec)

        completion_prompt = f"""
        You are given a subset of chemical reactions in SMILES format and a question.
        <context>
        {retrieved_context}
        </context>
        <question>
        {prompt_question}
        </question>
        """

        print_task16_sample_context(
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
                "sample_count": len(FIXED_QUESTIONS),
                "prefix_length": PREFIX_LENGTH,
                "question_id": question.question_id,
                "agent": "llm_baseline",
            },
            tags=["llm-baseline", "sample", "task16_TRUNCATED_SYNTHESIS"],
        ):
            response = await llm.achat([ChatMessage(role="user", content=completion_prompt)])

        response_text = extract_response_text(response)
        parsed_chains = parse_chains(response_text)
        scores = score_chain_predictions(
            pred_chains=parsed_chains,
            gt=gt,
            records=records,
            full_chains=full_chains,
            filters=filters,
        )
        is_exact_match = bool(scores["is_exact_match"])
        if is_exact_match:
            exact_match_count += 1
        macro_precision += float(scores["precision"])
        macro_recall += float(scores["recall"])
        macro_f1 += float(scores["f1"])
        samples_run += 1

        print_task16_sample_metrics(
            sample_index=i,
            question=question,
            response=response_text,
            parsed_chains=parsed_chains,
            gt_chains=list(gt_chains),
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
            build_task16_wandb_sample_log(
                sample_index=i,
                question=question,
                gt=gt,
                sampling=sampling,
                full_support_indices=full_support_indices,
                support_indices=support_indices,
                support_in_context=support_in_context,
                filters=filters,
                parsed_chains=parsed_chains,
                gt_chains=list(gt_chains),
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
                "running_exact_match_accuracy": exact_match_count / samples_run,
                "running_macro_precision": macro_precision / samples_run,
                "running_macro_recall": macro_recall / samples_run,
                "running_macro_f1": macro_f1 / samples_run,
            }
        )

    total = samples_run
    macro_precision = macro_precision / total if total else 0.0
    macro_recall = macro_recall / total if total else 0.0
    macro_f1 = macro_f1 / total if total else 0.0

    print_task16_run_summary(
        total=total,
        exact_match_count=exact_match_count,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
    )

    update_task16_run_summary(
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
        )
    )
