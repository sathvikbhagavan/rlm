import argparse
import asyncio
import os
import random
import time
import uuid

from llama_index.core.workflow import Context
from task1_hardcoded_cases import (
    TASK1_HARDCODED_GROUND_TRUTH_INDICES,
    TASK1_HARDCODED_PRODUCTS,
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
from rlm.codeact_helpers import (
    build_context_pipeline,
    extract_response_text,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import get_tracer, init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens
from rxnhaystack.concurrency import map_async_bounded
from rxnhaystack.metrics import RunMetrics, cost_chf_from_usd, write_run_metrics
from rxnhaystack.providers import build_benchmark_llm, provider_reports_cost
from rxnhaystack.worker import BenchmarkRuntime, codeact_workflow_timeout

DATASET_PATH = "~/datasets/rxnhaystack/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = __import__("os").environ.get("RXNHAYSTACK_MODEL", "openai/gpt-5-mini")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = codeact_workflow_timeout(default=900.0)
NUM_QUESTIONS = 10
SEED = int(__import__("os").environ.get("RXNHAYSTACK_SEED", "42"))
CONTEXT_SIZE = int(__import__("os").environ.get("RXNHAYSTACK_CONTEXT_SIZE", "500"))
CONTEXT_PIPELINE_NAME = "random"
MIN_SELECTED_GROUND_TRUTH = 5
MAX_OUTPUT_TOKENS = 20_000
MAX_ITERATIONS = 8
REASONING_EFFORT = "low"
LLM_TIMEOUT_RETRIES = 2
LLM_TIMEOUT_RETRY_BACKOFF_S = 2.0
# os.environ["WANDB_MODE"] = "disabled"


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
    There is a list of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Find all the indices of the reactions for the following PRODUCT
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
    started = time.monotonic()
    runtime = BenchmarkRuntime.from_defaults(
        model=model_name,
        dataset_path=dataset_path,
        seed=SEED,
    )
    model_name = runtime.model
    dataset_path = str(runtime.dataset_path)
    maybe_init_tracing()
    tracer = get_tracer("codeact-task1")
    lines = load_lines(dataset_path=dataset_path)
    rng = random.Random(runtime.seed)
    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=rng,
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )
    if num_questions > len(TASK1_HARDCODED_PRODUCTS):
        raise ValueError(
            f"num_questions={num_questions} exceeds "
            f"available hardcoded products={len(TASK1_HARDCODED_PRODUCTS)}"
        )
    selected_products = TASK1_HARDCODED_PRODUCTS[:num_questions]
    selected_ground_truth = TASK1_HARDCODED_GROUND_TRUTH_INDICES[:num_questions]
    questions = [build_question(product) for product in selected_products]
    print(f"[QUESTION-SAMPLING] using_hardcoded_products={len(selected_products)}")
    run_session_id = f"codeact-task1-{uuid.uuid4()}"

    run = wandb.init(
        project="CodeAct-Task1",
        config={
            "MODEL_NAME": model_name,
            "NUM_QUESTIONS": num_questions,
            "dataset_path": dataset_path,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": runtime.seed,
            "question_parallelism": runtime.question_parallelism,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "min_selected_ground_truth": MIN_SELECTED_GROUND_TRUTH,
            "reasoning_effort": REASONING_EFFORT,
            "llm_timeout_retries": LLM_TIMEOUT_RETRIES,
            "llm_timeout_retry_backoff_s": LLM_TIMEOUT_RETRY_BACKOFF_S,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    retrieval_hits = 0
    total_cost_usd = 0.0
    samples_with_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_calls = 0
    total_tool_time = 0.0
    question_latencies = [0.0] * len(questions)

    async def evaluate_one(item: tuple[int, str]) -> None:
        nonlocal precision_sum, recall_sum, f1_sum, retrieval_hits
        nonlocal total_cost_usd, samples_with_cost
        nonlocal total_calls, total_input_tokens, total_output_tokens, total_tool_time
        i, question = item
        print(f"Question {i + 1}/{len(questions)}")
        target_product = selected_products[i]
        ground_truth_index_set = set(selected_ground_truth[i])
        target_index = sorted(ground_truth_index_set)[0]
        target_line = lines[target_index]
        retrieved_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=ground_truth_index_set,
            query=target_product,
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
        context_has_ground_truth = bool(ground_truth_in_context_set)
        retrieval_hits += int(context_has_ground_truth)
        if not context_has_ground_truth:
            print(
                f"[WARNING] Ground truth missing from retrieved context for target_index={target_index}"
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
            llm=build_benchmark_llm(
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
            **runtime.codeact_callbacks(sample_id=i),
        )
        ctx = Context(agent)

        with runtime.timed_sample("codeact", i) as timer:
            with tracer.start_as_current_span(f"codeact_task1_sample_{i}") as sample_span:
                sample_span.set_attributes(
                    {
                        "sample.index": i,
                        "sample.count": len(questions),
                        "target.index": target_index,
                        "agent.name": "codeact",
                    }
                )
                with using_tracing_attributes(
                    session_id=run_session_id,
                    metadata={
                        "sample_index": i,
                        "sample_count": len(questions),
                        "target_index": target_index,
                        "agent": "codeact",
                    },
                    tags=["codeact", "sample"],
                ):
                    response = await run_agent_verbose(agent, ctx, completion_prompt)
        assert timer.duration_seconds is not None
        question_latencies[i] = timer.duration_seconds

        response_text = extract_response_text(response)
        # print(f"Raw response: {response_text!r}")
        # print("-" * 60)
        llm_turn_metrics = await ctx.store.get("llm_turn_metrics", default=[])
        tool_turn_metrics = await ctx.store.get("tool_turn_metrics", default=[])
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
        predicted_index_set = set(parsed_indices)
        precision, recall, f1 = precision_recall_f1(
            predicted_index_set, ground_truth_in_context_set
        )
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        if f1 < 1.0:
            print(
                f"Mismatch for target_index={target_index}: "
                f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
            )
            print(f"Line in context: {target_line}")
            print(f"Product: {target_product}")
            print(f"Predicted indices: {sorted(predicted_index_set)}")
            print(f"Ground truth indices (in context): {sorted(ground_truth_in_context_set)}")
            print("--------------------------------")
        else:
            print(f"F1 is 1.0 for target_index={target_index}")

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
        final_cost = sum(
            float(metric.get("iteration_cost_usd", 0.0)) for metric in llm_turn_metrics
        )
        has_cost = any("iteration_cost_usd" in metric for metric in llm_turn_metrics)
        if has_cost:
            total_cost_usd += final_cost
            samples_with_cost += 1
        total_input_tokens += final_input_tokens
        total_output_tokens += final_output_tokens
        total_calls += len(llm_turn_metrics)
        sample_tool_time = sum(float(metric["tool_time_seconds"]) for metric in tool_turn_metrics)
        total_tool_time += sample_tool_time

        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/final_total_input_tokens": final_input_tokens,
                f"sample/{i}/final_total_output_tokens": final_output_tokens,
                f"sample/{i}/final_total_tokens": final_total_tokens,
                f"sample/{i}/iterations": len(llm_turn_metrics),
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/target_index": target_index,
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/context_size": context_size,
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_coverage": context_coverage,
                f"sample/{i}/context_has_ground_truth": int(context_has_ground_truth),
                f"sample/{i}/latency_seconds": timer.duration_seconds,
                f"sample/{i}/tool_time_seconds": sample_tool_time,
                **({f"sample/{i}/final_total_cost_usd": final_cost} if has_cost else {}),
            }
        )

    await map_async_bounded(
        evaluate_one,
        list(enumerate(questions)),
        max_concurrency=runtime.question_parallelism,
    )

    total = len(questions)
    avg_precision = (precision_sum / total) if total else 0.0
    avg_recall = (recall_sum / total) if total else 0.0
    avg_f1 = (f1_sum / total) if total else 0.0
    retrieval_hit_rate = (retrieval_hits / total) if total else 0.0
    print(f"Macro Precision: {avg_precision:.4f}")
    print(f"Macro Recall: {avg_recall:.4f}")
    print(f"Macro F1: {avg_f1:.4f}")
    print(f"Retrieval hit-rate (ground truth in context): {retrieval_hit_rate:.4f}")

    run.summary["total"] = total
    run.summary["macro_precision"] = avg_precision
    run.summary["macro_recall"] = avg_recall
    run.summary["macro_f1"] = avg_f1
    run.summary["retrieval_hits"] = retrieval_hits
    run.summary["retrieval_hit_rate"] = retrieval_hit_rate
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    if runtime.launched:
        if provider_reports_cost() and samples_with_cost != total:
            raise RuntimeError("OpenRouter did not report cost for every launched sample")
        write_run_metrics(
            RunMetrics(
                calls=total_calls,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                latency_seconds=time.monotonic() - started,
                tool_time_seconds=total_tool_time,
                cost_usd=total_cost_usd,
                cost_chf=cost_chf_from_usd(total_cost_usd),
                wandb_url=getattr(run, "url", None),
                results={
                    "macro_precision": avg_precision,
                    "macro_recall": avg_recall,
                    "macro_f1": avg_f1,
                    "retrieval_hit_rate": retrieval_hit_rate,
                    "question_latencies_seconds": question_latencies,
                },
            )
        )
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
