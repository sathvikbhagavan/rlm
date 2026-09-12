import asyncio
import os
import random
import time
import uuid
from dataclasses import dataclass

import wandb
from llama_index.core.llms import ChatMessage
from rxnhaystack.providers import build_benchmark_llm
from task1_hardcoded_cases import (
    TASK1_HARDCODED_GROUND_TRUTH_INDICES,
    TASK1_HARDCODED_PRODUCTS,
)

from rlm.codeact_helpers import (
    build_context_pipeline,
    extract_response_text,
    extract_usage_metrics,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens
from rxnhaystack.concurrency import map_async_bounded
from rxnhaystack.metrics import RunMetrics, cost_chf_from_usd, write_run_metrics
from rxnhaystack.worker import BenchmarkRuntime

DATASET_PATH = "~/datasets/rxnhaystack/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = __import__("os").environ.get("RXNHAYSTACK_MODEL", "openai/gpt-5-mini")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
NUM_QUESTIONS = 10
SEED = int(__import__("os").environ.get("RXNHAYSTACK_SEED", "42"))
CONTEXT_SIZE = int(__import__("os").environ.get("RXNHAYSTACK_CONTEXT_SIZE", "100"))
CONTEXT_PIPELINE_NAME = "random"
MIN_SELECTED_GROUND_TRUTH = 5
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 40_000
# os.environ["WANDB_MODE"] = "disabled"


@dataclass(frozen=True)
class Sample:
    index: int
    question: str
    target_product: str
    target_index: int
    target_line: str
    retrieved_context: str
    retrieved_lines: tuple[str, ...]
    ground_truth_in_context: frozenset[int]
    context_has_ground_truth: bool
    context_coverage: float
    completion_prompt: str


@dataclass(frozen=True)
class SampleResult:
    sample: Sample
    response_text: str
    predicted_indices: frozenset[int]
    precision: float
    recall: float
    f1: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    latency_seconds: float


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


async def main() -> None:
    started = time.monotonic()
    runtime = BenchmarkRuntime.from_defaults(
        model=MODEL_NAME,
        dataset_path=DATASET_PATH,
        seed=SEED,
    )
    maybe_init_tracing()
    lines = load_lines(runtime.dataset_path)
    rng = random.Random(runtime.seed)
    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=rng,
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )
    if NUM_QUESTIONS > len(TASK1_HARDCODED_PRODUCTS):
        raise ValueError(
            f"NUM_QUESTIONS={NUM_QUESTIONS} exceeds "
            f"available hardcoded products={len(TASK1_HARDCODED_PRODUCTS)}"
        )
    selected_products = TASK1_HARDCODED_PRODUCTS[:NUM_QUESTIONS]
    selected_ground_truth = TASK1_HARDCODED_GROUND_TRUTH_INDICES[:NUM_QUESTIONS]
    questions = [build_question(product) for product in selected_products]
    print(f"[QUESTION-SAMPLING] using_hardcoded_products={len(selected_products)}")
    run_session_id = f"llm-task1-{uuid.uuid4()}"

    run = wandb.init(
        project="LLM-Task1",
        config={
            "MODEL_NAME": runtime.model,
            "NUM_QUESTIONS": NUM_QUESTIONS,
            "dataset_path": str(runtime.dataset_path),
            "seed": runtime.seed,
            "question_parallelism": runtime.question_parallelism,
            "context_size": CONTEXT_SIZE,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "min_selected_ground_truth": MIN_SELECTED_GROUND_TRUTH,
            "reasoning_effort": REASONING_EFFORT,
            "mode": "llm_baseline_no_tools",
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    samples: list[Sample] = []
    for i, question in enumerate(questions):
        target_product = selected_products[i]
        ground_truth_index_set = set(selected_ground_truth[i])
        target_index = sorted(ground_truth_index_set)[0]
        target_line = lines[target_index]
        retrieved_context = context_pipeline.build_context(
            context_size=CONTEXT_SIZE,
            correct_indices=ground_truth_index_set,
            query=target_product,
        )
        retrieved_lines = tuple(line for line in retrieved_context.splitlines() if line.strip())
        retrieved_indices = {
            int(line.split(" ", 1)[0])
            for line in retrieved_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        ground_truth_in_context_set = ground_truth_index_set & retrieved_indices
        gt_in_context_count = len(ground_truth_in_context_set)
        print(
            f"[CONTEXT] requested_size={CONTEXT_SIZE} actual_size={len(retrieved_lines)} "
            f"ground_truth_in_context={gt_in_context_count}/{len(ground_truth_index_set)}"
        )
        context_has_ground_truth = bool(ground_truth_in_context_set)
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

        samples.append(
            Sample(
                index=i,
                question=question,
                target_product=target_product,
                target_index=target_index,
                target_line=target_line,
                retrieved_context=retrieved_context,
                retrieved_lines=retrieved_lines,
                ground_truth_in_context=frozenset(ground_truth_in_context_set),
                context_has_ground_truth=context_has_ground_truth,
                context_coverage=context_coverage,
                completion_prompt=completion_prompt,
            )
        )

    async def evaluate(sample: Sample) -> SampleResult:
        print(f"Question {sample.index + 1}/{len(samples)}")
        llm = build_benchmark_llm(
            model=runtime.model,
            api_key=OPENROUTER_API_KEY,
            max_tokens=MAX_OUTPUT_TOKENS,
            reasoning_effort=REASONING_EFFORT,
            additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
        )
        with runtime.timed_sample("llm", sample.index) as timer:
            with using_tracing_attributes(
                session_id=run_session_id,
                metadata={
                    "sample_index": sample.index,
                    "sample_count": len(samples),
                    "target_index": sample.target_index,
                    "agent": "llm_baseline",
                },
                tags=["llm-baseline", "sample"],
            ):
                response = await llm.achat(
                    [ChatMessage(role="user", content=sample.completion_prompt)]
                )
        response_text = extract_response_text(response)
        parsed_indices = parse_indices(response_text)
        predicted_index_set = set(parsed_indices)
        precision, recall, f1 = precision_recall_f1(
            predicted_index_set, set(sample.ground_truth_in_context)
        )
        if f1 < 1.0:
            print(
                f"Mismatch for target_index={sample.target_index}: "
                f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
            )
            print(f"Line in context: {sample.target_line}")
            print(f"Product: {sample.target_product}")
            print(f"Predicted indices: {sorted(predicted_index_set)}")
            print(f"Ground truth indices (in context): {sorted(sample.ground_truth_in_context)}")
            print("--------------------------------")
        else:
            print(f"F1 is 1.0 for target_index={sample.target_index}")

        usage_metrics = extract_usage_metrics(response)
        prompt_tokens = int(usage_metrics.get("prompt_tokens", 0))
        completion_tokens = int(usage_metrics.get("completion_tokens", 0))
        total_tokens = int(usage_metrics.get("total_tokens", 0))
        sample_cost = float(usage_metrics["cost_usd"]) if "cost_usd" in usage_metrics else None
        if total_tokens == 0:
            prompt_tokens = count_tokens(
                [{"role": "user", "content": sample.completion_prompt}], runtime.model
            )
            completion_tokens = count_tokens(
                [{"role": "assistant", "content": response_text}],
                runtime.model,
            )
            total_tokens = prompt_tokens + completion_tokens
        assert timer.duration_seconds is not None
        return SampleResult(
            sample=sample,
            response_text=response_text,
            predicted_indices=frozenset(predicted_index_set),
            precision=precision,
            recall=recall,
            f1=f1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=sample_cost,
            latency_seconds=timer.duration_seconds,
        )

    results = await map_async_bounded(
        evaluate,
        samples,
        max_concurrency=runtime.question_parallelism,
    )
    precision_sum = sum(result.precision for result in results)
    recall_sum = sum(result.recall for result in results)
    f1_sum = sum(result.f1 for result in results)
    retrieval_hits = sum(int(result.sample.context_has_ground_truth) for result in results)
    costs = [result.cost_usd for result in results if result.cost_usd is not None]
    total_cost_usd = sum(costs)
    samples_with_cost = len(costs)

    for completed_count, result in enumerate(results, start=1):
        i = result.sample.index

        wandb.log(
            {
                "sample_iteration": 1,
                f"sample/{i}/iteration_input_tokens": result.prompt_tokens,
                f"sample/{i}/iteration_output_tokens": result.completion_tokens,
                f"sample/{i}/iteration_total_tokens": result.total_tokens,
                **(
                    {f"sample/{i}/iteration_cost_usd": result.cost_usd}
                    if result.cost_usd is not None
                    else {}
                ),
            }
        )
        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/final_total_input_tokens": result.prompt_tokens,
                f"sample/{i}/final_total_output_tokens": result.completion_tokens,
                f"sample/{i}/final_total_tokens": result.total_tokens,
                f"sample/{i}/iterations": 1,
                f"sample/{i}/precision": result.precision,
                f"sample/{i}/recall": result.recall,
                f"sample/{i}/f1": result.f1,
                f"sample/{i}/target_index": result.sample.target_index,
                f"sample/{i}/completion_prompt_char_count": len(result.sample.completion_prompt),
                f"sample/{i}/context_char_count": len(result.sample.retrieved_context),
                f"sample/{i}/context_size": CONTEXT_SIZE,
                f"sample/{i}/retrieved_line_count": len(result.sample.retrieved_lines),
                f"sample/{i}/context_coverage": result.sample.context_coverage,
                f"sample/{i}/context_has_ground_truth": int(result.sample.context_has_ground_truth),
                f"sample/{i}/latency_seconds": result.latency_seconds,
                **(
                    {f"sample/{i}/final_total_cost_usd": result.cost_usd}
                    if result.cost_usd is not None
                    else {}
                ),
            }
        )
        wandb.log(
            {
                "completed_samples": completed_count,
            }
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
        if samples_with_cost != total:
            raise RuntimeError("OpenRouter did not report cost for every launched sample")
        write_run_metrics(
            RunMetrics(
                calls=total,
                input_tokens=sum(result.prompt_tokens for result in results),
                output_tokens=sum(result.completion_tokens for result in results),
                total_tokens=sum(result.total_tokens for result in results),
                latency_seconds=time.monotonic() - started,
                tool_time_seconds=0,
                cost_usd=total_cost_usd,
                cost_chf=cost_chf_from_usd(total_cost_usd),
                wandb_url=getattr(run, "url", None),
                results={
                    "macro_precision": avg_precision,
                    "macro_recall": avg_recall,
                    "macro_f1": avg_f1,
                    "retrieval_hit_rate": retrieval_hit_rate,
                    "question_latencies_seconds": [result.latency_seconds for result in results],
                },
            )
        )
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
