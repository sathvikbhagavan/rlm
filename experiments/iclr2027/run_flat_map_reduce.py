from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from typing import Any

import wandb
from llama_index.core.llms import ChatMessage

from rlm.codeact_helpers import extract_response_text, extract_usage_metrics, load_lines
from rlm.utils.token_utils import count_tokens
from rxnhaystack.manifest import ManifestError
from rxnhaystack.map_reduce import (
    DEFAULT_CHUNK_SIZE,
    CorpusChunk,
    MapperReply,
    run_map_reduce,
    score_predictions,
)
from rxnhaystack.map_reduce_tasks import mapper_prompt, selected_questions
from rxnhaystack.metrics import RunMetrics, cost_chf_from_usd, write_run_metrics
from rxnhaystack.providers import benchmark_provider, build_benchmark_llm
from rxnhaystack.runtime import WorkerConfig, atomic_write_json

QUESTION_ID_ENV = "RXNHAYSTACK_MAP_REDUCE_QUESTION_ID"
CHUNK_SIZE_ENV = "RXNHAYSTACK_MAP_REDUCE_CHUNK_SIZE"
MAX_PARALLEL_ENV = "RXNHAYSTACK_MAP_REDUCE_MAX_PARALLEL"
MAX_OUTPUT_TOKENS_ENV = "RXNHAYSTACK_MAP_REDUCE_OUTPUT_LIMIT"


def positive_integer(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ManifestError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ManifestError(f"{name} must be a positive integer")
    return value


async def main() -> None:
    config = WorkerConfig.from_environment()
    if config.method != "map-reduce":
        raise ManifestError("Flat map-reduce runner requires method='map-reduce'")
    question_id = os.environ.get(QUESTION_ID_ENV, "")
    questions = selected_questions()
    if question_id not in questions:
        raise ManifestError(f"{QUESTION_ID_ENV} must be one of: {', '.join(sorted(questions))}")
    question = questions[question_id]
    if config.task != question.benchmark_task:
        raise ManifestError(
            f"Run task {config.task!r} does not match question {question.benchmark_task!r}"
        )
    chunk_size = positive_integer(CHUNK_SIZE_ENV, DEFAULT_CHUNK_SIZE)
    max_parallel = positive_integer(MAX_PARALLEL_ENV, 1)
    max_output_tokens = positive_integer(MAX_OUTPUT_TOKENS_ENV, 4096)
    rows = load_lines(str(config.require_dataset().cleaned))

    llm = build_benchmark_llm(
        model=config.model,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        max_tokens=max_output_tokens,
        max_retries=0,
        reasoning_effort="high",
        additional_kwargs={"max_completion_tokens": max_output_tokens},
    )

    async def mapper(chunk: CorpusChunk, _attempt: int) -> MapperReply:
        prompt = mapper_prompt(question=question.question, chunk_text=chunk.text)
        response = await llm.achat([ChatMessage(role="user", content=prompt)])
        response_text = extract_response_text(response)
        usage = extract_usage_metrics(response)
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        if int(usage.get("total_tokens", 0)) == 0:
            input_tokens = count_tokens([{"role": "user", "content": prompt}], config.model)
            output_tokens = count_tokens(
                [{"role": "assistant", "content": response_text}], config.model
            )
        return MapperReply(
            text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=float(usage["cost_usd"]) if "cost_usd" in usage else None,
        )

    run = wandb.init(
        project="RxnHaystack-Flat-Map-Reduce",
        config={
            "rxnhaystack_run_id": config.run_id,
            "question_id": question.question_id,
            "benchmark_task": question.benchmark_task,
            "model": config.model,
            "provider": benchmark_provider(),
            "chunk_size": chunk_size,
            "chunk_order": "dataset-order-contiguous",
            "reduction": "deterministic-set-union",
            "max_parallel": max_parallel,
            "seed": config.seed,
            "repetition": config.repetition,
        },
    )
    try:
        # Keep progress beside attempt directories so launcher retries resume
        # this run instead of starting again from chunk zero.
        checkpoint_path = config.run_dir.parent / "map-reduce-checkpoint.json"
        result = await run_map_reduce(
            rows,
            question=question.question,
            model=config.model,
            mapper=mapper,
            checkpoint_path=checkpoint_path,
            chunk_size=chunk_size,
            max_parallel=max_parallel,
        )
        score = score_predictions(
            result.predicted_indices,
            set(question.ground_truth_indices),
            invalid_claims=result.invalid_claims,
        )
        provider_cost_usd = result.cost_usd
        if provider_cost_usd is None and benchmark_provider() == "swissai":
            provider_cost_usd = 0.0
            cost_status = "provider-free"
        else:
            cost_status = result.cost_status
        details: dict[str, Any] = {
            "question_id": question.question_id,
            "benchmark_task": question.benchmark_task,
            "model": config.model,
            "provider": benchmark_provider(),
            "chunk_size": chunk_size,
            "chunk_count": result.chunk_count,
            "completed_chunk_count": len(result.chunks),
            "resumed_chunk_count": result.resumed_chunks,
            "invalid_claim_count": result.invalid_claims,
            "predicted_indices": list(result.predicted_indices),
            "ground_truth_count": len(question.ground_truth_indices),
            "cost_status": cost_status,
            "cost_usd": provider_cost_usd,
            "score": asdict(score),
            "chunks": [asdict(chunk) for chunk in result.chunks],
        }
        details_path = config.run_dir / "map-reduce-details.json"
        atomic_write_json(details_path, details)
        run.log(
            {
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
                "exact_match": int(score.exact_match),
                "calls": result.calls,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "invalid_claims": result.invalid_claims,
                "completed_chunks": len(result.chunks),
            }
        )
        run.summary.update(
            {
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
                "exact_match": int(score.exact_match),
                "ground_truth_count": len(question.ground_truth_indices),
                "predicted_count": len(result.predicted_indices),
                "invalid_claims": result.invalid_claims,
                "chunk_count": result.chunk_count,
                "cost_status": cost_status,
            }
        )
        if provider_cost_usd is None:
            raise RuntimeError(
                "Scientific map-reduce details were preserved, but provider cost is unavailable; "
                "current archival metrics require known accounting"
            )
        write_run_metrics(
            RunMetrics(
                calls=result.calls,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.input_tokens + result.output_tokens,
                latency_seconds=result.latency_seconds,
                tool_time_seconds=0.0,
                cost_usd=provider_cost_usd,
                cost_chf=cost_chf_from_usd(provider_cost_usd),
                wandb_url=run.url,
                results={
                    "question_id": question.question_id,
                    "precision": score.precision,
                    "recall": score.recall,
                    "f1": score.f1,
                    "exact_match": score.exact_match,
                    "invalid_claims": result.invalid_claims,
                    "chunk_count": result.chunk_count,
                    "resumed_chunks": result.resumed_chunks,
                    "cost_status": cost_status,
                    "details_path": str(details_path),
                },
            )
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
