from __future__ import annotations

import asyncio
import time

from rxnhaystack.concurrency import map_async_bounded
from rxnhaystack.metrics import RunMetrics, write_run_metrics
from rxnhaystack.runtime import WorkerConfig
from rxnhaystack.worker import BenchmarkRuntime


async def main() -> None:
    config = WorkerConfig.from_environment()
    runtime = BenchmarkRuntime(
        model=config.model,
        dataset_path=config.cleaned_dataset or config.run_dir,
        seed=config.seed,
        question_parallelism=config.question_parallelism,
        worker_config=config,
    )
    active = 0
    maximum_active = 0
    lock = asyncio.Lock()
    started = time.monotonic()

    async def question(index: int) -> int:
        nonlocal active, maximum_active
        with runtime.timed_sample("concurrency-smoke", index):
            async with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.08)
            async with lock:
                active -= 1
            return index

    results = await map_async_bounded(
        question,
        range(6),
        max_concurrency=config.question_parallelism,
    )
    duration = time.monotonic() - started
    write_run_metrics(
        RunMetrics(
            calls=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_seconds=duration,
            tool_time_seconds=0,
            cost_usd=0,
            cost_chf=0,
            results={
                "question_count": len(results),
                "maximum_active_questions": maximum_active,
                "result_order": results,
            },
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
