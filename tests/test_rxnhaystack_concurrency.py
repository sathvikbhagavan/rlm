from __future__ import annotations

import asyncio

import pytest

from rxnhaystack.concurrency import (
    OrderedAsyncGate,
    map_async_bounded,
    map_threads_bounded,
    question_parallelism_from_environment,
)


def test_ordered_gate_preserves_preparation_order() -> None:
    async def run_test() -> list[int]:
        gate = OrderedAsyncGate()
        prepared: list[int] = []

        async def worker(position: int) -> None:
            if position == 0:
                await asyncio.sleep(0.01)
            async with gate.turn(position):
                prepared.append(position)

        await asyncio.gather(*(worker(position) for position in reversed(range(4))))
        return prepared

    assert asyncio.run(run_test()) == [0, 1, 2, 3]


def test_question_parallelism_uses_safe_default_and_environment(monkeypatch) -> None:
    monkeypatch.delenv("RXNHAYSTACK_QUESTION_PARALLELISM", raising=False)
    assert question_parallelism_from_environment() == 1
    monkeypatch.setenv("RXNHAYSTACK_QUESTION_PARALLELISM", "4")
    assert question_parallelism_from_environment() == 4


def test_question_parallelism_rejects_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_QUESTION_PARALLELISM", "0")
    with pytest.raises(ValueError, match="at least 1"):
        question_parallelism_from_environment()


def test_async_map_is_bounded_and_preserves_input_order() -> None:
    async def run_test() -> tuple[list[int], int]:
        active = 0
        maximum_active = 0
        lock = asyncio.Lock()

        async def work(value: int) -> int:
            nonlocal active, maximum_active
            async with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01 * (4 - value))
            async with lock:
                active -= 1
            return value * 2

        results = await map_async_bounded(work, [1, 2, 3], max_concurrency=2)
        return results, maximum_active

    results, maximum_active = asyncio.run(run_test())

    assert results == [2, 4, 6]
    assert maximum_active == 2


def test_thread_map_runs_sync_workers_and_preserves_order() -> None:
    async def run_test() -> list[int]:
        return await map_threads_bounded(lambda value: value * 2, [3, 1, 2], max_concurrency=2)

    results = asyncio.run(run_test())

    assert results == [6, 2, 4]


def test_bounded_map_rejects_zero_concurrency() -> None:
    async def identity(value: int) -> int:
        return value

    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(map_async_bounded(identity, [1], max_concurrency=0))
