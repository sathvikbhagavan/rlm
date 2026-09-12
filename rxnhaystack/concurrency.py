from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from typing import TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class OrderedAsyncGate:
    """Serialize deterministic preparation by zero-based item position."""

    def __init__(self) -> None:
        self._next = 0
        self._condition = asyncio.Condition()

    @asynccontextmanager
    async def turn(self, position: int) -> AsyncIterator[None]:
        if position < 0:
            raise ValueError("position must be non-negative")
        async with self._condition:
            await self._condition.wait_for(lambda: position == self._next)
        try:
            yield
        finally:
            async with self._condition:
                self._next += 1
                self._condition.notify_all()


def question_parallelism_from_environment(*, default: int = 1) -> int:
    """Read the launcher's per-worker question limit with a safe standalone default."""

    raw = os.environ.get("RXNHAYSTACK_QUESTION_PARALLELISM", str(default))
    try:
        parallelism = int(raw)
    except ValueError as error:
        raise ValueError("RXNHAYSTACK_QUESTION_PARALLELISM must be an integer") from error
    if parallelism < 1:
        raise ValueError("RXNHAYSTACK_QUESTION_PARALLELISM must be at least 1")
    return parallelism


async def map_async_bounded(
    function: Callable[[InputT], Awaitable[OutputT]],
    items: Iterable[InputT],
    *,
    max_concurrency: int,
) -> list[OutputT]:
    """Map an async function concurrently while preserving input order."""

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(item: InputT) -> OutputT:
        async with semaphore:
            return await function(item)

    return await asyncio.gather(*(run_one(item) for item in items))


async def map_threads_bounded(
    function: Callable[[InputT], OutputT],
    items: Iterable[InputT],
    *,
    max_concurrency: int,
) -> list[OutputT]:
    """Run isolated synchronous callables in bounded worker threads."""

    async def run_one(item: InputT) -> OutputT:
        return await asyncio.to_thread(function, item)

    return await map_async_bounded(run_one, items, max_concurrency=max_concurrency)
