from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rxnhaystack import rate_limit
from rxnhaystack.rate_limit import CrossProcessRateLimiter, call_swissai_async


def test_cross_process_limiter_spaces_persisted_request_starts(tmp_path: Path) -> None:
    now = [100.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    first = CrossProcessRateLimiter(
        api_key="private",
        requests_per_minute=15,
        state_dir=tmp_path,
        clock=clock,
        sleep=sleep,
    )
    second = CrossProcessRateLimiter(
        api_key="private",
        requests_per_minute=15,
        state_dir=tmp_path,
        clock=clock,
        sleep=sleep,
    )

    first.acquire()
    second.acquire()

    assert sleeps == [pytest.approx(4.25)]
    state_files = list(tmp_path.iterdir())
    assert len(state_files) == 1
    assert state_files[0].stat().st_mode & 0o777 == 0o600


def test_swissai_429_waits_for_provider_delay_and_retries(monkeypatch) -> None:
    class FakeLimiter:
        def __init__(self) -> None:
            self.acquisitions = 0

        async def acquire_async(self) -> None:
            self.acquisitions += 1

    class FakeResponse:
        headers = {"retry-after": "2.5"}

    class FakeRateLimitError(Exception):
        status_code = 429
        response = FakeResponse()

    limiter = FakeLimiter()
    sleeps: list[float] = []
    calls = 0

    async def request() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FakeRateLimitError("retry after 99 seconds")
        return "ok"

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setenv(rate_limit.SWISSAI_RATE_LIMIT_RETRIES_ENV, "1")
    monkeypatch.setattr(rate_limit, "_swissai_limiter", lambda _key: limiter)
    monkeypatch.setattr(rate_limit.asyncio, "sleep", sleep)

    result = asyncio.run(call_swissai_async(request, api_key="private"))

    assert result == "ok"
    assert calls == 2
    assert limiter.acquisitions == 2
    assert sleeps == [2.5]
