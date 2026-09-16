from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import re
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

from rxnhaystack.manifest import ManifestError

SWISSAI_REQUESTS_PER_MINUTE_ENV = "RXNHAYSTACK_SWISSAI_REQUESTS_PER_MINUTE"
SWISSAI_REQUESTS_PER_MINUTE = 15.0
SWISSAI_HOST_REQUESTS_PER_MINUTE_CAP_ENV = (
    "RXNHAYSTACK_SWISSAI_HOST_REQUESTS_PER_MINUTE_CAP"
)
SWISSAI_RATE_LIMIT_RETRIES_ENV = "RXNHAYSTACK_SWISSAI_RATE_LIMIT_RETRIES"
SWISSAI_RATE_LIMIT_RETRIES = 2
SWISSAI_RATE_LIMIT_MARGIN_SECONDS = 0.25
SWISSAI_BASE_URL = "https://api.swissai.svc.cscs.ch/v1"

ResultT = TypeVar("ResultT")


def _positive_float_from_environment(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as error:
        raise ManifestError(f"{name} must be a positive number") from error
    if value <= 0:
        raise ManifestError(f"{name} must be a positive number")
    return value


def _nonnegative_integer_from_environment(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ManifestError(f"{name} must be a non-negative integer") from error
    if value < 0:
        raise ManifestError(f"{name} must be a non-negative integer")
    return value


class CrossProcessRateLimiter:
    """Space request starts across processes sharing one provider credential."""

    def __init__(
        self,
        *,
        api_key: str,
        requests_per_minute: float,
        state_dir: Path | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ManifestError("A provider API key is required for rate limiting")
        if requests_per_minute <= 0:
            raise ManifestError("requests_per_minute must be positive")
        credential_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        directory = state_dir or Path(tempfile.gettempdir()) / "rxnhaystack-rate-limits"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._state_path = directory / f"swissai-{credential_id}.timestamp"
        self._interval = 60.0 / requests_per_minute + SWISSAI_RATE_LIMIT_MARGIN_SECONDS
        self._clock = clock
        self._sleep = sleep

    def acquire(self) -> None:
        descriptor = os.open(self._state_path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as state:
            fcntl.flock(state, fcntl.LOCK_EX)
            raw_last_start = state.read().strip()
            last_start = float(raw_last_start) if raw_last_start else 0.0
            delay = max(0.0, last_start + self._interval - self._clock())
            if delay:
                self._sleep(delay)
            started_at = self._clock()
            state.seek(0)
            state.truncate()
            state.write(f"{started_at:.9f}\n")
            state.flush()

    async def acquire_async(self) -> None:
        await asyncio.to_thread(self.acquire)


def _swissai_limiter(api_key: str) -> CrossProcessRateLimiter:
    requests_per_minute = _positive_float_from_environment(
        SWISSAI_REQUESTS_PER_MINUTE_ENV,
        SWISSAI_REQUESTS_PER_MINUTE,
    )
    if SWISSAI_HOST_REQUESTS_PER_MINUTE_CAP_ENV in os.environ:
        host_cap = _positive_float_from_environment(
            SWISSAI_HOST_REQUESTS_PER_MINUTE_CAP_ENV,
            requests_per_minute,
        )
        requests_per_minute = min(requests_per_minute, host_cap)
    return CrossProcessRateLimiter(
        api_key=api_key,
        requests_per_minute=requests_per_minute,
    )


def _is_rate_limit_error(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 429


def retry_after_seconds(error: Exception, *, default: float = 60.0) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) if response is not None else {}
    raw_header = headers.get("retry-after") or headers.get("Retry-After")
    if raw_header is not None:
        try:
            return max(0.0, float(raw_header))
        except (TypeError, ValueError):
            pass
    match = re.search(r"retry after\s+([0-9]+(?:\.[0-9]+)?)\s*seconds?", str(error), re.I)
    return float(match.group(1)) if match else default


async def call_swissai_async(
    call: Callable[[], Awaitable[ResultT]],
    *,
    api_key: str,
) -> ResultT:
    limiter = _swissai_limiter(api_key)
    retries = _nonnegative_integer_from_environment(
        SWISSAI_RATE_LIMIT_RETRIES_ENV,
        SWISSAI_RATE_LIMIT_RETRIES,
    )
    for attempt in range(retries + 1):
        await limiter.acquire_async()
        try:
            return await call()
        except Exception as error:
            if not _is_rate_limit_error(error) or attempt == retries:
                raise
            delay = retry_after_seconds(error)
            print(
                f"[SWISSAI-RATE-LIMIT] HTTP 429; retrying after {delay:.1f} seconds "
                f"({attempt + 1}/{retries})",
                flush=True,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def call_swissai_sync(call: Callable[[], ResultT], *, api_key: str) -> ResultT:
    limiter = _swissai_limiter(api_key)
    retries = _nonnegative_integer_from_environment(
        SWISSAI_RATE_LIMIT_RETRIES_ENV,
        SWISSAI_RATE_LIMIT_RETRIES,
    )
    for attempt in range(retries + 1):
        limiter.acquire()
        try:
            return call()
        except Exception as error:
            if not _is_rate_limit_error(error) or attempt == retries:
                raise
            delay = retry_after_seconds(error)
            print(
                f"[SWISSAI-RATE-LIMIT] HTTP 429; retrying after {delay:.1f} seconds "
                f"({attempt + 1}/{retries})",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def is_swissai_url(base_url: str | None) -> bool:
    return bool(base_url and base_url.rstrip("/") == SWISSAI_BASE_URL)
