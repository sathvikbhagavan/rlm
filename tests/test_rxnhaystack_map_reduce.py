from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rxnhaystack.map_reduce import (
    CorpusChunk,
    MapperReply,
    MapReduceError,
    PartialCoverageError,
    build_chunks,
    run_map_reduce,
    score_predictions,
)


def rows(count: int) -> list[str]:
    return [f"{index} C>>C" for index in range(count)]


def reply(text: str, *, cost: float | None = 0.01) -> MapperReply:
    return MapperReply(text=text, input_tokens=10, output_tokens=2, cost_usd=cost)


def test_chunks_cover_rows_exactly_once_and_keep_short_last_chunk() -> None:
    chunks = build_chunks(rows(7), chunk_size=3)

    assert [chunk.chunk_id for chunk in chunks] == [0, 1, 2]
    assert [len(chunk.rows) for chunk in chunks] == [3, 3, 1]
    assert [index for chunk in chunks for index in sorted(chunk.indices)] == list(range(7))
    assert chunks[2].text == "6 C>>C"


@pytest.mark.parametrize("bad_rows", [["C>>C"], ["0 C>>C", "0 N>>N"]])
def test_chunks_reject_missing_or_duplicate_indices(bad_rows: list[str]) -> None:
    with pytest.raises(ValueError):
        build_chunks(bad_rows, chunk_size=1)


def test_chunks_reject_empty_corpus() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_chunks([], chunk_size=1)


def test_union_is_ordered_deduplicated_and_invalid_claims_are_false_positives(
    tmp_path: Path,
) -> None:
    async def mapper(chunk: CorpusChunk, _attempt: int) -> MapperReply:
        answers = {0: "2, 1, 2, 99", 1: "4, 2"}
        return reply(answers[chunk.chunk_id])

    result = asyncio.run(
        run_map_reduce(
            rows(6),
            question="q",
            model="mock",
            mapper=mapper,
            checkpoint_path=tmp_path / "checkpoint.json",
            chunk_size=3,
            max_parallel=2,
        )
    )
    score = score_predictions(
        result.predicted_indices,
        {1, 2, 4},
        invalid_claims=result.invalid_claims,
    )

    assert result.predicted_indices == (1, 2, 4)
    assert result.invalid_claims == 2
    assert score.true_positives == 3
    assert score.false_positives == 2
    assert score.precision == pytest.approx(0.6)
    assert score.recall == 1
    assert not score.exact_match


def test_empty_and_malformed_responses_have_bounded_retries(tmp_path: Path) -> None:
    responses = iter(["", "these are 1 and 2", "1, 2"])
    sleeps: list[float] = []

    async def mapper(_chunk: CorpusChunk, _attempt: int) -> MapperReply:
        return reply(next(responses))

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    result = asyncio.run(
        run_map_reduce(
            rows(3),
            question="q",
            model="mock",
            mapper=mapper,
            checkpoint_path=tmp_path / "checkpoint.json",
            chunk_size=3,
            max_attempts=3,
            sleep=sleep,
        )
    )

    assert result.predicted_indices == (1, 2)
    assert result.calls == 3
    assert result.input_tokens == 30
    assert sleeps == [1.0, 2.0]


def test_non_transient_provider_error_is_not_retried(tmp_path: Path) -> None:
    class ForbiddenError(RuntimeError):
        status_code = 403

    calls = 0

    async def mapper(_chunk: CorpusChunk, _attempt: int) -> MapperReply:
        nonlocal calls
        calls += 1
        raise ForbiddenError("forbidden")

    with pytest.raises(PartialCoverageError) as captured:
        asyncio.run(
            run_map_reduce(
                rows(3),
                question="q",
                model="mock",
                mapper=mapper,
                checkpoint_path=tmp_path / "checkpoint.json",
                chunk_size=3,
            )
        )

    assert calls == 1
    assert isinstance(captured.value.__cause__, ForbiddenError)


def test_transient_provider_error_is_retried(tmp_path: Path) -> None:
    class ServiceError(RuntimeError):
        status_code = 500

    calls = 0

    async def mapper(_chunk: CorpusChunk, _attempt: int) -> MapperReply:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ServiceError("temporary")
        return reply("-1")

    result = asyncio.run(
        run_map_reduce(
            rows(3),
            question="q",
            model="mock",
            mapper=mapper,
            checkpoint_path=tmp_path / "checkpoint.json",
            chunk_size=3,
            sleep=lambda _delay: asyncio.sleep(0),
        )
    )

    assert calls == 2
    assert result.calls == 2


def test_rate_limit_retry_after_is_honored_and_bounded(tmp_path: Path) -> None:
    class Response:
        headers = {"Retry-After": "99"}

    class RateLimitError(RuntimeError):
        status_code = 429
        response = Response()

    calls = 0
    sleeps: list[float] = []

    async def mapper(_chunk: CorpusChunk, _attempt: int) -> MapperReply:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError("limited")
        return reply("-1")

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    asyncio.run(
        run_map_reduce(
            rows(3),
            question="q",
            model="mock",
            mapper=mapper,
            checkpoint_path=tmp_path / "checkpoint.json",
            chunk_size=3,
            sleep=sleep,
        )
    )

    assert calls == 2
    assert sleeps == [60.0]


def test_missing_usage_is_preserved_as_unknown_not_zero(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"

    async def mapper(_chunk: CorpusChunk, _attempt: int) -> MapperReply:
        return reply("-1", cost=None)

    result = asyncio.run(
        run_map_reduce(
            rows(4),
            question="q",
            model="mock",
            mapper=mapper,
            checkpoint_path=checkpoint,
            chunk_size=2,
        )
    )

    assert result.cost_usd is None
    assert result.cost_status == "unavailable"
    assert all(chunk.cost_usd is None for chunk in result.chunks)
    stored = json.loads(checkpoint.read_text(encoding="utf-8"))["completed"]
    assert stored["0"]["cost_usd"] is None
    assert stored["0"]["input_tokens"] == 10
    assert stored["0"]["output_tokens"] == 2
    assert stored["0"]["latency_seconds"] >= 0


def test_request_gate_runs_before_every_attempt(tmp_path: Path) -> None:
    gates = 0

    async def gate() -> None:
        nonlocal gates
        gates += 1

    async def mapper(_chunk: CorpusChunk, _attempt: int) -> MapperReply:
        return reply("-1")

    result = asyncio.run(
        run_map_reduce(
            rows(5),
            question="q",
            model="mock",
            mapper=mapper,
            checkpoint_path=tmp_path / "checkpoint.json",
            chunk_size=2,
            request_gate=gate,
        )
    )

    assert gates == result.calls == 3


def test_parallelism_is_bounded(tmp_path: Path) -> None:
    active = 0
    peak = 0

    async def mapper(_chunk: CorpusChunk, _attempt: int) -> MapperReply:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return reply("-1")

    asyncio.run(
        run_map_reduce(
            rows(10),
            question="q",
            model="mock",
            mapper=mapper,
            checkpoint_path=tmp_path / "checkpoint.json",
            chunk_size=2,
            max_parallel=2,
        )
    )

    assert peak == 2


def test_partial_coverage_resumes_only_missing_chunks(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    first_calls: list[int] = []

    async def failing_mapper(chunk: CorpusChunk, _attempt: int) -> MapperReply:
        first_calls.append(chunk.chunk_id)
        if chunk.chunk_id == 1:
            raise RuntimeError("stop")
        return reply(str(min(chunk.indices)))

    with pytest.raises(PartialCoverageError):
        asyncio.run(
            run_map_reduce(
                rows(6),
                question="q",
                model="mock",
                mapper=failing_mapper,
                checkpoint_path=checkpoint,
                chunk_size=2,
                max_parallel=1,
            )
        )
    assert first_calls == [0, 1]

    resumed_calls: list[int] = []

    async def resumed_mapper(chunk: CorpusChunk, _attempt: int) -> MapperReply:
        resumed_calls.append(chunk.chunk_id)
        return reply(str(min(chunk.indices)))

    result = asyncio.run(
        run_map_reduce(
            rows(6),
            question="q",
            model="mock",
            mapper=resumed_mapper,
            checkpoint_path=checkpoint,
            chunk_size=2,
            max_parallel=1,
        )
    )

    assert result.resumed_chunks == 1
    assert resumed_calls == [1, 2]
    assert result.predicted_indices == (0, 2, 4)

    async def must_not_run(_chunk: CorpusChunk, _attempt: int) -> MapperReply:
        raise AssertionError("completed checkpoint was not idempotent")

    repeated = asyncio.run(
        run_map_reduce(
            rows(6),
            question="q",
            model="mock",
            mapper=must_not_run,
            checkpoint_path=checkpoint,
            chunk_size=2,
        )
    )
    assert repeated.resumed_chunks == 3
    assert repeated.predicted_indices == result.predicted_indices


def test_checkpoint_identity_rejects_changed_question(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"

    async def mapper(_chunk: CorpusChunk, _attempt: int) -> MapperReply:
        return reply("-1")

    asyncio.run(
        run_map_reduce(
            rows(2),
            question="first",
            model="mock",
            mapper=mapper,
            checkpoint_path=checkpoint,
            chunk_size=2,
        )
    )
    with pytest.raises(MapReduceError, match="does not match"):
        asyncio.run(
            run_map_reduce(
                rows(2),
                question="changed",
                model="mock",
                mapper=mapper,
                checkpoint_path=checkpoint,
                chunk_size=2,
            )
        )


def test_oracle_mapper_reaches_f1_one(tmp_path: Path) -> None:
    truth = {1, 4, 7}

    async def mapper(chunk: CorpusChunk, _attempt: int) -> MapperReply:
        local = sorted(truth & chunk.indices)
        return reply(",".join(map(str, local)) if local else "-1")

    result = asyncio.run(
        run_map_reduce(
            rows(9),
            question="q",
            model="oracle",
            mapper=mapper,
            checkpoint_path=tmp_path / "checkpoint.json",
            chunk_size=2,
        )
    )
    score = score_predictions(
        result.predicted_indices,
        truth,
        invalid_claims=result.invalid_claims,
    )

    assert score.f1 == 1
    assert score.exact_match
