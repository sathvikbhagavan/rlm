from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rxnhaystack.rate_limit import retry_after_seconds
from rxnhaystack.runtime import atomic_write_json

CHECKPOINT_VERSION = 1
MAP_PROTOCOL_VERSION = 1
DEFAULT_CHUNK_SIZE = 500
DEFAULT_MAX_ATTEMPTS = 3
INDEX_RESPONSE_PATTERN = re.compile(
    r"^(?:ANSWER\s*:\s*)?(?:-1|\d+(?:\s*,\s*\d+)*)\s*[.!]?\s*$",
    flags=re.IGNORECASE,
)


class MapReduceError(RuntimeError):
    """Base error for the fixed, non-recursive map-and-union baseline."""


class MalformedMapResponse(MapReduceError):
    """Raised when a mapper does not return the required index-only answer."""


class PartialCoverageError(MapReduceError):
    """Raised when at least one corpus chunk has no valid mapper result."""


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: int
    rows: tuple[str, ...]
    indices: frozenset[int]

    @property
    def text(self) -> str:
        return "\n".join(self.rows)


@dataclass(frozen=True)
class MapperReply:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("Mapper token counts must be non-negative")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("Mapper cost must be non-negative when available")


@dataclass(frozen=True)
class ChunkResult:
    chunk_id: int
    attempts: int
    valid_indices: tuple[int, ...]
    invalid_indices: tuple[int, ...]
    response_text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_seconds: float


@dataclass(frozen=True)
class SetMetrics:
    precision: float
    recall: float
    f1: float
    exact_match: bool
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class MapReduceResult:
    predicted_indices: tuple[int, ...]
    invalid_claims: int
    chunk_count: int
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    cost_status: str
    latency_seconds: float
    resumed_chunks: int
    chunks: tuple[ChunkResult, ...]


Mapper = Callable[[CorpusChunk, int], Awaitable[MapperReply]]
RequestGate = Callable[[], Awaitable[None]]
Sleep = Callable[[float], Awaitable[None]]


def parse_index_response(text: str) -> tuple[int, ...]:
    cleaned = text.strip()
    if not cleaned:
        raise MalformedMapResponse("Mapper returned an empty response")
    if INDEX_RESPONSE_PATTERN.fullmatch(cleaned) is None:
        raise MalformedMapResponse("Mapper response was not an index-only answer")
    answer = re.sub(r"^ANSWER\s*:\s*", "", cleaned, flags=re.IGNORECASE).rstrip(".! ")
    if answer == "-1":
        return ()
    return tuple(dict.fromkeys(int(value.strip()) for value in answer.split(",")))


def build_chunks(rows: Sequence[str], *, chunk_size: int) -> tuple[CorpusChunk, ...]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not rows:
        raise ValueError("Corpus must contain at least one indexed reaction")
    parsed: list[tuple[int, str]] = []
    seen: set[int] = set()
    for position, row in enumerate(rows):
        try:
            raw_index, _ = row.split(" ", 1)
            index = int(raw_index)
        except (ValueError, AttributeError) as error:
            raise ValueError(f"Corpus row {position} is not an indexed reaction") from error
        if index in seen:
            raise ValueError(f"Duplicate corpus index: {index}")
        seen.add(index)
        parsed.append((index, row))
    return tuple(
        CorpusChunk(
            chunk_id=chunk_id,
            rows=tuple(row for _, row in parsed[start : start + chunk_size]),
            indices=frozenset(index for index, _ in parsed[start : start + chunk_size]),
        )
        for chunk_id, start in enumerate(range(0, len(parsed), chunk_size))
    )


def score_predictions(
    predicted_indices: Sequence[int],
    ground_truth_indices: set[int],
    *,
    invalid_claims: int,
) -> SetMetrics:
    if invalid_claims < 0:
        raise ValueError("invalid_claims must be non-negative")
    predicted = set(predicted_indices)
    true_positives = len(predicted & ground_truth_indices)
    false_positives = len(predicted - ground_truth_indices) + invalid_claims
    false_negatives = len(ground_truth_indices - predicted)
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SetMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=predicted == ground_truth_indices and invalid_claims == 0,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def corpus_fingerprint(rows: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def experiment_fingerprint(
    rows: Sequence[str], *, question: str, chunk_size: int, model: str
) -> str:
    payload = {
        "chunk_size": chunk_size,
        "corpus_sha256": corpus_fingerprint(rows),
        "map_protocol_version": MAP_PROTOCOL_VERSION,
        "model": model,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def provider_status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if status is None:
        return None
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def retryable_provider_error(error: Exception) -> bool:
    code = provider_status_code(error)
    if code is None:
        return isinstance(error, (TimeoutError, ConnectionError))
    return code in {408, 409, 429} or 500 <= code <= 599


async def run_map_reduce(
    rows: Sequence[str],
    *,
    question: str,
    model: str,
    mapper: Mapper,
    checkpoint_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_parallel: int = 1,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    request_gate: RequestGate | None = None,
    sleep: Sleep = asyncio.sleep,
) -> MapReduceResult:
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    chunks = build_chunks(rows, chunk_size=chunk_size)
    fingerprint = experiment_fingerprint(
        rows, question=question, chunk_size=chunk_size, model=model
    )
    completed = _load_checkpoint(checkpoint_path, expected_fingerprint=fingerprint)
    resumed_chunks = len(completed)
    checkpoint_lock = asyncio.Lock()
    started = time.monotonic()
    pending = asyncio.Queue[CorpusChunk]()
    for chunk in chunks:
        if chunk.chunk_id not in completed:
            pending.put_nowait(chunk)
    errors: list[BaseException] = []
    stop = asyncio.Event()

    async def worker() -> None:
        while not stop.is_set():
            try:
                chunk = pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                result = await _map_chunk(
                    chunk,
                    mapper=mapper,
                    max_attempts=max_attempts,
                    request_gate=request_gate,
                    sleep=sleep,
                )
            except BaseException as error:
                errors.append(error)
                stop.set()
                return
            async with checkpoint_lock:
                completed[chunk.chunk_id] = result
                _write_checkpoint(checkpoint_path, fingerprint=fingerprint, completed=completed)

    await asyncio.gather(*(worker() for _ in range(min(max_parallel, pending.qsize()))))
    missing = sorted(set(range(len(chunks))) - completed.keys())
    if errors or missing:
        detail = f"missing chunks={missing[:20]}" if missing else "mapper failure"
        cause = errors[0] if errors else None
        raise PartialCoverageError(f"Map phase did not cover the full corpus: {detail}") from cause

    ordered = tuple(completed[index] for index in range(len(chunks)))
    predicted = tuple(sorted({index for result in ordered for index in result.valid_indices}))
    invalid_claims = sum(len(result.invalid_indices) for result in ordered)
    known_costs = [result.cost_usd for result in ordered if result.cost_usd is not None]
    all_costs_known = len(known_costs) == len(ordered)
    return MapReduceResult(
        predicted_indices=predicted,
        invalid_claims=invalid_claims,
        chunk_count=len(chunks),
        calls=sum(result.attempts for result in ordered),
        input_tokens=sum(result.input_tokens for result in ordered),
        output_tokens=sum(result.output_tokens for result in ordered),
        cost_usd=sum(known_costs) if all_costs_known else None,
        cost_status="available" if all_costs_known else "unavailable",
        latency_seconds=time.monotonic() - started,
        resumed_chunks=resumed_chunks,
        chunks=ordered,
    )


async def _map_chunk(
    chunk: CorpusChunk,
    *,
    mapper: Mapper,
    max_attempts: int,
    request_gate: RequestGate | None,
    sleep: Sleep,
) -> ChunkResult:
    started = time.monotonic()
    cumulative_input = 0
    cumulative_output = 0
    cumulative_cost = 0.0
    all_costs_known = True
    for attempt in range(1, max_attempts + 1):
        if request_gate is not None:
            await request_gate()
        try:
            reply = await mapper(chunk, attempt)
            cumulative_input += reply.input_tokens
            cumulative_output += reply.output_tokens
            if reply.cost_usd is None:
                all_costs_known = False
            else:
                cumulative_cost += reply.cost_usd
            parsed = parse_index_response(reply.text)
        except MalformedMapResponse:
            if attempt == max_attempts:
                raise
            await sleep(float(attempt))
            continue
        except Exception as error:
            if attempt == max_attempts or not retryable_provider_error(error):
                raise
            delay = (
                min(60.0, retry_after_seconds(error, default=float(attempt)))
                if provider_status_code(error) == 429
                else float(attempt)
            )
            await sleep(delay)
            continue
        valid = tuple(sorted(index for index in parsed if index in chunk.indices))
        invalid = tuple(sorted(index for index in parsed if index not in chunk.indices))
        return ChunkResult(
            chunk_id=chunk.chunk_id,
            attempts=attempt,
            valid_indices=valid,
            invalid_indices=invalid,
            response_text=reply.text,
            input_tokens=cumulative_input,
            output_tokens=cumulative_output,
            cost_usd=cumulative_cost if all_costs_known else None,
            latency_seconds=time.monotonic() - started,
        )
    raise AssertionError("unreachable")


def _load_checkpoint(path: Path, *, expected_fingerprint: str) -> dict[int, ChunkResult]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != CHECKPOINT_VERSION:
        raise MapReduceError(f"Unsupported checkpoint version in {path}")
    if data.get("experiment_fingerprint") != expected_fingerprint:
        raise MapReduceError(f"Checkpoint does not match this experiment: {path}")
    raw_completed = data.get("completed")
    if not isinstance(raw_completed, dict):
        raise MapReduceError(f"Malformed checkpoint: {path}")
    return {int(chunk_id): ChunkResult(**payload) for chunk_id, payload in raw_completed.items()}


def _write_checkpoint(path: Path, *, fingerprint: str, completed: dict[int, ChunkResult]) -> None:
    atomic_write_json(
        path,
        {
            "version": CHECKPOINT_VERSION,
            "experiment_fingerprint": fingerprint,
            "completed": {
                str(chunk_id): asdict(result) for chunk_id, result in sorted(completed.items())
            },
        },
    )
