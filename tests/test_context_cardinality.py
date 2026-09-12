from __future__ import annotations

import random

import pytest

from rlm.codeact_helpers import RandomContextPipeline, build_context_pipeline


def _indices(context: str) -> set[int]:
    return {int(line.split(" ", 1)[0]) for line in context.splitlines()}


def test_matched_context_has_exact_size_and_positive_cardinality() -> None:
    lines = [f"{index} reaction-{index}" for index in range(20)]
    positives = {1, 3, 5, 7, 9}
    pipeline = RandomContextPipeline(
        lines,
        random.Random(42),
        positive_cardinality=2,
    )

    selected = _indices(pipeline.build_context(context_size=10, correct_indices=positives))

    assert len(selected) == 10
    assert len(selected & positives) == 2


def test_zero_cardinality_excludes_every_positive() -> None:
    lines = [f"{index} reaction-{index}" for index in range(10)]
    positives = {1, 3, 5}
    pipeline = RandomContextPipeline(
        lines,
        random.Random(42),
        positive_cardinality=0,
    )

    selected = _indices(pipeline.build_context(context_size=6, correct_indices=positives))

    assert len(selected) == 6
    assert selected.isdisjoint(positives)


def test_full_matched_context_retains_all_negatives_and_exact_positives() -> None:
    lines = [f"{index} reaction-{index}" for index in range(10)]
    positives = {1, 3, 5}
    pipeline = RandomContextPipeline(
        lines,
        random.Random(42),
        positive_cardinality=1,
    )

    selected = _indices(pipeline.build_context(context_size=-1, correct_indices=positives))

    assert len(selected) == 8
    assert len(selected & positives) == 1
    assert selected >= set(range(10)) - positives


@pytest.mark.parametrize(
    ("context_size", "cardinality", "message"),
    [(10, 4, "only 3 are available"), (2, 3, "context of size 2")],
)
def test_impossible_cardinality_fails_loudly(
    context_size: int,
    cardinality: int,
    message: str,
) -> None:
    pipeline = RandomContextPipeline(
        [f"{index} reaction-{index}" for index in range(10)],
        random.Random(42),
        positive_cardinality=cardinality,
    )

    with pytest.raises(ValueError, match=message):
        pipeline.build_context(context_size=context_size, correct_indices={1, 3, 5})


def test_factory_reads_campaign_cardinality(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_POSITIVE_CARDINALITY", "2")

    pipeline = build_context_pipeline(
        "random",
        [f"{index} reaction-{index}" for index in range(10)],
        random.Random(42),
    )

    assert isinstance(pipeline, RandomContextPipeline)
    assert pipeline.positive_cardinality == 2


def test_baseline_sampler_does_not_read_cardinality_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RXNHAYSTACK_POSITIVE_CARDINALITY", raising=False)
    pipeline = build_context_pipeline(
        "random",
        [f"{index} reaction-{index}" for index in range(10)],
        random.Random(42),
    )

    assert isinstance(pipeline, RandomContextPipeline)
    assert pipeline.positive_cardinality is None
