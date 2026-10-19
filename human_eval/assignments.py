from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def assign_questions(
    questions: Iterable[dict[str, Any]],
    *,
    seed: int,
    quotas: dict[str, int] | None = None,
    annotator_id: str = "shared",
) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        grouped[str(question["category"])].append(str(question["question_id"]))
    rng = random.Random(stable_seed(seed, annotator_id))
    selected: list[str] = []
    for category in sorted(grouped):
        values = sorted(grouped[category])
        rng.shuffle(values)
        quota = len(values) if quotas is None else max(0, quotas.get(category, 0))
        selected.extend(values[:quota])
    rng.shuffle(selected)
    return selected


def resolve_assignment(
    questions: Iterable[dict[str, Any]],
    *,
    seed: int,
    annotator_id: str,
    quotas: dict[str, int] | None = None,
    question_ids: Iterable[str] | None = None,
) -> list[str]:
    """Resolve an explicit ordered assignment or a deterministic quota sample."""
    values = list(questions)
    if question_ids is None:
        return assign_questions(values, seed=seed, quotas=quotas, annotator_id=annotator_id)
    selected = [str(question_id) for question_id in question_ids]
    if len(selected) != len(set(selected)):
        raise ValueError("Explicit study assignments must not contain duplicate question IDs")
    available = {str(question["question_id"]) for question in values}
    unknown = sorted(set(selected) - available)
    if unknown:
        raise ValueError(f"Explicit study assignment contains unknown question IDs: {unknown}")
    return selected


def blinded_order(item_ids: Iterable[str], *, seed: int, annotator_id: str) -> list[str]:
    values = sorted(item_ids)
    random.Random(stable_seed(seed, f"candidate:{annotator_id}")).shuffle(values)
    return values
