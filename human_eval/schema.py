from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from . import SCHEMA_VERSION

ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]+$")
ANSWER_TYPES = {"index_set", "reaction_chains", "reaction_pair_set", "smiles_set", "single_chain"}


@dataclass(frozen=True)
class Question:
    question_id: str
    tier: int
    category: str
    subcategory: str
    canonical_prompt: str
    answer_type: str
    source_files: tuple[str, ...]
    dataset_sha256: str
    ground_truth_ref: str
    suggested_time_minutes: int = 30
    schema_version: str = SCHEMA_VERSION
    scoring: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not ID_RE.fullmatch(self.question_id):
            raise ValueError(f"Invalid question_id: {self.question_id!r}")
        if self.tier not in {1, 2, 3, 4}:
            raise ValueError(f"Invalid tier for {self.question_id}: {self.tier}")
        if self.answer_type not in ANSWER_TYPES:
            raise ValueError(f"Invalid answer type for {self.question_id}: {self.answer_type}")
        if not self.canonical_prompt.strip() or not self.source_files:
            raise ValueError(f"Question {self.question_id} is missing source material")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Question:
        payload = dict(value)
        payload["source_files"] = tuple(payload["source_files"])
        question = cls(**payload)
        question.validate()
        return question


@dataclass(frozen=True)
class GroundTruth:
    ground_truth_ref: str
    question_id: str
    representation: Any
    relevant_reaction_indices: tuple[int, ...]
    evaluator: dict[str, Any]
    source_files: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def normalize_structured_answer(raw: str) -> tuple[str, list[list[str]]]:
    """Conservatively parse lines/commas while retaining the exact raw submission."""
    entries: list[list[str]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        entries.append([part.strip() for part in stripped.split(",") if part.strip()])
    return raw, entries
