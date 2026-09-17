from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _stable_id(prefix: str, *parts: object) -> str:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _load_prediction_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("task") != "tier4/task16":
        raise ValueError(f"Not a Task-16 prediction artifact: {path}")
    if not isinstance(payload.get("questions"), list):
        raise ValueError(f"Prediction artifact has no questions list: {path}")
    return payload


def build_candidate_pack(
    prediction_paths: Iterable[Path],
    *,
    pack_id: str,
    version: str,
    seed: int,
    max_false_positives: int | None = None,
    positive_controls_per_question: int = 1,
) -> dict[str, Any]:
    """Build a blinded-import candidate pack from structured Task-16 artifacts."""
    false_positives: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    positives: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for path in sorted(prediction_paths):
        payload = _load_prediction_artifact(path)
        run_id = str(payload.get("run_id") or path.parent.name)
        source = {
            "model_id": payload.get("model"),
            "run_id": run_id,
            "prompting_method": payload.get("prompting_method"),
            "prompt_condition": payload.get("prompt_condition"),
        }
        for question in payload["questions"]:
            question_id = str(question["canonical_question_id"])
            for chain_value in question.get("false_positive_chains", []):
                chain = tuple(int(value) for value in chain_value)
                key = (question_id, chain)
                false_positives.setdefault(
                    key,
                    {
                        "candidate_id": _stable_id("fp", question_id, chain),
                        "question_id": question_id,
                        "candidate_answer": [list(chain)],
                        "reaction_indices": list(chain),
                        "evaluator_outcome": "incorrect",
                        "control_type": None,
                        **source,
                    },
                )
            for chain_value in question.get("ground_truth_chains", []):
                chain = tuple(int(value) for value in chain_value)
                key = (question_id, chain)
                positives.setdefault(
                    key,
                    {
                        "candidate_id": _stable_id("positive", question_id, chain),
                        "question_id": question_id,
                        "candidate_answer": [list(chain)],
                        "reaction_indices": list(chain),
                        "evaluator_outcome": "correct",
                        "control_type": "positive",
                        "model_id": None,
                        "run_id": run_id,
                        "prompting_method": "ground_truth_control",
                        "prompt_condition": payload.get("prompt_condition"),
                    },
                )

    rng = random.Random(seed)
    false_values = sorted(false_positives.values(), key=lambda item: str(item["candidate_id"]))
    rng.shuffle(false_values)
    if max_false_positives is not None:
        false_values = false_values[:max_false_positives]

    positive_values: list[dict[str, Any]] = []
    by_question: dict[str, list[dict[str, Any]]] = {}
    for item in positives.values():
        by_question.setdefault(str(item["question_id"]), []).append(item)
    for question_id in sorted(by_question):
        choices = sorted(by_question[question_id], key=lambda item: str(item["candidate_id"]))
        rng.shuffle(choices)
        positive_values.extend(choices[:positive_controls_per_question])

    candidates = false_values + positive_values
    rng.shuffle(candidates)
    return {
        "schema_version": "1.1.0",
        "pack_id": pack_id,
        "version": version,
        "seed": seed,
        "candidates": candidates,
    }


def write_candidate_pack(payload: dict[str, Any], output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(output)
    return output
