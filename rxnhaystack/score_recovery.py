"""Exact score recovery helpers for versioned RxnHaystack ground-truth changes."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_QUESTION_IDS: dict[str, tuple[str, ...]] = {
    "tier3/task6": (
        "rxh-t3-task6-acyl-chloride-with-primary-amine",
        "rxh-t3-task6-carboxylic-acid-with-primary-amine",
        "rxh-t3-task6-ester-with-primary-amine",
        "rxh-t3-task6-ester-with-secondary-amine",
    ),
    "tier3/task7": (
        "rxh-t3-task7-alcohol-to-azide",
        "rxh-t3-task7-alcohol-to-carboxylic-acid",
        "rxh-t3-task7-grignard-ketone-to-tertiary-alcohol",
        "rxh-t3-task7-nitrile-to-amine",
        "rxh-t3-task7-nitro-groups-to-amines",
    ),
    "tier3/task10": (
        "rxh-t3-task10-ester-hydrolysis-deprotection-with-oh",
        "rxh-t3-task10-mitsunobu-reaction-family",
        "rxh-t3-task10-wittig-olefination",
        "rxh-t3-task10-knoevenagel-aldol-condensation",
        "rxh-t3-task10-azide-alkyne-huisgen-cycloaddition",
    ),
    "tier3/task18": ("rxh-t3-task18",),
    "tier3/task23": ("rxh-t3-task23",),
}

QUESTION_LOG_LABELS: dict[str, str] = {
    question_id: "-".join(question_id.split("-")[3:]).replace("-", "_")
    for question_ids in TASK_QUESTION_IDS.values()
    for question_id in question_ids
}
QUESTION_LOG_LABELS["rxh-t3-task18"] = "new_ring_construction"
QUESTION_LOG_LABELS["rxh-t3-task23"] = "stereocenter_from_achiral_reactants"
LOG_LABEL_QUESTION_IDS = {label: question_id for question_id, label in QUESTION_LOG_LABELS.items()}

ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-_][0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
PREDICTED_COUNT_RE = re.compile(r"Predicted \[([^\]]+)\] count:\s*(\d+)")
METRICS_RE = re.compile(
    r"Metrics \[([^\]]+)\].*?precision=([0-9.]+)\s+recall=([0-9.]+)\s+"
    r"f1=([0-9.]+)\s+exact_match=(True|False)",
    re.DOTALL,
)


@dataclass(frozen=True)
class SampleMetric:
    label: str
    predicted_count: int
    precision: float
    recall: float
    f1: float
    exact_match: bool


@dataclass(frozen=True)
class ExtractedPrediction:
    question_id: str
    indices: tuple[int, ...]
    extraction_method: str


@dataclass(frozen=True)
class AggregateMetricRecovery:
    historical_true_positives: tuple[int, ...]
    corrected_true_positives: int
    corrected_precision: float
    corrected_recall: float
    corrected_f1: float


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_ground_truth(path: Path) -> dict[str, frozenset[int]]:
    output: dict[str, frozenset[int]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            output[str(payload["question_id"])] = frozenset(
                int(index) for index in payload.get("relevant_reaction_indices", ())
            )
    return output


def apply_corrected_score_recoveries(
    rows: list[dict[str, Any]], path: Path
) -> tuple[dict[str, Any], int]:
    """Restore exactly rescored rows after the correction invalidation pass.

    A recovery may be shared by the main and control tables, so entries absent
    from the table being built are intentionally ignored. Every entry that is
    present is checked against its task, metric, and preserved historical score.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    recoveries: dict[str, dict[str, Any]] = {}
    for recovery in payload.get("recoveries", ()):
        run_id = str(recovery["run_id"])
        if run_id in recoveries:
            raise ValueError(f"Duplicate corrected score recovery: {run_id}")
        recoveries[run_id] = recovery

    seen_rows: set[str] = set()
    applied = 0
    for row in rows:
        run_id = str(row["run_id"])
        if run_id in seen_rows:
            raise ValueError(f"Duplicate record run ID: {run_id}")
        seen_rows.add(run_id)
        recovery = recoveries.get(run_id)
        if recovery is None:
            continue
        if str(row["task"]) != str(recovery["task"]):
            raise ValueError(f"Corrected recovery task mismatch for {run_id}")
        score_name = row.get("score_name")
        if score_name and str(score_name) != str(recovery["score_name"]):
            raise ValueError(f"Corrected recovery metric mismatch for {run_id}")
        if row.get("score_correction_status") != "historical_score_invalidated":
            raise ValueError(f"Corrected recovery targets a non-invalidated row: {run_id}")
        historical = row.get("original_f1")
        if (
            historical in (None, "")
            or abs(float(historical) - float(recovery["historical_score"])) > 5e-4
        ):
            raise ValueError(f"Corrected recovery historical score mismatch for {run_id}")
        row["f1"] = float(recovery["score_value"])
        row["score_available"] = True
        row["score_correction_status"] = "corrected_exact_rescore"
        marker = f"corrected-score-recovery:{payload['recovery_id']}"
        row["sources"] = f"{row['sources']};{marker}"
        applied += 1
    return payload, applied


def parse_indices(value: str) -> tuple[int, ...]:
    candidate = value.strip().strip("│").strip()
    if candidate.replace(" ", "") == "-1":
        return ()
    seen: set[int] = set()
    output: list[int] = []
    for token in re.findall(r"-?\d+", candidate):
        index = int(token)
        if index >= 0 and index not in seen:
            seen.add(index)
            output.append(index)
    return tuple(output)


def precision_recall_f1(
    predicted: set[int] | frozenset[int], ground_truth: set[int] | frozenset[int]
) -> tuple[float, float, float]:
    true_positives = len(predicted & ground_truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def context_indices(
    *,
    dataset_size: int,
    context_size: int,
    correct_indices: set[int] | frozenset[int],
    rng: random.Random,
    min_selected_ground_truth: int = 1,
    positive_cardinality: int | None = None,
) -> frozenset[int]:
    """Reproduce ``RandomContextPipeline`` without materializing reaction strings."""

    valid_correct = sorted(index for index in correct_indices if 0 <= index < dataset_size)
    if positive_cardinality is not None:
        if positive_cardinality > len(valid_correct):
            raise ValueError("positive cardinality exceeds available ground truth")
        if context_size >= 0 and positive_cardinality > min(context_size, dataset_size):
            raise ValueError("positive cardinality exceeds context size")
        forced = rng.sample(valid_correct, k=positive_cardinality)
        excluded = set(valid_correct) - set(forced)
        if context_size < 0:
            return frozenset(index for index in range(dataset_size) if index not in excluded)
        return _sample_indices(
            dataset_size=dataset_size,
            top_k=context_size,
            forced=forced,
            excluded=excluded,
            rng=rng,
        )

    if context_size < 0:
        return frozenset(range(dataset_size))
    top_k = min(context_size, dataset_size)
    if top_k == 0:
        return frozenset()
    if not valid_correct:
        return _sample_indices(
            dataset_size=dataset_size,
            top_k=top_k,
            forced=[],
            excluded=set(),
            rng=rng,
        )
    forced_count = min(
        len(valid_correct),
        top_k // 2,
        max(min_selected_ground_truth, int((len(valid_correct) / dataset_size) * top_k)),
    )
    forced = rng.sample(valid_correct, k=forced_count)
    excluded = set(valid_correct) - set(forced)
    return _sample_indices(
        dataset_size=dataset_size,
        top_k=top_k,
        forced=forced,
        excluded=excluded,
        rng=rng,
    )


def _sample_indices(
    *,
    dataset_size: int,
    top_k: int,
    forced: list[int],
    excluded: set[int],
    rng: random.Random,
) -> frozenset[int]:
    deduplicated_forced = list(dict.fromkeys(forced))[:top_k]
    forced_set = set(deduplicated_forced)
    remainder = [
        index for index in range(dataset_size) if index not in forced_set and index not in excluded
    ]
    sampled = deduplicated_forced + rng.sample(
        remainder, k=min(top_k - len(deduplicated_forced), len(remainder))
    )
    rng.shuffle(sampled)
    return frozenset(sampled)


def reconstruct_task_contexts(
    *,
    task: str,
    context_size: int,
    historical_ground_truth: dict[str, frozenset[int]],
    dataset_size: int,
    seed: int = 42,
    min_selected_ground_truth: int = 1,
    positive_cardinality: int | None = None,
) -> dict[str, frozenset[int]]:
    rng = random.Random(seed)
    output: dict[str, frozenset[int]] = {}
    for question_id in TASK_QUESTION_IDS[task]:
        output[question_id] = context_indices(
            dataset_size=dataset_size,
            context_size=context_size,
            correct_indices=historical_ground_truth[question_id],
            rng=rng,
            min_selected_ground_truth=min_selected_ground_truth,
            positive_cardinality=positive_cardinality,
        )
    return output


def historical_min_selected_ground_truth(*, task: str, method: str) -> int:
    """Return the sampler setting used by the original task runner.

    Most Tier-3 runners explicitly requested five positives. The direct-LLM
    Task-6 runner and the direct-LLM/RLM Task-18 runners retained the context
    pipeline's default of one.
    """

    if task == "tier3/task6" and method == "llm":
        return 1
    if task == "tier3/task18" and method in {"llm", "rlm"}:
        return 1
    return 5


def parse_sample_metrics(log_text: str) -> dict[str, SampleMetric]:
    cleaned = ANSI_ESCAPE_RE.sub("", log_text)
    counts = [
        (match.start(), match.group(1), int(match.group(2)))
        for match in PREDICTED_COUNT_RE.finditer(cleaned)
    ]
    output: dict[str, SampleMetric] = {}
    for match in METRICS_RE.finditer(cleaned):
        label = match.group(1)
        preceding = [item for item in counts if item[0] < match.start() and item[1] == label]
        if not preceding:
            continue
        output[label] = SampleMetric(
            label=label,
            predicted_count=preceding[-1][2],
            precision=float(match.group(2)),
            recall=float(match.group(3)),
            f1=float(match.group(4)),
            exact_match=match.group(5) == "True",
        )
    return output


def extract_tier3_predictions(
    log_text: str, *, task: str, method: str
) -> dict[str, ExtractedPrediction]:
    cleaned = ANSI_ESCAPE_RE.sub("", log_text)
    count_matches = list(PREDICTED_COUNT_RE.finditer(cleaned))
    output: dict[str, ExtractedPrediction] = {}
    previous_end = 0
    for count_match in count_matches:
        label = count_match.group(1)
        question_id = LOG_LABEL_QUESTION_IDS.get(label)
        segment = cleaned[previous_end : count_match.start()]
        previous_end = count_match.end()
        if question_id not in TASK_QUESTION_IDS.get(task, ()):
            continue
        expected_count = int(count_match.group(2))
        candidates = (
            _codeact_candidates(segment, task=task)
            if method == "codeact"
            else _rlm_candidates(segment)
        )
        selected: tuple[int, ...] | None = None
        selected_method = ""
        for extraction_method, candidate in reversed(candidates):
            parsed = parse_indices(candidate)
            if len(set(parsed)) == expected_count:
                selected = parsed
                selected_method = extraction_method
                break
        if selected is not None:
            output[question_id] = ExtractedPrediction(
                question_id=question_id,
                indices=selected,
                extraction_method=selected_method,
            )
    return output


def _codeact_candidates(segment: str, *, task: str) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    output.extend(
        ("codeact-answer-tag", match.group(1))
        for match in re.finditer(
            r"<answer>(.*?)</answer>", segment, flags=re.IGNORECASE | re.DOTALL
        )
    )
    output.extend(
        ("codeact-answer-line", match.group(1))
        for match in re.finditer(r"ANSWER:\s*([^\r\n]+)", segment, flags=re.IGNORECASE)
    )
    output.extend(_final_answer_box_candidates(segment))
    iteration_start = segment.rfind("===== ITERATION 1 =====")
    finish_start = segment.rfind("---- FINISH REASON:")
    if 0 <= iteration_start < finish_start:
        transcript = segment[iteration_start:finish_start]
        historical_answer = re.search(
            r"ANSWER:\s*(.*)", transcript, flags=re.IGNORECASE | re.DOTALL
        )
        if historical_answer:
            output.append(("codeact-historical-transcript", historical_answer.group(1)))
        elif task == "tier3/task18":
            output.extend(
                ("codeact-final-code-block", match.group(1))
                for match in re.finditer(
                    r"```(?:python)?\s*\n(.*?)```",
                    transcript,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            )
    return sorted(output, key=lambda item: segment.rfind(item[1]))


def _rlm_candidates(segment: str) -> list[tuple[str, str]]:
    output = _final_answer_box_candidates(segment)
    output.extend(
        ("rlm-final-call", match.group(1))
        for match in re.finditer(r"\bFINAL\s*\((.*?)\)", segment, flags=re.DOTALL)
    )
    return output


def _final_answer_box_candidates(segment: str) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for match in re.finditer(
        r"[★*]\s*Final Answer[^\n]*\n(.*?)(?:\n╰[^\n]*|\Z)",
        segment,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        lines: list[str] = []
        for line in match.group(1).splitlines():
            stripped = line.strip().strip("│").strip()
            if stripped:
                lines.append(stripped)
        joined = " ".join(lines)
        wrapped = "".join(lines)
        output.append(("rlm-final-answer-box", joined))
        output.append(("rlm-final-answer-box-wrapped", wrapped))
        for method, candidate in (
            ("rlm-final-answer-box-historical-answer", joined),
            ("rlm-final-answer-box-wrapped-historical-answer", wrapped),
        ):
            historical_answer = re.search(
                r"ANSWER:\s*(.*)", candidate, flags=re.IGNORECASE | re.DOTALL
            )
            if historical_answer:
                output.append((method, historical_answer.group(1)))
    return output


def infer_prediction(
    *,
    metric: SampleMetric,
    historical_ground_truth_in_context: frozenset[int],
    corrected_ground_truth_in_context: frozenset[int],
) -> tuple[frozenset[int] | None, str]:
    """Recover a prediction only when aggregate evidence determines it exactly."""

    if historical_ground_truth_in_context == corrected_ground_truth_in_context:
        return None, "ground-truth-unchanged-in-context"
    if metric.predicted_count == 0:
        return frozenset(), "empty-prediction-from-count"
    if metric.exact_match:
        return historical_ground_truth_in_context, "prediction-equals-historical-ground-truth"
    return None, "prediction-underdetermined"


def infer_metric_from_aggregate(
    *,
    metric: SampleMetric,
    historical_ground_truth_in_context: frozenset[int],
    corrected_ground_truth_in_context: frozenset[int],
) -> AggregateMetricRecovery | None:
    """Recover a corrected metric when aggregate evidence makes it unique.

    The submitted indices need not be identifiable. We enumerate historical
    true-positive counts compatible with the four-decimal console metrics,
    then bound how many of those predictions can lie in the unchanged,
    removed, and added ground-truth partitions. A result is returned only if
    every compatible contingency table gives the same corrected TP count.

    Predictions outside the sampled context are conservatively treated as an
    unbounded pool of negatives because the historical evaluator did not
    reject out-of-context or out-of-range integers.
    """

    predicted_count = metric.predicted_count
    historical_count = len(historical_ground_truth_in_context)
    corrected_count = len(corrected_ground_truth_in_context)

    def matches_logged(value: float, logged: float) -> bool:
        return f"{value:.4f}" == f"{logged:.4f}"

    compatible_historical_tp: list[int] = []
    corrected_tp_options: set[int] = set()
    common_count = len(historical_ground_truth_in_context & corrected_ground_truth_in_context)
    removed_count = len(historical_ground_truth_in_context - corrected_ground_truth_in_context)
    added_count = len(corrected_ground_truth_in_context - historical_ground_truth_in_context)

    for historical_tp in range(min(predicted_count, historical_count) + 1):
        precision = historical_tp / predicted_count if predicted_count else 0.0
        recall = historical_tp / historical_count if historical_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if not (
            matches_logged(precision, metric.precision)
            and matches_logged(recall, metric.recall)
            and matches_logged(f1, metric.f1)
        ):
            continue
        compatible_historical_tp.append(historical_tp)

        common_tp_min = max(0, historical_tp - removed_count)
        common_tp_max = min(historical_tp, common_count)
        added_tp_max = min(predicted_count - historical_tp, added_count)
        for common_tp in range(common_tp_min, common_tp_max + 1):
            corrected_tp_options.update(range(common_tp, common_tp + added_tp_max + 1))

    if not compatible_historical_tp or len(corrected_tp_options) != 1:
        return None

    corrected_tp = next(iter(corrected_tp_options))
    precision = corrected_tp / predicted_count if predicted_count else 0.0
    recall = corrected_tp / corrected_count if corrected_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return AggregateMetricRecovery(
        historical_true_positives=tuple(compatible_historical_tp),
        corrected_true_positives=corrected_tp,
        corrected_precision=precision,
        corrected_recall=recall,
        corrected_f1=f1,
    )


def aggregate_question_score(
    *,
    question_id: str,
    metric: SampleMetric,
    historical_ground_truth_in_context: frozenset[int],
    corrected_ground_truth_in_context: frozenset[int],
) -> dict[str, Any] | None:
    aggregate = infer_metric_from_aggregate(
        metric=metric,
        historical_ground_truth_in_context=historical_ground_truth_in_context,
        corrected_ground_truth_in_context=corrected_ground_truth_in_context,
    )
    if aggregate is None:
        return None
    return {
        "question_id": question_id,
        "old_f1": metric.f1,
        "corrected_f1": aggregate.corrected_f1,
        "recovery": "aggregate-contingency-bounds",
        "predicted_count": metric.predicted_count,
        "historical_ground_truth_count": len(historical_ground_truth_in_context),
        "corrected_ground_truth_count": len(corrected_ground_truth_in_context),
        "historical_true_positive_options": list(aggregate.historical_true_positives),
        "corrected_true_positives": aggregate.corrected_true_positives,
        "corrected_precision": aggregate.corrected_precision,
        "corrected_recall": aggregate.corrected_recall,
        "validation_old_precision": metric.precision,
        "validation_old_recall": metric.recall,
    }


def positive_cardinality_from_condition(condition: str) -> int | None:
    match = re.search(r"(?:^|-)k(\d+)(?:-|$)", condition)
    return int(match.group(1)) if match else None


def context_size_from_label(value: str) -> int:
    return -1 if value == "full" else int(value)


def score_tier3_run(
    *,
    task: str,
    context_by_question: dict[str, frozenset[int]],
    historical_ground_truth: dict[str, frozenset[int]],
    corrected_ground_truth: dict[str, frozenset[int]],
    metrics: dict[str, SampleMetric],
    extracted: dict[str, ExtractedPrediction],
) -> tuple[float | None, list[dict[str, Any]]]:
    question_scores: list[dict[str, Any]] = []
    for question_id in TASK_QUESTION_IDS[task]:
        label = QUESTION_LOG_LABELS[question_id]
        metric = metrics.get(label)
        if metric is None:
            return None, question_scores
        context = context_by_question[question_id]
        historical_in_context = frozenset(historical_ground_truth[question_id] & context)
        corrected_in_context = frozenset(corrected_ground_truth[question_id] & context)

        prediction = extracted.get(question_id)
        source = prediction.extraction_method if prediction is not None else ""
        if prediction is not None:
            predicted = frozenset(prediction.indices)
        elif historical_in_context == corrected_in_context:
            question_scores.append(
                {
                    "question_id": question_id,
                    "old_f1": metric.f1,
                    "corrected_f1": metric.f1,
                    "recovery": "ground-truth-unchanged-in-context",
                    "historical_ground_truth_count": len(historical_in_context),
                    "corrected_ground_truth_count": len(corrected_in_context),
                }
            )
            continue
        else:
            predicted, source = infer_prediction(
                metric=metric,
                historical_ground_truth_in_context=historical_in_context,
                corrected_ground_truth_in_context=corrected_in_context,
            )
            if predicted is None:
                aggregate_score = aggregate_question_score(
                    question_id=question_id,
                    metric=metric,
                    historical_ground_truth_in_context=historical_in_context,
                    corrected_ground_truth_in_context=corrected_in_context,
                )
                if aggregate_score is not None:
                    question_scores.append(aggregate_score)
                    continue
                return None, question_scores
        old_precision, old_recall, old_f1 = precision_recall_f1(predicted, historical_in_context)
        if abs(old_f1 - metric.f1) > 5e-4:
            inferred, inferred_source = infer_prediction(
                metric=metric,
                historical_ground_truth_in_context=historical_in_context,
                corrected_ground_truth_in_context=corrected_in_context,
            )
            if inferred is None:
                aggregate_score = aggregate_question_score(
                    question_id=question_id,
                    metric=metric,
                    historical_ground_truth_in_context=historical_in_context,
                    corrected_ground_truth_in_context=corrected_in_context,
                )
                if aggregate_score is not None:
                    question_scores.append(aggregate_score)
                    continue
                return None, question_scores
            predicted = inferred
            source = inferred_source
            old_precision, old_recall, old_f1 = precision_recall_f1(
                predicted, historical_in_context
            )
            if abs(old_f1 - metric.f1) > 5e-4:
                return None, question_scores
        precision, recall, f1 = precision_recall_f1(predicted, corrected_in_context)
        question_scores.append(
            {
                "question_id": question_id,
                "old_f1": old_f1,
                "corrected_f1": f1,
                "recovery": source,
                "predicted_count": len(predicted),
                "historical_ground_truth_count": len(historical_in_context),
                "corrected_ground_truth_count": len(corrected_in_context),
                "corrected_precision": precision,
                "corrected_recall": recall,
                "validation_old_precision": old_precision,
                "validation_old_recall": old_recall,
            }
        )
    return sum(item["corrected_f1"] for item in question_scores) / len(
        question_scores
    ), question_scores
