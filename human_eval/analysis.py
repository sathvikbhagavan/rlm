from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .schema import ground_truth_answer_set, normalize_structured_answer, submitted_answer_set


def set_scores(predicted: set[Any], expected: set[Any]) -> dict[str, float]:
    if not predicted and not expected:
        return {"exact_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    overlap = len(predicted & expected)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_match": float(predicted == expected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def answer_scores(
    predicted: set[tuple[str, ...]],
    expected: set[tuple[str, ...]],
    answer_type: str,
) -> dict[str, float]:
    """Score exhaustive sets or a single accepted alternative as declared by the task."""
    if answer_type == "one_of_reaction_chains":
        accepted = len(predicted) == 1 and predicted.issubset(expected)
        score = float(accepted)
        return {"exact_match": score, "precision": score, "recall": score, "f1": score}
    return set_scores(predicted, expected)


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    pairs = [(a, b) for a, b in zip(labels_a, labels_b, strict=True) if a and b]
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(ca[x] * cb[x] for x in set(ca) | set(cb)) / (len(pairs) ** 2)
    return (
        1.0
        if expected == 1 and observed == 1
        else ((observed - expected) / (1 - expected) if expected < 1 else None)
    )


def krippendorff_alpha_nominal(ratings: dict[str, list[str]]) -> float | None:
    usable = {
        item: [x for x in values if x]
        for item, values in ratings.items()
        if len([x for x in values if x]) >= 2
    }
    if not usable:
        return None
    disagreements = total_pairs = 0
    counts: Counter[str] = Counter()
    for values in usable.values():
        counts.update(values)
        for i, left in enumerate(values):
            for right in values[i + 1 :]:
                total_pairs += 1
                disagreements += left != right
    observed = disagreements / total_pairs
    n = sum(counts.values())
    expected = 1 - sum((count / n) ** 2 for count in counts.values())
    return (
        1.0 if expected == 0 and observed == 0 else (1 - observed / expected if expected else None)
    )


def load_export(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if not {"manifest.json", "annotations.jsonl", "timing.jsonl"} <= names:
            raise ValueError(f"Not a RxnHaystack export: {path}")
        manifest = json.loads(archive.read("manifest.json"))
        annotations = [
            json.loads(line)
            for line in archive.read("annotations.jsonl").decode().splitlines()
            if line
        ]
        timing = [
            json.loads(line) for line in archive.read("timing.jsonl").decode().splitlines() if line
        ]
    return manifest, annotations, timing


def analyze(
    export_paths: list[Path],
    questions_path: Path,
    ground_truth_path: Path,
    output: Path,
    unblinding_paths: list[Path] | None = None,
) -> dict[str, Any]:
    questions = {x["question_id"]: x for x in read_jsonl(questions_path)}
    truth = {x["question_id"]: x for x in read_jsonl(ground_truth_path)}
    rows: list[dict[str, Any]] = []
    prospective_ratings: dict[str, list[str]] = defaultdict(list)
    agreement_ratings: dict[str, list[str]] = defaultdict(list)
    issue_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    time_by_user_item: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"active": 0.0, "wall": 0.0}
    )
    for export_path in export_paths:
        manifest, annotations, timing = load_export(export_path)
        user = manifest["anonymous_annotator_id"]
        for session in timing:
            key = (user, session["mode"], session["item_id"])
            time_by_user_item[key]["active"] += float(session["active_seconds"])
            time_by_user_item[key]["wall"] += float(session["wall_seconds"])
        for ann in annotations:
            payload = (
                ann.get("first_submission_payload")
                if ann["mode"] == "baseline" and ann.get("first_submission_payload")
                else ann["payload"]
            )
            item_id = ann["item_id"]
            mode = ann["mode"]
            base = {
                "annotator_id": user,
                "mode": mode,
                "item_id": item_id,
                "submitted": bool(ann.get("submitted_at")),
                "abstention": payload.get("abstention", ""),
                "confidence": payload.get("confidence"),
                "offline_minutes": payload.get("offline_minutes"),
                **time_by_user_item[(user, mode, item_id)],
            }
            for tool in payload.get("tools", []):
                tool_counts[str(tool)] += 1
            for issue in payload.get("issue_tags", []):
                issue_counts[str(issue)] += 1
            if mode == "baseline" and item_id in truth and not payload.get("abstention"):
                answer_type = questions[item_id]["answer_type"]
                expected_set = ground_truth_answer_set(
                    truth[item_id]["representation"], answer_type
                )
                entries = payload.get("answer_entries")
                if not isinstance(entries, list):
                    _, entries = normalize_structured_answer(payload.get("answer_exact", ""))
                predicted = submitted_answer_set(entries, answer_type)
                base.update(answer_scores(predicted, expected_set, answer_type))
                base.update(
                    {"tier": questions[item_id]["tier"], "category": questions[item_id]["category"]}
                )
            if mode == "prospective":
                base["overall_label"] = payload.get("overall_label", "")
                prospective_ratings[item_id].append(payload.get("overall_label", ""))
                agreement_ratings[f"prospective:{item_id}"].append(payload.get("overall_label", ""))
            elif mode == "audit":
                agreement_ratings[f"audit:{item_id}"].append(payload.get("severity", ""))
            rows.append(base)
    submitted = [x for x in rows if x["submitted"]]
    baseline = [x for x in submitted if x["mode"] == "baseline"]
    pairwise = None
    if len(export_paths) == 2:
        mappings = []
        for path in export_paths:
            _, anns, _ = load_export(path)
            mappings.append(
                {
                    f"{x['mode']}:{x['item_id']}": (
                        x["payload"].get("overall_label", "")
                        if x["mode"] == "prospective"
                        else x["payload"].get("severity", "")
                    )
                    for x in anns
                    if x["mode"] in {"prospective", "audit"}
                }
            )
        common = sorted(set(mappings[0]) & set(mappings[1]))
        pairwise = cohen_kappa([mappings[0][x] for x in common], [mappings[1][x] for x in common])
    raw_agreement = None
    if agreement_ratings:
        comparable = [values for values in agreement_ratings.values() if len(values) >= 2]
        if comparable:
            raw_agreement = sum(len(set(values)) == 1 for values in comparable) / len(comparable)
    reliability = administrator_reliability(rows, unblinding_paths or [])
    summary = {
        "exports": len(export_paths),
        "completion_by_mode": dict(Counter(x["mode"] for x in submitted)),
        "completion_by_tier_category": dict(
            Counter(f"T{x.get('tier')}:{x.get('category')}" for x in baseline)
        ),
        "baseline_macro": {
            metric: mean([x[metric] for x in baseline if metric in x])
            for metric in ("exact_match", "precision", "recall", "f1")
        },
        "timing_totals": {
            "active_seconds": sum(float(x.get("active", 0)) for x in submitted),
            "wall_seconds": sum(float(x.get("wall", 0)) for x in submitted),
            "self_reported_offline_minutes": sum(
                float(x.get("offline_minutes") or 0) for x in submitted
            ),
        },
        "mean_confidence": mean(
            [float(x["confidence"]) for x in submitted if x.get("confidence") not in {None, ""}]
        ),
        "abstentions": sum(bool(x["abstention"]) for x in submitted),
        "time_limit_outcomes": sum(x["abstention"] == "time_limit" for x in submitted),
        "tool_use": dict(tool_counts),
        "audit_issues": dict(issue_counts),
        "prospective_fractions": fractions(prospective_ratings),
        "agreement": {
            "raw_proportion": raw_agreement,
            "cohen_kappa_two_annotators": pairwise,
            "krippendorff_alpha_nominal": krippendorff_alpha_nominal(agreement_ratings),
        },
        "missing_data_rule": "Only first submitted, non-abstained baseline answers enter accuracy; post-reveal revisions do not replace them, and uncertain prospective labels remain a separate category.",
        "disagreements": {
            item: values
            for item, values in agreement_ratings.items()
            if len(set(x for x in values if x)) > 1
        },
        "administrator_control_and_duplicate_checks": reliability,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(output / "item_metrics.csv", rows)
    write_csv(
        output / "disagreements.csv",
        [
            {"item_id": key, "labels": "|".join(values)}
            for key, values in summary["disagreements"].items()
        ],
    )
    return summary


def administrator_reliability(
    rows: list[dict[str, Any]], unblinding_paths: list[Path]
) -> dict[str, Any] | None:
    if not unblinding_paths:
        return None
    hidden: dict[str, dict[str, Any]] = {}
    for path in unblinding_paths:
        hidden.update(json.loads(path.read_text())["map"])
    controls: list[bool] = []
    duplicate_labels: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["mode"] != "prospective" or not row["submitted"]:
            continue
        secret = hidden.get(row["item_id"], {}).get("hidden_fields", {})
        label = str(row.get("overall_label", ""))
        control_type = secret.get("control_type")
        if control_type == "positive" and label:
            controls.append(label == "plausible_alternative")
        elif control_type == "negative" and label:
            controls.append(label == "implausible")
        duplicate_group = secret.get("duplicate_group")
        if duplicate_group and label:
            duplicate_labels[str(duplicate_group)].append(label)
    usable_duplicates = [values for values in duplicate_labels.values() if len(values) >= 2]
    return {
        "control_count": len(controls),
        "control_accuracy": sum(controls) / len(controls) if controls else None,
        "duplicate_group_count": len(usable_duplicates),
        "duplicate_raw_agreement": (
            sum(len(set(values)) == 1 for values in usable_duplicates) / len(usable_duplicates)
            if usable_duplicates
            else None
        ),
    }


def fractions(ratings: dict[str, list[str]]) -> dict[str, float]:
    values = [x for labels in ratings.values() for x in labels if x]
    counts = Counter(values)
    return {key: count / len(values) for key, count in counts.items()} if values else {}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
