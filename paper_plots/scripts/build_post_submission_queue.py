#!/usr/bin/env python3
"""Freeze the exact corrected-rescore queue for post-submission completion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rxnhaystack.score_recovery import SUBMISSION_SCORE_FREEZE_ID

MODEL_ALIASES = {
    "RCP-AIaaS/Qwen/Qwen3.5-397B-A17B": ("qwen3.5", "Qwen 3.5"),
    "qwen3.5-397b": ("qwen3.5", "Qwen 3.5"),
    "qwen3.5": ("qwen3.5", "Qwen 3.5"),
    "deepseek-v4-flash": ("deepseek-v4-flash", "DeepSeek V4 Flash"),
    "gemini-3.7-flash": ("gemini-3.7-flash", "Gemini 3.7 Flash"),
    "glm-5.2": ("glm-5.2", "GLM 5.2"),
    "gpt-5-mini": ("gpt-5-mini", "GPT-5 mini"),
    "openai/gpt-5-mini": ("gpt-5-mini", "GPT-5 mini"),
    "claude-haiku-4.5": ("claude-haiku-4.5", "Claude Haiku 4.5"),
    "anthropic/claude-haiku-4.5": ("claude-haiku-4.5", "Claude Haiku 4.5"),
    "none": ("deterministic-executor", "Deterministic executor"),
}

TABLES = (
    ("full_benchmark_records.csv", "main_benchmark"),
    ("codeact_x1000_records.csv", "x1000_extension"),
    ("rlm_x1000_records.csv", "x1000_extension"),
    ("causal_controls/records.csv", "causal_control"),
    ("causal_controls/matched_qwen_terminal.csv", "matched_control"),
)

FIELDNAMES = (
    "schema_version",
    "recovery_id",
    "submission_score_freeze_id",
    "queue_status",
    "run_id",
    "experiments",
    "arms",
    "model",
    "model_label",
    "method",
    "tier",
    "task",
    "context",
    "configured_seed",
    "repetition",
    "question_count",
    "historical_score_excluded_from_submission",
    "score_name",
    "source_entity",
    "wandb_url",
    "reason",
    "source_sha256",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def normalized_model(row: dict[str, str]) -> tuple[str, str]:
    value = row.get("model", "")
    if value in MODEL_ALIASES:
        return MODEL_ALIASES[value]
    label = row.get("model_label", "")
    for slug, known_label in MODEL_ALIASES.values():
        if label == known_label:
            return slug, known_label
    raise ValueError(f"Unrecognized queued model: {value!r} / {label!r}")


def occurrence_arm(row: dict[str, str], experiment: str) -> str:
    if experiment in {"main_benchmark", "x1000_extension"}:
        return f"{row['scope']}/{row['method']}/x{row['context']}"
    return "/".join(
        (
            row["study"],
            row["arm"],
            row.get("condition", ""),
            f"x{row['context']}",
        )
    )


def source_entity(url: str) -> str:
    parts = url.split("/")
    return parts[3] if len(parts) > 3 else ""


def build_queue(
    *, recovery: dict[str, Any], gold_dir: Path, configured_seed: int
) -> list[dict[str, Any]]:
    audit_rows = [*recovery["recoveries"], *recovery["unresolved"]]
    unresolved = {str(row["run_id"]): row for row in audit_rows}
    if len(unresolved) != len(audit_rows):
        raise ValueError("Recovery manifest contains duplicate affected run IDs")

    occurrences: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for relative_path, experiment in TABLES:
        path = gold_dir / relative_path
        for row in read_csv(path):
            run_id = row["run_id"]
            if run_id in unresolved:
                occurrences[run_id].append((experiment, row))

    missing = sorted(unresolved.keys() - occurrences.keys())
    if missing:
        raise ValueError(f"Queued runs absent from frozen experiment tables: {missing[:3]!r}")

    output: list[dict[str, Any]] = []
    for run_id, pending in sorted(unresolved.items()):
        run_occurrences = occurrences[run_id]
        models = {normalized_model(row) for _experiment, row in run_occurrences}
        tasks = {row["task"] for _experiment, row in run_occurrences}
        contexts = {row["context"] for _experiment, row in run_occurrences}
        repetitions = {int(row["repetition"]) for _experiment, row in run_occurrences}
        scores = {
            float(row["original_f1"])
            for _experiment, row in run_occurrences
            if row.get("original_f1", "") != ""
        }
        question_counts = {
            int(row["question_count"])
            for _experiment, row in run_occurrences
            if row.get("question_count", "") != ""
        }
        statuses = {row["score_correction_status"] for _experiment, row in run_occurrences}
        if (
            len(models) != 1
            or len(tasks) != 1
            or len(contexts) != 1
            or len(repetitions) != 1
            or len(scores) != 1
            or len(question_counts) != 1
            or statuses != {"historical_score_invalidated"}
        ):
            raise ValueError(f"Inconsistent frozen metadata for queued run {run_id}")
        model, model_label = next(iter(models))
        task = next(iter(tasks))
        score_names = {
            row.get("score_name", "macro_f1") or "macro_f1" for _experiment, row in run_occurrences
        }
        if len(score_names) != 1:
            raise ValueError(f"Inconsistent score names for queued run {run_id}")
        methods = {row.get("method", "rlm") or "rlm" for _experiment, row in run_occurrences}
        if len(methods) != 1:
            raise ValueError(f"Inconsistent methods for queued run {run_id}")
        output.append(
            {
                "schema_version": 1,
                "recovery_id": recovery["recovery_id"],
                "submission_score_freeze_id": SUBMISSION_SCORE_FREEZE_ID,
                "queue_status": "pending_exact_corrected_rescore",
                "run_id": run_id,
                "experiments": ";".join(
                    sorted({experiment for experiment, _row in run_occurrences})
                ),
                "arms": ";".join(
                    sorted({occurrence_arm(row, experiment) for experiment, row in run_occurrences})
                ),
                "model": model,
                "model_label": model_label,
                "method": next(iter(methods)),
                "tier": int(task.removeprefix("tier").split("/", 1)[0]),
                "task": task,
                "context": next(iter(contexts)),
                "configured_seed": configured_seed,
                "repetition": next(iter(repetitions)),
                "question_count": next(iter(question_counts)),
                "historical_score_excluded_from_submission": next(iter(scores)),
                "score_name": next(iter(score_names)),
                "source_entity": source_entity(str(pending.get("wandb_url", ""))),
                "wandb_url": pending.get("wandb_url", ""),
                "reason": pending.get("reason", "complete-post-submission-corrected-score-audit"),
                "source_sha256": pending["source_sha256"],
            }
        )
    return output


def counter_table(title: str, counter: Counter[str]) -> list[str]:
    lines = [f"## {title}", "", "| Value | Runs |", "| --- | ---: |"]
    lines.extend(f"| `{value}` | {count} |" for value, count in sorted(counter.items()))
    lines.append("")
    return lines


def write_outputs(
    *, rows: list[dict[str, Any]], recovery: dict[str, Any], output_dir: Path
) -> None:
    configured_seeds = {int(row["configured_seed"]) for row in rows}
    if len(configured_seeds) != 1:
        raise ValueError("Post-submission queue must use one configured sampling seed")
    configured_seed = next(iter(configured_seeds))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pending_corrected_rescores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    jsonl_path = output_dir / "pending_corrected_rescores.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    lines = [
        "# Post-submission corrected-rescore queue",
        "",
        "This is internal audit material, not manuscript prose. It freezes every run whose "
        "score is excluded from submission-time aggregates. Historical values are retained "
        "only for reproducibility; every listed run requires a post-submission audit against "
        "the corrected human bundle.",
        "",
        f"- Unique queued runs: **{len(rows)}**",
        f"- Recovery ledger: `{recovery['recovery_id']}`",
        f"- Submission score freeze: `{SUBMISSION_SCORE_FREEZE_ID}`",
        f"- Configured sampling seed: **{configured_seed}** for every queued run; `repetition` records the "
        "independent model repetition (`r01` through `r05`).",
        "",
        "After submission: recover or rerun exactly these run IDs against the corrected bundle, "
        "replace the carried score with an exact rescore, regenerate every gold table and figure, "
        "and compare the resulting manuscript claims against the frozen submission manifest.",
        "",
    ]
    lines.extend(counter_table("By experiment", Counter(row["experiments"] for row in rows)))
    lines.extend(counter_table("By model", Counter(row["model"] for row in rows)))
    lines.extend(counter_table("By task", Counter(row["task"] for row in rows)))
    lines.extend(counter_table("By method", Counter(row["method"] for row in rows)))
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "recovery_id": recovery["recovery_id"],
        "submission_score_freeze_id": SUBMISSION_SCORE_FREEZE_ID,
        "queued_run_count": len(rows),
        "configured_seed": configured_seed,
        "counts": {
            "by_experiment": dict(Counter(row["experiments"] for row in rows)),
            "by_model": dict(Counter(row["model"] for row in rows)),
            "by_task": dict(Counter(row["task"] for row in rows)),
            "by_method": dict(Counter(row["method"] for row in rows)),
            "by_context": dict(Counter(str(row["context"]) for row in rows)),
            "by_repetition": dict(Counter(str(row["repetition"]) for row in rows)),
        },
        "files": {
            csv_path.name: sha256_file(csv_path),
            jsonl_path.name: sha256_file(jsonl_path),
            readme_path.name: sha256_file(readme_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recovery",
        type=Path,
        default=Path("paper_plots/gold/corrected_score_recoveries.json"),
    )
    parser.add_argument("--gold-dir", type=Path, default=Path("paper_plots/gold/iclr2027"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_plots/gold/iclr2027/post_submission"),
    )
    parser.add_argument("--configured-seed", type=int, default=42)
    args = parser.parse_args()
    recovery = json.loads(args.recovery.read_text(encoding="utf-8"))
    rows = build_queue(
        recovery=recovery,
        gold_dir=args.gold_dir,
        configured_seed=args.configured_seed,
    )
    if len(rows) != int(recovery["counts"]["unique_affected_runs"]):
        raise ValueError(
            "Queue cardinality mismatch: "
            f"{len(rows)} != {recovery['counts']['unique_affected_runs']}"
        )
    write_outputs(rows=rows, recovery=recovery, output_dir=args.output_dir)
    print(f"Wrote {len(rows)} post-submission corrected-rescore rows")


if __name__ == "__main__":
    main()
