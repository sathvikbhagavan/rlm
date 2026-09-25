#!/usr/bin/env python3
"""Inventory failure traces and prepare stratified packets for human review.

The emitted heuristic signals are retrieval aids, not scientific labels. A
chemistry-specific cause enters the paper only after a reviewer records direct
support from the trace in ``reviewed_trace_causes.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PAPER_METHODS = ("llm", "codeact", "rlm")
PAPER_MODELS = (
    "Claude Haiku 4.5",
    "DeepSeek V4 Flash",
    "Gemini 3.7 Flash",
    "GPT-5 mini",
    "Qwen 3.5",
)

TASK_GROUPS = {
    "tier3/task6": "functional groups",
    "tier3/task7": "functional groups",
    "tier3/task8": "functional groups",
    "tier3/task9": "named reactions",
    "tier3/task10": "mechanisms",
    "tier3/task10b": "mechanisms",
    "tier3/task13": "bond changes",
    "tier3/task14": "bond changes",
    "tier3/task15": "bond changes",
    "tier3/task17": "scaffolds",
    "tier3/task18": "scaffolds",
    "tier3/task20": "scaffolds",
    "tier3/task21": "reagents",
    "tier3/task22": "reagents",
    "tier3/task23": "stereochemistry",
    "tier3/task24": "stereochemistry",
    "tier4/task11": "mechanical graph",
    "tier4/task12": "mechanical graph",
    "tier4/task12b": "mechanical graph",
    "tier4/task13": "chemically constrained graph",
    "tier4/task14": "chemically constrained graph",
    "tier4/task15": "chemically constrained graph",
    "tier4/task16": "route construction",
    "tier4/task17": "multi-constraint chains",
    "tier4/task17b": "multi-constraint chains",
}

SIGNALS = {
    "unsupported_action_wrapper": re.compile(
        r"<invoke\s+name=[\"'](?:python|bash|shell)[\"']",
        re.IGNORECASE,
    ),
    "rdkit_runtime_error": re.compile(
        r"Boost\.Python\.ArgumentError|Pre-condition Violation|Range Error|"
        r"Python argument types in|RDKit ERROR|Invariant Violation",
        re.IGNORECASE,
    ),
    "python_exception": re.compile(
        r"Traceback \(most recent call last\)|(?:Type|Value|Attribute|Index|Key)Error:",
        re.IGNORECASE,
    ),
    "smiles_parse_failure": re.compile(
        r"SMILES Parse Error|MolFromSmiles\([^\n]*\)\s*(?:is|==)\s*None|"
        r"failed to parse.*SMILES",
        re.IGNORECASE,
    ),
    "literal_string_proxy": re.compile(
        r"substring|string check|\.count\(['\"](?:N|O|Br|Cl|F)|"
        r"['\"](?:Br|Cl|Boc|Pd)['\"]\s+in\s+",
        re.IGNORECASE,
    ),
    "reagent_name_proxy": re.compile(
        r"reagent(?:s| field)?[^\n]{0,80}(?:contain|presence|name|string)|"
        r"(?:Pd|phosphine|azide|Boc)[^\n]{0,50}(?:reagent|presence)",
        re.IGNORECASE,
    ),
    "generated_smarts": re.compile(r"MolFromSmarts|ReactionFromSmarts|RunReactants"),
    "explicit_truncation": re.compile(
        r"\[:\s*(?:10|20|50|100|200|500)\s*\]|\b(?:top|first|sample)\s+"
        r"(?:10|20|50|100|200|500)\b|max_(?:results|chains|candidates)",
        re.IGNORECASE,
    ),
    "iteration_or_time_limit": re.compile(
        r"max(?:imum)? iterations|iteration limit|workflow timeout|timed out|"
        r"time limit|execution limit",
        re.IGNORECASE,
    ),
    "memory_failure": re.compile(
        r"memory limit|out of memory|cannot allocate memory|std::bad_alloc|Killed",
        re.IGNORECASE,
    ),
    "provider_or_transport_failure": re.compile(
        r"HTTP (?:4\d\d|5\d\d)|rate limit|ConnectionError|ReadTimeout|"
        r"connection refused|policy violation|empty choices",
        re.IGNORECASE,
    ),
    "zero_record_parse": re.compile(
        r"(?:parsed|loaded|found|read)\s+(?:a total of\s+)?0\s+(?:reaction|record|row)",
        re.IGNORECASE,
    ),
    "placeholder_or_empty_answer": re.compile(
        r"(?:final_answer_result|ANSWER|FINAL ANSWER)\s*[:=]\s*(?:\{\}|\[\]|None|null)?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_local_logs(search_root: Path) -> list[Path]:
    patterns = (
        "rlm*/artifacts/*/runs/*/attempt-*/stdout.log",
        "rlm*/**/artifacts/*/runs/*/attempt-*/stdout.log",
        ".cache/rxnhaystack-failure-traces/runs/*/attempt-*/stdout.log",
    )
    return sorted({path for pattern in patterns for path in search_root.glob(pattern)})


def canonical_run_id(local_run_id: str, known_run_ids: tuple[str, ...]) -> str | None:
    if local_run_id in known_run_ids:
        return local_run_id
    matches = [run_id for run_id in known_run_ids if local_run_id.endswith(run_id)]
    return max(matches, key=len) if matches else None


def locate_traces(records: list[dict[str, str]], search_root: Path) -> dict[str, list[Path]]:
    known = tuple(row["run_id"] for row in records)
    found: dict[str, list[Path]] = defaultdict(list)
    for path in find_local_logs(search_root):
        matched = canonical_run_id(path.parents[1].name, known)
        if matched is not None:
            found[matched].append(path)
    for paths in found.values():
        paths.sort(key=lambda path: (path.stat().st_size, str(path)), reverse=True)
    return found


def wandb_urls(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    return {
        row["run_id"]: row.get("wandb_url", "") for row in read_csv(path) if row.get("wandb_url")
    }


def signal_counts(text: str) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in SIGNALS.items()}


def evidence_excerpt(text: str, limit: int = 8) -> str:
    evidence: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern.search(stripped) for pattern in SIGNALS.values()):
            evidence.append(stripped[:300])
        if len(evidence) >= limit:
            break
    return " || ".join(evidence)


def eligible_wrong_answer(row: dict[str, str], pending: set[str]) -> bool:
    if row["run_id"] in pending or row["status"] != "succeeded":
        return False
    if row["method"] not in PAPER_METHODS or int(row["tier"]) < 3:
        return False
    return bool(row["f1"]) and float(row["f1"]) < 1.0 - 1e-12


def select_review_rows(
    rows: list[dict[str, Any]], *, per_model_method: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["local_trace"] or row["wandb_url"]:
            groups[(str(row["model_label"]), str(row["method"]))].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        candidates = groups[key]
        candidates.sort(
            key=lambda row: (
                not bool(row["local_trace"]),
                float(row["f1"]),
                -int(row["context_numeric"]),
                str(row["task"]),
                str(row["run_id"]),
            )
        )
        used_tasks: set[str] = set()
        diverse: list[dict[str, Any]] = []
        for row in candidates:
            if row["task"] not in used_tasks:
                diverse.append(row)
                used_tasks.add(str(row["task"]))
            if len(diverse) == per_model_method:
                break
        if len(diverse) < per_model_method:
            for row in candidates:
                if row not in diverse:
                    diverse.append(row)
                if len(diverse) == per_model_method:
                    break
        selected.extend(diverse)
    return selected


def build(args: argparse.Namespace) -> None:
    root = args.repo_root.resolve()
    gold = root / "paper_plots/gold/iclr2027"
    records = read_csv(gold / "final_arm_records.csv")
    pending = {
        row["run_id"] for row in read_csv(gold / "post_submission/pending_corrected_rescores.csv")
    }
    traces = locate_traces(records, args.search_root.resolve())
    urls = wandb_urls(args.wandb_records)

    inventory: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for row in records:
        paths = traces.get(row["run_id"], [])
        path = paths[0] if paths else None
        context_numeric = 122456 if row["context"] == "full" else int(row["context"])
        item: dict[str, Any] = {
            "run_id": row["run_id"],
            "model_label": row["model_label"],
            "method": row["method"],
            "tier": row["tier"],
            "task": row["task"],
            "task_group": TASK_GROUPS.get(row["task"], "other"),
            "context": row["context"],
            "context_numeric": context_numeric,
            "repetition": row["repetition"],
            "status": row["status"],
            "f1": row["f1"],
            "correction_status": row["score_correction_status"],
            "pending_corrected_rescore": row["run_id"] in pending,
            "local_trace": str(path) if path else "",
            "local_trace_sha256": sha256(path) if path else "",
            "local_trace_bytes": path.stat().st_size if path else "",
            "wandb_url": urls.get(row["run_id"], ""),
        }
        inventory.append(item)
        if eligible_wrong_answer(row, pending):
            candidate = dict(item)
            if path:
                text = path.read_text(errors="replace")
                candidate.update(signal_counts(text))
                candidate["automated_evidence_excerpt"] = evidence_excerpt(text)
            else:
                candidate.update({name: 0 for name in SIGNALS})
                candidate["automated_evidence_excerpt"] = ""
            candidates.append(candidate)

    output = args.output_dir.resolve()
    base_fields = tuple(inventory[0])
    candidate_fields = tuple(candidates[0])
    write_csv(output / "trace_inventory.csv", inventory, base_fields)
    write_csv(output / "wrong_answer_candidates.csv", candidates, candidate_fields)
    selected = select_review_rows(candidates, per_model_method=args.per_model_method)
    write_csv(output / "review_sample.csv", selected, candidate_fields)

    lines = [
        "# Failure-trace review packets",
        "",
        "Automated signals below retrieve evidence for review; they are not scientific labels.",
        "Runs awaiting corrected rescoring are excluded.",
        "",
    ]
    for index, row in enumerate(selected, start=1):
        signals = ", ".join(name for name in SIGNALS if int(row[name]) > 0) or "none"
        lines.extend(
            (
                f"## {index}. `{row['run_id']}`",
                "",
                f"- Model/interface: {row['model_label']} / {row['method']}",
                f"- Task/context/F1: {row['task']} / {row['context']} / {row['f1']}",
                f"- Task group: {row['task_group']}",
                f"- Trace: `{row['local_trace'] or row['wandb_url'] or 'unavailable'}`",
                f"- Candidate signals: {signals}",
                f"- Extract: {row['automated_evidence_excerpt'] or 'none'}",
                "- Reviewer cause:",
                "- Direct evidence:",
                "- Confidence:",
                "",
            )
        )
    (output / "review_packets.md").write_text("\n".join(lines))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--search-root", type=Path, default=Path("/home/amin"))
    result.add_argument("--wandb-records", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--per-model-method", type=int, default=3)
    return result


if __name__ == "__main__":
    build(parser().parse_args())
