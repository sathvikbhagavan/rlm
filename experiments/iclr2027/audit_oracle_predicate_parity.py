"""Audit answer-free oracle helpers against every row of the pinned dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rlm.codeact_helpers import load_lines

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def add_path(tier: str) -> None:
    sys.modules.pop("oracle_predicates", None)
    sys.path.insert(0, str(PROJECT_ROOT / tier))


def digest(indices: list[int]) -> str:
    payload = ",".join(map(str, indices)).encode()
    return hashlib.sha256(payload).hexdigest()


def audit_task6(lines: list[str]) -> dict[str, object]:
    add_path("tier3")
    from oracle_predicates import task6_reaction_matches
    from rlm_task6 import AMIDE_COUPLING_LABELS
    from task6_hardcoded_ground_truth import TASK6_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION

    results = {}
    for key in AMIDE_COUPLING_LABELS:
        actual = [int(line.split(" ", 1)[0]) for line in lines if task6_reaction_matches(line, key)]
        expected = TASK6_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION[key]
        results[key] = {
            "match": actual == expected,
            "count": len(actual),
            "sha256": digest(actual),
        }
    return results


def audit_task10(lines: list[str]) -> dict[str, object]:
    add_path("tier3")
    from oracle_predicates import task10_reaction_matches
    from task10_hardcoded_ground_truth import TASK10_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION
    from task10_prompt_config import REACTION_KEYS

    results = {}
    for key in REACTION_KEYS:
        actual = [
            int(line.split(" ", 1)[0]) for line in lines if task10_reaction_matches(line, key)
        ]
        expected = TASK10_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION[key]
        results[key] = {
            "match": actual == expected,
            "count": len(actual),
            "sha256": digest(actual),
        }
    return results


def audit_task23(lines: list[str]) -> dict[str, object]:
    add_path("tier3")
    from oracle_predicates import task23_reaction_matches
    from task23_hardcoded_ground_truth import TASK23_HARDCODED_GROUND_TRUTH_INDICES

    actual = [int(line.split(" ", 1)[0]) for line in lines if task23_reaction_matches(line)]
    return {
        "match": actual == TASK23_HARDCODED_GROUND_TRUTH_INDICES,
        "count": len(actual),
        "sha256": digest(actual),
    }


def audit_task13(lines: list[str]) -> dict[str, object]:
    add_path("tier4")
    from oracle_predicates import task13_detect_functional_groups
    from task13_fg_chain_graph import canonicalize_components, detect_functional_groups

    checked = 0
    mismatches = 0
    for line in lines:
        reaction = line.split(" ", 1)[1]
        for side in (reaction.split(">", 2)[0], reaction.rsplit(">", 1)[-1]):
            for smiles in canonicalize_components(side):
                checked += 1
                mismatches += task13_detect_functional_groups(smiles) != detect_functional_groups(
                    smiles
                )
    return {"match": mismatches == 0, "molecules_checked": checked, "mismatches": mismatches}


def audit_task14(lines: list[str]) -> dict[str, object]:
    add_path("tier4")
    from oracle_predicates import task14_protection_events
    from task14_protecting_group_graph import (
        PROTECTING_GROUPS,
        mine_protection_events,
        parse_records_from_lines,
    )

    records = parse_records_from_lines(lines)
    expected = {
        (event.reaction_index, event.pg_label, event.direction, event.scaffold_key)
        for event in mine_protection_events(records)
    }
    actual = {
        (int(line.split(" ", 1)[0]), spec.label, event.direction, event.scaffold_key)
        for line in lines
        for spec in PROTECTING_GROUPS
        for event in task14_protection_events(line, spec.label)
    }
    return {
        "match": actual == expected,
        "events_checked": len(expected),
        "missing": len(expected - actual),
        "extra": len(actual - expected),
    }


AUDITS = {
    "tier3-task6": audit_task6,
    "tier3-task10": audit_task10,
    "tier3-task23": audit_task23,
    "tier4-task13": audit_task13,
    "tier4-task14": audit_task14,
}


def all_match(value: object) -> bool:
    if isinstance(value, dict) and "match" in value:
        return bool(value["match"])
    if isinstance(value, dict):
        return all(all_match(item) for item in value.values())
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", action="append", choices=tuple(AUDITS), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lines = load_lines(str(args.dataset))
    report = {
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "row_count": len(lines),
        "audits": {name: AUDITS[name](lines) for name in args.task},
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if all_match(report["audits"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
