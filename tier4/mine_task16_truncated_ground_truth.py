"""Mine full-dataset length-5 synthesis chains for tier4 task16."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rlm.codeact_helpers import load_lines
from task16_truncated_synthesis_graph import (
    FULL_CHAIN_LENGTH,
    PREFIX_LENGTH,
    mine_full_chains_for_target,
    prefixes_from_full_chains,
    terminal_indices_from_full_chains,
)
from task16_truncated_synthesis_ground_truth import DEFAULT_TARGET_QUESTIONS, question_key

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
TIER4_DIR = Path(__file__).resolve().parent
CHAINS_JSON_PATH = TIER4_DIR / "task16_truncated_hardcoded_chains.json"
GROUND_TRUTH_PY_PATH = TIER4_DIR / "task16_truncated_synthesis_ground_truth.py"


def mine_full_dataset() -> dict[str, dict[str, object]]:
    lines = load_lines(DATASET_PATH)
    print(f"Loaded {len(lines)} reactions")

    mined: dict[str, dict[str, object]] = {}
    for spec in DEFAULT_TARGET_QUESTIONS:
        full_chains, terminal_pool = mine_full_chains_for_target(
            lines,
            spec.target_smiles,
            chain_length=FULL_CHAIN_LENGTH,
        )
        if not full_chains:
            raise ValueError(
                f"No full-dataset {FULL_CHAIN_LENGTH}-reaction chains for {spec.question_id}"
            )

        full_tuple = tuple(full_chains)
        prefixes = prefixes_from_full_chains(full_tuple)
        terminals = terminal_indices_from_full_chains(full_tuple, prefixes)
        context_excluded = terminal_indices_from_full_chains(full_tuple, prefixes)
        support = sorted({idx for chain in full_tuple for idx in chain})
        prefix_support = sorted({idx for prefix in prefixes for idx in prefix})
        key = question_key(spec.question_id)
        example_full = full_tuple[0]
        example_prefix = example_full[:PREFIX_LENGTH]

        mined[key] = {
            "question_id": spec.question_id,
            "target_smiles": spec.target_smiles,
            "label": spec.label,
            "full_chain_length": FULL_CHAIN_LENGTH,
            "prefix_length": PREFIX_LENGTH,
            "full_chain_count": len(full_tuple),
            "prefix_count": len(prefixes),
            "terminal_index_count": len(terminals),
            "context_excluded_terminal_count": len(context_excluded),
            "terminal_pool_count": len(terminal_pool),
            "support_index_count": len(support),
            "prefix_support_index_count": len(prefix_support),
            "example_full_chain": list(example_full),
            "example_prefix": list(example_prefix),
            "terminal_indices": sorted(terminals),
            "context_excluded_terminals": sorted(context_excluded),
            "full_chains": [list(chain) for chain in full_tuple],
            "prefix_chains": [list(prefix) for prefix in prefixes],
        }
        print(
            f"{key}: full_chains={len(full_tuple)} prefixes={len(prefixes)} "
            f"terminals={len(terminals)} example_prefix={list(example_prefix)}"
        )
    return mined


def write_chains_json(mined: dict[str, dict[str, object]]) -> None:
    with CHAINS_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(mined, handle, separators=(",", ":"))


def update_ground_truth_module(mined: dict[str, dict[str, object]]) -> None:
    prefix_count_lines = [
        f'    {key!r}: {payload["prefix_count"]},' for key, payload in mined.items()
    ]
    full_count_lines = [
        f'    {key!r}: {payload["full_chain_count"]},' for key, payload in mined.items()
    ]
    example_lines = [
        f'    {key!r}: {tuple(payload["example_prefix"])},'
        for key, payload in mined.items()
    ]

    text = GROUND_TRUTH_PY_PATH.read_text(encoding="utf-8")

    prefix_pattern = r"HARDCODED_GT_PREFIX_COUNTS: dict\[str, int\] = \{[^}]*\}"
    prefix_replacement = (
        "HARDCODED_GT_PREFIX_COUNTS: dict[str, int] = {\n"
        + "\n".join(prefix_count_lines)
        + "\n}"
    )
    text, count = re.subn(prefix_pattern, prefix_replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Failed to update HARDCODED_GT_PREFIX_COUNTS")

    full_pattern = r"HARDCODED_GT_FULL_CHAIN_COUNTS: dict\[str, int\] = \{[^}]*\}"
    full_replacement = (
        "HARDCODED_GT_FULL_CHAIN_COUNTS: dict[str, int] = {\n"
        + "\n".join(full_count_lines)
        + "\n}"
    )
    text, count = re.subn(full_pattern, full_replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Failed to update HARDCODED_GT_FULL_CHAIN_COUNTS")

    example_pattern = r"HARDCODED_GT_EXAMPLE: dict\[str, tuple\[int, \.\.\.\]\] = \{[^}]*\}"
    example_replacement = (
        "HARDCODED_GT_EXAMPLE: dict[str, tuple[int, ...]] = {\n"
        + "\n".join(example_lines)
        + "\n}"
    )
    text, count = re.subn(example_pattern, example_replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Failed to update HARDCODED_GT_EXAMPLE")

    GROUND_TRUTH_PY_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    mined = mine_full_dataset()
    write_chains_json(mined)
    update_ground_truth_module(mined)
    print(f"Wrote {CHAINS_JSON_PATH}")
    print(f"Updated {GROUND_TRUTH_PY_PATH}")


if __name__ == "__main__":
    main()
