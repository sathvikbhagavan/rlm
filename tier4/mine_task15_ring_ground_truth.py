"""Mine full-dataset 3-reaction ring-construction chains for tier4 task15."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rlm.codeact_helpers import load_lines
from task15_ring_chain_graph import (
    PATH_LENGTH,
    build_molecule_graphs,
    context_filters_from_records,
    parse_records_from_lines,
    shortest_ring_construction_path,
)
from task15_ring_chain_ground_truth import DEFAULT_RING_QUERIES, question_key

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
TIER4_DIR = Path(__file__).resolve().parent
CHAINS_JSON_PATH = TIER4_DIR / "task15_ring_hardcoded_chains.json"
GROUND_TRUTH_PY_PATH = TIER4_DIR / "task15_ring_chain_ground_truth.py"


def mine_full_dataset() -> dict[str, dict[str, object]]:
    lines = load_lines(DATASET_PATH)
    records = parse_records_from_lines(lines)
    filters = context_filters_from_records(records)
    _forward, reverse_graph, annotations, _ = build_molecule_graphs(records, filters)

    print(
        f"Loaded {len(records)} reactions; "
        f"molecule_freq_cap={filters.molecule_freq_cap} "
        f"frequent_molecules={len(filters.frequent_molecules)} "
        f"annotated_nodes={len(annotations)}"
    )

    mined: dict[str, dict[str, object]] = {}
    for ring_system in DEFAULT_RING_QUERIES:
        gt = shortest_ring_construction_path(
            reverse_graph=reverse_graph,
            annotations=annotations,
            ring_system=ring_system,
            min_path_reactions=PATH_LENGTH,
            max_path_reactions=PATH_LENGTH,
        )
        if gt is None:
            raise ValueError(f"No full-dataset {PATH_LENGTH}-reaction chain for {ring_system}")

        chains = [list(rxns) for rxns in gt.accepted_reaction_indices]
        support = sorted({idx for chain in chains for idx in chain})
        key = question_key(ring_system)
        mined[key] = {
            "ring_system": ring_system,
            "path_length": PATH_LENGTH,
            "chain_count": len(chains),
            "support_index_count": len(support),
            "example_chain": list(gt.reaction_indices),
            "chains": chains,
        }
        print(
            f"{key}: chains={len(chains)} "
            f"path_length={PATH_LENGTH} "
            f"support={len(support)} "
            f"example={list(gt.reaction_indices)}"
        )
    return mined


def write_chains_json(mined: dict[str, dict[str, object]]) -> None:
    with CHAINS_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(mined, handle, separators=(",", ":"))


def update_ground_truth_module(mined: dict[str, dict[str, object]]) -> None:
    count_lines = [f'    {key!r}: {payload["chain_count"]},' for key, payload in mined.items()]
    example_lines = [
        f'    {key!r}: {tuple(payload["example_chain"])},'
        for key, payload in mined.items()
    ]

    text = GROUND_TRUTH_PY_PATH.read_text(encoding="utf-8")

    count_pattern = r"HARDCODED_GT_CHAIN_COUNTS: dict\[str, int\] = \{[^}]*\}"
    count_replacement = (
        "HARDCODED_GT_CHAIN_COUNTS: dict[str, int] = {\n"
        + "\n".join(count_lines)
        + "\n}"
    )
    text, count = re.subn(count_pattern, count_replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Failed to update HARDCODED_GT_CHAIN_COUNTS")

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
