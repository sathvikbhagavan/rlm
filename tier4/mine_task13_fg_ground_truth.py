"""Mine full-dataset 7-reaction FG chains for tier4 task13.

Uses full-dataset graph filters (molecule_freq_cap=200, heavy atoms 3–90).
Writes task13_fg_hardcoded_chains.json and refreshes summary constants in
task13_fg_chain_ground_truth.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from rlm.codeact_helpers import load_lines
from task13_fg_chain_ground_truth import FIXED_FG_QUERIES
from task13_fg_chain_graph import (
    DATASET_TOTAL_REACTIONS,
    MAX_HEAVY_ATOMS,
    MAX_MOLECULE_FREQ_REFERENCE,
    MIN_HEAVY_ATOMS,
    MIN_LOCAL_MOLECULE_FREQ,
    PATH_LENGTH,
    build_molecule_graph,
    context_filters_from_records,
    longest_fg_path,
    parse_records_from_lines,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
TIER4_DIR = Path(__file__).resolve().parent
CHAINS_JSON_PATH = TIER4_DIR / "task13_fg_hardcoded_chains.json"
GROUND_TRUTH_PY_PATH = TIER4_DIR / "task13_fg_chain_ground_truth.py"


def fg_pair_key(source_fg: str, target_fg: str) -> str:
    return f"{source_fg}->{target_fg}"


def mine_full_dataset() -> dict[str, dict[str, object]]:
    lines = load_lines(DATASET_PATH)
    records = parse_records_from_lines(lines)
    filters = context_filters_from_records(records)
    graph, group_cache, _ = build_molecule_graph(records, filters)

    print(
        f"Loaded {len(records)} reactions; "
        f"molecule_freq_cap={filters.molecule_freq_cap} "
        f"frequent_molecules={len(filters.frequent_molecules)} "
        f"graph_nodes={len(graph)}"
    )

    mined: dict[str, dict[str, object]] = {}
    for source_fg, target_fg in FIXED_FG_QUERIES:
        gt = longest_fg_path(
            graph=graph,
            group_cache=group_cache,
            source_fg=source_fg,
            target_fg=target_fg,
            path_length=PATH_LENGTH,
        )
        if gt is None:
            raise ValueError(f"No full-dataset chain for {source_fg}->{target_fg}")

        chains = [list(rxns) for rxns in gt.accepted_reaction_indices]
        support = sorted({idx for chain in chains for idx in chain})
        key = fg_pair_key(source_fg, target_fg)
        mined[key] = {
            "source_fg": source_fg,
            "target_fg": target_fg,
            "path_length": PATH_LENGTH,
            "chain_count": len(chains),
            "support_index_count": len(support),
            "example_chain": list(gt.reaction_indices),
            "chains": chains,
        }
        print(
            f"{key}: chains={len(chains)} "
            f"support={len(support)} example={list(gt.reaction_indices)}"
        )
    return mined


def write_chains_json(mined: dict[str, dict[str, object]]) -> None:
    with CHAINS_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(mined, handle, separators=(",", ":"))


def update_ground_truth_module(mined: dict[str, dict[str, object]]) -> None:
    example_lines = []
    count_lines = []
    for key, payload in mined.items():
        source_fg = payload["source_fg"]
        target_fg = payload["target_fg"]
        example = tuple(payload["example_chain"])
        example_lines.append(f'    ({source_fg!r}, {target_fg!r}): {example},')
        count_lines.append(f'    ({source_fg!r}, {target_fg!r}): {payload["chain_count"]},')

    module_text = f'''"""Hardcoded functional-group chain ground truth for tier4 task13.

Full-dataset 7-reaction chains were mined from reactionSmilesFigShareUSPTO2023_cleaned.txt
with molecule_freq_cap={MAX_MOLECULE_FREQ_REFERENCE} at {DATASET_TOTAL_REACTIONS} reactions,
heavy-atom window [{MIN_HEAVY_ATOMS}, {MAX_HEAVY_ATOMS}], and RDKit SMARTS FG detection.

All accepted chains live in task13_fg_hardcoded_chains.json. At evaluation time,
in-context GT is the subset of those chains present in the sampled context that also
pass context-local hub and heavy-atom filters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from task13_fg_chain_graph import (
    DATASET_TOTAL_REACTIONS,
    MAX_HEAVY_ATOMS,
    MAX_MOLECULE_FREQ_REFERENCE,
    MIN_HEAVY_ATOMS,
    MIN_LOCAL_MOLECULE_FREQ,
    PATH_LENGTH,
)

TASK13_TOTAL_REACTIONS = DATASET_TOTAL_REACTIONS
TASK13_MIN_SELECTED_GROUND_TRUTH = PATH_LENGTH
TASK13_GROUND_TRUTH_DEFINITION = (
    "reaction chain of exactly 7 reactions converting source functional group to target "
    "functional group via exact canonical-SMILES molecule edges, with FG detection by "
    f"RDKit SMARTS; full-dataset hub filter (>{{MAX_MOLECULE_FREQ_REFERENCE}} at "
    f"{{DATASET_TOTAL_REACTIONS}} reactions); context-local hub filter scaled from that "
    f"reference with floor {{MIN_LOCAL_MOLECULE_FREQ}}; heavy-atom window "
    f"[{{MIN_HEAVY_ATOMS}}, {{MAX_HEAVY_ATOMS}}]; only reactions present in context"
)

FIXED_FG_QUERIES: list[tuple[str, str]] = [
    ("primary_alcohol", "tertiary_amide"),
    ("primary_alcohol", "carboxylic_acid"),
    ("alkyl_halide", "tertiary_amine"),
    ("ester", "tertiary_amide"),
    ("nitrile", "primary_amide"),
]

HARDCODED_CHAINS_JSON = Path(__file__).with_name("task13_fg_hardcoded_chains.json")

HARDCODED_GT_CHAIN_COUNTS: dict[tuple[str, str], int] = {{
{chr(10).join(count_lines)}
}}

HARDCODED_GT_EXAMPLE: dict[tuple[str, str], tuple[int, ...]] = {{
{chr(10).join(example_lines)}
}}


@dataclass(frozen=True)
class FgQuestion:
    source_fg: str
    target_fg: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.source_fg, self.target_fg)


FIXED_QUESTIONS: list[FgQuestion] = [
    FgQuestion(source_fg, target_fg) for source_fg, target_fg in FIXED_FG_QUERIES
]


def fg_pair_key(source_fg: str, target_fg: str) -> str:
    return f"{{source_fg}}->{{target_fg}}"


@lru_cache(maxsize=1)
def _load_mined_payload() -> dict[str, dict[str, object]]:
    with HARDCODED_CHAINS_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hardcoded_chains_for_pair(source_fg: str, target_fg: str) -> tuple[tuple[int, ...], ...]:
    payload = _load_mined_payload()[fg_pair_key(source_fg, target_fg)]
    return tuple(tuple(chain) for chain in payload["chains"])


def full_support_indices_for_question(question: FgQuestion) -> set[int]:
    return {{
        idx
        for chain in hardcoded_chains_for_pair(question.source_fg, question.target_fg)
        for idx in chain
    }}


@dataclass(frozen=True)
class Task13ContextSampling:
    selected_chains: tuple[tuple[int, ...], ...]
    support_indices: frozenset[int]
    forced_count: int
    selected_chain_count: int


def tier3_forced_reaction_count(
    answer_count: int,
    context_size: int,
    *,
    dataset_size: int = TASK13_TOTAL_REACTIONS,
    min_selected_ground_truth: int = TASK13_MIN_SELECTED_GROUND_TRUTH,
) -> int:
    if context_size < 0:
        top_k = dataset_size
    else:
        top_k = min(context_size, dataset_size)
    if top_k == 0 or answer_count == 0:
        return 0

    ratio_scaled_floor = int((answer_count / dataset_size) * top_k)
    half_cap = top_k // 2
    return min(
        answer_count,
        half_cap,
        max(min_selected_ground_truth, ratio_scaled_floor),
    )


def chains_for_context_sampling(
    question: FgQuestion,
    context_size: int,
) -> Task13ContextSampling:
    all_chains = hardcoded_chains_for_pair(question.source_fg, question.target_fg)
    full_support = full_support_indices_for_question(question)
    forced_count = tier3_forced_reaction_count(len(full_support), context_size)
    target_chain_count = min(
        max(1, math.ceil(forced_count / PATH_LENGTH)),
        len(all_chains),
    )

    selected_chains: list[tuple[int, ...]] = []
    support_acc: set[int] = set()
    for chain in all_chains:
        if len(selected_chains) >= target_chain_count:
            break
        candidate_support = support_acc | set(chain)
        pipeline_forced = tier3_forced_reaction_count(len(candidate_support), context_size)
        if len(candidate_support) > pipeline_forced:
            continue
        selected_chains.append(chain)
        support_acc = candidate_support

    if not selected_chains:
        selected_chains = [all_chains[0]]
        support_acc = set(all_chains[0])

    selected = tuple(selected_chains)
    support_indices = frozenset(support_acc)
    return Task13ContextSampling(
        selected_chains=selected,
        support_indices=support_indices,
        forced_count=forced_count,
        selected_chain_count=len(selected),
    )


def example_chain_for_question(question: FgQuestion) -> tuple[int, ...]:
    chain = HARDCODED_GT_EXAMPLE.get(question.key)
    if chain is None:
        raise KeyError(f"No hardcoded example chain for {{question.key}}")
    if len(chain) != PATH_LENGTH:
        raise ValueError(
            f"Expected chain length {{PATH_LENGTH}}, got {{len(chain)}} for {{question.key}}"
        )
    return chain


def support_indices_for_question(question: FgQuestion) -> set[int]:
    return full_support_indices_for_question(question)


def full_dataset_chain_count(question: FgQuestion) -> int:
    return HARDCODED_GT_CHAIN_COUNTS[question.key]
'''
    GROUND_TRUTH_PY_PATH.write_text(module_text, encoding="utf-8")


def main() -> None:
    mined = mine_full_dataset()
    write_chains_json(mined)
    update_ground_truth_module(mined)
    print(f"Wrote {CHAINS_JSON_PATH}")
    print(f"Updated {GROUND_TRUTH_PY_PATH}")


if __name__ == "__main__":
    main()
