"""Generate the exhaustive full-dataset ground truth for Tier-4 Task 12b."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat

from rdkit import rdBase
from task12b_hub_molecule_graph import hub_molecules_in_context


def indexed_lines(dataset: Path) -> list[str]:
    reactions = [line.strip() for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    return [f"{index} {reaction}" for index, reaction in enumerate(reactions)]


def write_module(output: Path, molecules: list[str], record_count: int) -> None:
    output.write_text(
        '"""Generated exhaustive full-dataset ground truth for Tier-4 Task 12b.\n\n'
        "Do not edit manually. Regenerate with generate_task12b_full_ground_truth.py.\n"
        '"""\n\n'
        f'TASK12B_FULL_DATASET_RDKIT_VERSION = "{rdBase.rdkitVersion}"\n'
        f"TASK12B_FULL_DATASET_RECORDS = {record_count}\n"
        f"TASK12B_FULL_DATASET_HUB_COUNT = {len(molecules)}\n"
        'TASK12B_FULL_DATASET_CANONICALIZATION = "RDKit Chem.CanonSmiles; isomeric canonical SMILES"\n\n'
        "TASK12B_FULL_DATASET_HUB_MOLECULES = " + pformat(tuple(molecules), width=100) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("task12b_full_dataset_ground_truth.py"),
    )
    args = parser.parse_args()
    lines = indexed_lines(args.dataset)
    molecules = hub_molecules_in_context(lines, min_downstream=3, dag_mode="index_asc")
    write_module(args.output, molecules, len(lines))
    print(
        f"Wrote {args.output}: records={len(lines)} hubs={len(molecules)} "
        f"rdkit={rdBase.rdkitVersion}"
    )


if __name__ == "__main__":
    main()
