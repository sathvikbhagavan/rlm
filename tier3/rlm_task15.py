import argparse
import math
import os
import uuid
from dataclasses import dataclass
from time import perf_counter

import wandb
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rlm import RLM
from rlm.codeact_helpers import parse_indices, precision_recall_f1
from rlm.tracing import init_tracing, using_tracing_attributes

os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = False
SPOT_CHECK_LIMIT = 5
MIN_CORE_ATOMS = 3
MIN_CORE_FRACTION = 0.5
MCS_TIMEOUT_SECONDS = 2
GT_PROGRESS_EVERY = 2000

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}

TASK_LABEL = "Achiral substrates to chiral products"
TASK_DESCRIPTION = (
    "A reaction matches when all reactant-side molecules are achiral, and at least one "
    "product-side stereocenter exists on the product atoms that belong to the preserved "
    "core of the largest reactant (substrate). Ignore reagents in the middle field."
)


@dataclass
class AchiralToChiralResult:
    index: int
    substrate_heavy_atoms: int
    product_heavy_atoms: int
    all_reactants_achiral: bool
    product_chiral_center_count: int
    preserved_core_atom_count: int
    preserved_core_chiral_center_count: int
    is_valid: bool
    is_positive: bool
    error: str | None = None


@dataclass
class GroundTruthResult:
    indices: list[int]
    total_reactions: int
    valid_reactions: int
    skipped_reactions: int
    positive_reactions: int
    results_by_index: dict[int, AchiralToChiralResult]


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task15_tier3",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RLM task 15 achiral-to-chiral product evaluation."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for backend (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DATASET_PATH,
        help=f"Path to reaction dataset (default: {DATASET_PATH}).",
    )
    parser.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="Only compute and print ground-truth diagnostics without running RLM.",
    )
    parser.add_argument(
        "--spot-check-limit",
        type=int,
        default=SPOT_CHECK_LIMIT,
        help=f"Number of positive/negative examples to print in GT-only mode (default: {SPOT_CHECK_LIMIT}).",
    )
    return parser.parse_args()


def load_lines(dataset_path: str) -> list[str]:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        raw_lines = [line.strip() for line in handle if line.strip()]
    return [f"{i} {line}" for i, line in enumerate(raw_lines)]


def parse_reaction_sides(indexed_line: str) -> tuple[int, list[str], list[str]]:
    idx_str, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    if len(parts) != 3:
        raise ValueError("Reaction must have reactants>reagents>products format.")
    reactant_smiles = [s for s in parts[0].split(".") if s]
    product_smiles = [s for s in parts[2].split(".") if s]
    return int(idx_str), reactant_smiles, product_smiles


def mols_from_smiles(smiles_list: list[str]) -> list[Chem.Mol]:
    mols: list[Chem.Mol] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        mols.append(mol)
    return mols


def find_chiral_centers(mol: Chem.Mol) -> list[tuple[int, str]]:
    return Chem.FindMolChiralCenters(
        mol,
        includeUnassigned=True,
        useLegacyImplementation=False,
    )


def get_preserved_core_product_atoms(substrate: Chem.Mol, product: Chem.Mol) -> set[int]:
    substrate_heavy = substrate.GetNumHeavyAtoms()
    if substrate_heavy == 0:
        return set()

    mcs_params = rdFMCS.MCSParameters()
    mcs_params.Timeout = MCS_TIMEOUT_SECONDS
    mcs_params.AtomCompareParameters.MatchChiralTag = False
    mcs = rdFMCS.FindMCS([substrate, product], parameters=mcs_params)
    if mcs.canceled or mcs.numAtoms == 0:
        return set()

    min_core_atoms = max(MIN_CORE_ATOMS, math.ceil(MIN_CORE_FRACTION * substrate_heavy))
    if mcs.numAtoms < min_core_atoms:
        return set()

    core_pattern = Chem.MolFromSmarts(mcs.smartsString)
    if core_pattern is None:
        return set()

    matches = product.GetSubstructMatches(core_pattern, useChirality=False)
    if not matches:
        return set()

    product_atom_indices: set[int] = set()
    for match in matches:
        product_atom_indices.update(match)
    return product_atom_indices


def analyze_achiral_to_chiral(indexed_line: str) -> AchiralToChiralResult:
    idx = -1
    try:
        idx, reactant_smiles, product_smiles = parse_reaction_sides(indexed_line)
        reactants = mols_from_smiles(reactant_smiles)
        products = mols_from_smiles(product_smiles)
        if not reactants or not products:
            raise ValueError("Reaction must have at least one valid reactant and product.")

        all_reactants_achiral = all(len(find_chiral_centers(mol)) == 0 for mol in reactants)
        substrate = max(reactants, key=lambda mol: mol.GetNumHeavyAtoms())
        product = max(products, key=lambda mol: mol.GetNumHeavyAtoms())

        product_chiral_centers = find_chiral_centers(product)

        # Fast-fail negatives before expensive MCS matching.
        if not all_reactants_achiral or len(product_chiral_centers) == 0:
            return AchiralToChiralResult(
                index=idx,
                substrate_heavy_atoms=substrate.GetNumHeavyAtoms(),
                product_heavy_atoms=product.GetNumHeavyAtoms(),
                all_reactants_achiral=all_reactants_achiral,
                product_chiral_center_count=len(product_chiral_centers),
                preserved_core_atom_count=0,
                preserved_core_chiral_center_count=0,
                is_valid=True,
                is_positive=False,
            )

        preserved_core_product_atoms = get_preserved_core_product_atoms(
            substrate=substrate,
            product=product,
        )
        preserved_core_chiral_centers = [
            atom_idx
            for atom_idx, _ in product_chiral_centers
            if atom_idx in preserved_core_product_atoms
        ]

        is_positive = (
            all_reactants_achiral
            and len(product_chiral_centers) > 0
            and len(preserved_core_product_atoms) > 0
            and len(preserved_core_chiral_centers) > 0
        )

        return AchiralToChiralResult(
            index=idx,
            substrate_heavy_atoms=substrate.GetNumHeavyAtoms(),
            product_heavy_atoms=product.GetNumHeavyAtoms(),
            all_reactants_achiral=all_reactants_achiral,
            product_chiral_center_count=len(product_chiral_centers),
            preserved_core_atom_count=len(preserved_core_product_atoms),
            preserved_core_chiral_center_count=len(preserved_core_chiral_centers),
            is_valid=True,
            is_positive=is_positive,
        )
    except Exception as exc:
        return AchiralToChiralResult(
            index=idx,
            substrate_heavy_atoms=0,
            product_heavy_atoms=0,
            all_reactants_achiral=False,
            product_chiral_center_count=0,
            preserved_core_atom_count=0,
            preserved_core_chiral_center_count=0,
            is_valid=False,
            is_positive=False,
            error=str(exc),
        )


def ground_truth_indices(lines: list[str]) -> GroundTruthResult:
    indices: list[int] = []
    results_by_index: dict[int, AchiralToChiralResult] = {}
    valid_reactions = 0
    skipped_reactions = 0
    total_lines = len(lines)
    start_time = perf_counter()

    print(
        f"[GT] Starting ground-truth computation for {total_lines} reactions "
        f"(progress every {GT_PROGRESS_EVERY})."
    )

    for processed, line in enumerate(lines, start=1):
        result = analyze_achiral_to_chiral(line)
        if result.index >= 0:
            results_by_index[result.index] = result
        if not result.is_valid:
            skipped_reactions += 1
            continue
        valid_reactions += 1
        if result.is_positive:
            indices.append(result.index)

        if processed % GT_PROGRESS_EVERY == 0 or processed == total_lines:
            elapsed_s = perf_counter() - start_time
            print(
                f"[GT] {processed}/{total_lines} processed | "
                f"positives={len(indices)} valid={valid_reactions} skipped={skipped_reactions} "
                f"elapsed={elapsed_s:.1f}s"
            )

    indices.sort()
    elapsed_s = perf_counter() - start_time
    print(
        f"[GT] Completed in {elapsed_s:.1f}s | "
        f"positives={len(indices)} valid={valid_reactions} skipped={skipped_reactions}"
    )
    return GroundTruthResult(
        indices=indices,
        total_reactions=len(lines),
        valid_reactions=valid_reactions,
        skipped_reactions=skipped_reactions,
        positive_reactions=len(indices),
        results_by_index=results_by_index,
    )


def build_question() -> str:
    return f"""
    Context: You are given a large string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Task:
    Return all reaction indices that match this reaction type:
    - {TASK_LABEL}

    Description:
    - {TASK_DESCRIPTION}

    Guidance:
    - Use RDKit for all parsing and stereochemistry analysis.
    - Ignore reagents in the middle field.
    - For each reaction, consider reactant-side molecules as substrates.
    - All reactant-side molecules must be achiral (no assigned or unassigned stereocenters).
    - Select the largest reactant (by heavy atoms) as substrate and the largest product as main product.
    - The main product must contain at least one stereocenter (assigned or unassigned).
    - Confirm substrate-to-product continuity using an RDKit maximum common substructure (MCS) between substrate and main product.
    - Require at least one product stereocenter to lie on atoms belonging to that preserved substrate core.
    - Skip malformed reactions.
    - DO NOT assume/simulate output of the code. Wait for the code to get executed and only then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.

    Output format:
    - Return ONLY the matching reaction INDICES.
    - Format must be a comma-separated list of integers in ascending order (e.g., 3,8,21).
    - No other text, quotes, labels, punctuation, or formatting.
    - If no matching reaction exists, return -1.
    """


def print_ground_truth_diagnostics(
    lines: list[str],
    gt_result: GroundTruthResult,
    spot_check_limit: int,
) -> None:
    print(f"Total reactions: {gt_result.total_reactions}")
    print(f"Valid reactions: {gt_result.valid_reactions}")
    print(f"Skipped malformed reactions: {gt_result.skipped_reactions}")
    print(f"Ground truth achiral-to-chiral count: {gt_result.positive_reactions}")

    line_by_index: dict[int, str] = {}
    for line in lines:
        idx_str, _ = line.split(" ", 1)
        line_by_index[int(idx_str)] = line

    positives = gt_result.indices[: max(spot_check_limit, 0)]
    negatives: list[int] = []
    for idx, result in gt_result.results_by_index.items():
        if result.is_valid and not result.is_positive:
            negatives.append(idx)
        if len(negatives) >= spot_check_limit:
            break

    print("\nPositive spot checks:")
    for idx in positives:
        result = gt_result.results_by_index[idx]
        print(
            f"{idx}: reactants_achiral={result.all_reactants_achiral} "
            f"product_chiral={result.product_chiral_center_count} "
            f"core_atoms={result.preserved_core_atom_count} "
            f"core_chiral={result.preserved_core_chiral_center_count}"
        )
        print(line_by_index[idx])

    print("\nNegative spot checks:")
    for idx in negatives:
        result = gt_result.results_by_index[idx]
        print(
            f"{idx}: reactants_achiral={result.all_reactants_achiral} "
            f"product_chiral={result.product_chiral_center_count} "
            f"core_atoms={result.preserved_core_atom_count} "
            f"core_chiral={result.preserved_core_chiral_center_count}"
        )
        print(line_by_index[idx])


def main(model_name: str, dataset_path: str, ground_truth_only: bool, spot_check_limit: int) -> None:
    lines = load_lines(dataset_path)
    context = "\n".join(lines)
    question = build_question()
    gt_result = ground_truth_indices(lines)
    gt_indices = gt_result.indices
    gt_set = set(gt_indices)

    print(
        "Ground truth [achiral_to_chiral] "
        f"count={gt_result.positive_reactions} "
        f"valid={gt_result.valid_reactions} "
        f"skipped={gt_result.skipped_reactions}"
    )

    if ground_truth_only:
        print_ground_truth_diagnostics(
            lines=lines,
            gt_result=gt_result,
            spot_check_limit=spot_check_limit,
        )
        return

    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    run = wandb.init(
        project="RLMs-Task15_tier3",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": dataset_path,
            "num_questions": 1,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": TASK_DESCRIPTION,
            "task_label": TASK_LABEL,
            "ground_truth_count": gt_result.positive_reactions,
            "ground_truth_total_reactions": gt_result.total_reactions,
            "ground_truth_valid_reactions": gt_result.valid_reactions,
            "ground_truth_skipped_reactions": gt_result.skipped_reactions,
            "ground_truth_definition": (
                "all reactants achiral AND product has at least one stereocenter on "
                "the MCS-preserved substrate core"
            ),
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    completion_kwargs = {"prompt": context, "root_prompt": question}
    print("Question 1/1 task=achiral_to_chiral")

    with using_tracing_attributes(
        session_id=run_session_id,
        metadata={
            "sample_index": 0,
            "sample_count": 1,
            "task": "achiral_to_chiral",
            "ground_truth_definition": "stereocenter_on_preserved_core",
        },
        tags=["run_rlms", "sample", "task15_achiral_to_chiral"],
    ):
        completion = rlm.completion(**completion_kwargs)
        response = completion.response

    iteration_metrics = rlm.get_last_iteration_metrics()
    parsed_indices = parse_indices(response)
    pred_set = set(parsed_indices)
    precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
    predicted_count = len(pred_set)
    ground_truth_count = len(gt_set)
    count_error = abs(predicted_count - ground_truth_count)
    count_exact = int(predicted_count == ground_truth_count)
    sample_cost_usd = completion.usage_summary.total_cost
    is_exact_match = pred_set == gt_set

    print(f"Response [achiral_to_chiral]: {response}")
    print(f"Predicted [achiral_to_chiral] count: {predicted_count}")
    print(f"Ground truth [achiral_to_chiral] count: {ground_truth_count}")
    print(
        "Metrics [achiral_to_chiral] -> "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"exact_match={is_exact_match} count_error={count_error} count_exact={count_exact}"
    )

    for metric in iteration_metrics:
        wandb.log(
            {
                "sample_iteration": metric["iteration"],
                "sample/0/iteration_input_tokens": metric["iteration_input_tokens"],
                "sample/0/iteration_output_tokens": metric["iteration_output_tokens"],
                "sample/0/iteration_total_tokens": metric["iteration_total_tokens"],
            }
        )

    if iteration_metrics:
        last_metric = iteration_metrics[-1]
        wandb.log(
            {
                "sample_idx": 0,
                "sample/0/reaction_key": "achiral_to_chiral",
                "sample/0/final_total_input_tokens": last_metric["total_input_tokens"],
                "sample/0/final_total_output_tokens": last_metric["total_output_tokens"],
                "sample/0/final_total_tokens": last_metric["total_tokens"],
                "sample/0/iterations": len(iteration_metrics),
                "sample/0/response_parsed_count": predicted_count,
                "sample/0/ground_truth_count": ground_truth_count,
                "sample/0/ground_truth_valid_reactions": gt_result.valid_reactions,
                "sample/0/ground_truth_skipped_reactions": gt_result.skipped_reactions,
                "sample/0/precision": precision,
                "sample/0/recall": recall,
                "sample/0/f1": f1,
                "sample/0/is_exact_match": int(is_exact_match),
                "sample/0/predicted_count": predicted_count,
                "sample/0/count_error": count_error,
                "sample/0/count_exact": count_exact,
                "sample/0/completion_root_prompt": question,
                "sample/0/completion_prompt_char_count": len(context),
                **(
                    {"sample/0/final_total_cost_usd": sample_cost_usd}
                    if sample_cost_usd is not None
                    else {}
                ),
            }
        )

    run.summary["exact_match_correct"] = int(is_exact_match)
    run.summary["total"] = 1
    run.summary["exact_match_accuracy"] = float(is_exact_match)
    run.summary["macro_precision"] = precision
    run.summary["macro_recall"] = recall
    run.summary["macro_f1"] = f1
    run.summary["ground_truth/achiral_to_chiral/count"] = ground_truth_count
    run.summary["ground_truth/total_reactions"] = gt_result.total_reactions
    run.summary["ground_truth/valid_reactions"] = gt_result.valid_reactions
    run.summary["ground_truth/skipped_reactions"] = gt_result.skipped_reactions
    run.summary["predicted_count"] = predicted_count
    run.summary["count_error"] = count_error
    run.summary["count_exact"] = count_exact
    run.summary["samples_with_cost"] = int(sample_cost_usd is not None)
    if sample_cost_usd is not None:
        run.summary["total_cost_usd"] = sample_cost_usd
        run.summary["avg_cost_per_sample_usd"] = sample_cost_usd
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        ground_truth_only=args.ground_truth_only,
        spot_check_limit=args.spot_check_limit,
    )
