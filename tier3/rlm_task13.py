import argparse
import os
import uuid
from dataclasses import dataclass

import wandb
from rdkit import Chem
from rlm import RLM
from rlm.codeact_helpers import parse_indices, precision_recall_f1
from rlm.tracing import init_tracing, using_tracing_attributes

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
SPOT_CHECK_LIMIT = 5

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": False,
    "max_depth": 2,
}

TASK_LABEL = "Net C-O bond breaking by connectivity delta"
TASK_DESCRIPTION = (
    "A reaction breaks a C-O bond when the total number of carbon-oxygen "
    "connections in the products is less than the total number of "
    "carbon-oxygen connections in the reactants. Count connectivity only: "
    "single, double, triple, and aromatic C-O bonds each count as one C-O "
    "connection. Ignore the reagent field."
)


@dataclass
class COConnectivityResult:
    index: int
    reactant_co_count: int
    product_co_count: int
    delta: int
    is_valid: bool
    error: str | None = None


@dataclass
class GroundTruthResult:
    indices: list[int]
    total_reactions: int
    valid_reactions: int
    skipped_reactions: int
    positive_reactions: int
    results_by_index: dict[int, COConnectivityResult]


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task13_tier3",
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
        description="Run RLM task 13 C-O bond breaking connectivity evaluation."
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


def count_co_connections(mols: list[Chem.Mol]) -> int:
    count = 0
    for mol in mols:
        for bond in mol.GetBonds():
            atomic_nums = {
                bond.GetBeginAtom().GetAtomicNum(),
                bond.GetEndAtom().GetAtomicNum(),
            }
            if atomic_nums == {6, 8}:
                count += 1
    return count


def analyze_co_connectivity(indexed_line: str) -> COConnectivityResult:
    idx = -1
    try:
        idx, reactant_smiles, product_smiles = parse_reaction_sides(indexed_line)
        reactant_mols = mols_from_smiles(reactant_smiles)
        product_mols = mols_from_smiles(product_smiles)
        reactant_co_count = count_co_connections(reactant_mols)
        product_co_count = count_co_connections(product_mols)
        delta = product_co_count - reactant_co_count
        return COConnectivityResult(
            index=idx,
            reactant_co_count=reactant_co_count,
            product_co_count=product_co_count,
            delta=delta,
            is_valid=True,
        )
    except Exception as exc:
        return COConnectivityResult(
            index=idx,
            reactant_co_count=0,
            product_co_count=0,
            delta=0,
            is_valid=False,
            error=str(exc),
        )


def ground_truth_indices(lines: list[str]) -> GroundTruthResult:
    indices: list[int] = []
    results_by_index: dict[int, COConnectivityResult] = {}
    valid_reactions = 0
    skipped_reactions = 0

    for line in lines:
        result = analyze_co_connectivity(line)
        if result.index >= 0:
            results_by_index[result.index] = result
        if not result.is_valid:
            skipped_reactions += 1
            continue
        valid_reactions += 1
        if result.delta < 0:
            indices.append(result.index)

    indices.sort()
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
    - Use RDKit for all parsing and bond counting.
    - Convert reactants and products to RDKit molecules; do not count by string matching.
    - Ignore reagents in the middle field.
    - For each molecule, iterate through bonds with RDKit.
    - A C-O connection is any bond where one endpoint atom is carbon and the other endpoint atom is oxygen.
    - Count connectivity, not bond order: single, double, triple, and aromatic C-O bonds each count as one.
    - A reaction matches when product-side C-O connections minus reactant-side C-O connections is less than zero.
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
    print(f"Ground truth C-O breaking count: {gt_result.positive_reactions}")

    line_by_index: dict[int, str] = {}
    for line in lines:
        idx_str, _ = line.split(" ", 1)
        line_by_index[int(idx_str)] = line

    positives = gt_result.indices[: max(spot_check_limit, 0)]
    negatives: list[int] = []
    for idx, result in gt_result.results_by_index.items():
        if result.is_valid and result.delta >= 0:
            negatives.append(idx)
        if len(negatives) >= spot_check_limit:
            break

    print("\nPositive spot checks:")
    for idx in positives:
        result = gt_result.results_by_index[idx]
        print(
            f"{idx}: reactant_co={result.reactant_co_count} "
            f"product_co={result.product_co_count} delta={result.delta}"
        )
        print(line_by_index[idx])

    print("\nNegative spot checks:")
    for idx in negatives:
        result = gt_result.results_by_index[idx]
        print(
            f"{idx}: reactant_co={result.reactant_co_count} "
            f"product_co={result.product_co_count} delta={result.delta}"
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
        "Ground truth [co_bond_breaking] "
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
        project="RLMs-Task13_tier3",
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
            # "ground_truth_indices": gt_indices,
            "ground_truth_total_reactions": gt_result.total_reactions,
            "ground_truth_valid_reactions": gt_result.valid_reactions,
            "ground_truth_skipped_reactions": gt_result.skipped_reactions,
            "ground_truth_definition": "product C-O connectivity count - reactant C-O connectivity count < 0",
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    completion_kwargs = {"prompt": context, "root_prompt": question}
    print("Question 1/1 task=co_bond_breaking")

    with using_tracing_attributes(
        session_id=run_session_id,
        metadata={
            "sample_index": 0,
            "sample_count": 1,
            "task": "co_bond_breaking",
            "ground_truth_definition": "connectivity_delta",
        },
        tags=["run_rlms", "sample", "task13_co_bond_breaking"],
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

    # print(f"Response [co_bond_breaking]: {response}")
    print(f"Predicted [co_bond_breaking] count: {predicted_count}")
    print(f"Ground truth [co_bond_breaking] count: {ground_truth_count}")
    print(
        "Metrics [co_bond_breaking] -> "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"exact_match={is_exact_match} count_error={count_error} count_exact={count_exact}"
    )

    if iteration_metrics:
        last_metric = iteration_metrics[-1]
        iteration_input_tokens = [m["iteration_input_tokens"] for m in iteration_metrics]
        iteration_output_tokens = [m["iteration_output_tokens"] for m in iteration_metrics]
        iteration_total_tokens = [m["iteration_total_tokens"] for m in iteration_metrics]
        wandb.log(
            {
                "sample_idx": 0,
                "sample/0/reaction_key": "co_bond_breaking",
                "sample_iteration": last_metric["iteration"],
                "sample/0/iteration_input_tokens_sum": sum(iteration_input_tokens),
                "sample/0/iteration_output_tokens_sum": sum(iteration_output_tokens),
                "sample/0/iteration_total_tokens_sum": sum(iteration_total_tokens),
                "sample/0/iteration_input_tokens_max": max(iteration_input_tokens),
                "sample/0/iteration_output_tokens_max": max(iteration_output_tokens),
                "sample/0/iteration_total_tokens_max": max(iteration_total_tokens),
                "sample/0/final_total_input_tokens": last_metric["total_input_tokens"],
                "sample/0/final_total_output_tokens": last_metric["total_output_tokens"],
                "sample/0/final_total_tokens": last_metric["total_tokens"],
                "sample/0/iterations": len(iteration_metrics),
                # "sample/0/response_raw": response,
                # "sample/0/response_parsed_indices": ",".join(str(x) for x in parsed_indices),
                "sample/0/response_parsed_count": predicted_count,
                # "sample/0/ground_truth_indices": ",".join(str(x) for x in gt_indices),
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
    run.summary["ground_truth/co_bond_breaking/count"] = ground_truth_count
    # run.summary["ground_truth/co_bond_breaking/indices"] = ",".join(str(x) for x in gt_indices)
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
