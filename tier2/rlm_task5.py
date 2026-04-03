import os
import uuid
import argparse
from typing import Optional

import wandb
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rlm import RLM
from rlm.tracing import init_tracing, using_tracing_attributes

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
WEIGHT_THRESHOLDS_DA = [100, 150]
RING_X_VALUES = [1, 2]

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 1,
}


def parse_count(response: str) -> Optional[int]:
    response = response.strip().replace('"', "").replace("'", "")
    if response.isdigit():
        return int(response)
    for token in response.replace(",", " ").split():
        if token.isdigit():
            return int(token)
    return None


def parse_reaction_sides(indexed_line: str) -> tuple[str, str]:
    _, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    reactant_side = parts[0].strip()
    product_side = parts[-1].strip()
    return reactant_side, product_side


def heaviest_component_weight(side_smiles: str) -> Optional[float]:
    if not side_smiles:
        return None
    weights: list[float] = []
    for comp in side_smiles.split("."):
        comp = comp.strip()
        if not comp:
            continue
        mol = Chem.MolFromSmiles(comp)
        if mol is None:
            continue
        weights.append(Descriptors.MolWt(mol))
    if not weights:
        return None
    return max(weights)


def component_ring_counts(side_smiles: str) -> list[int]:
    if not side_smiles:
        return []
    ring_counts: list[int] = []
    for comp in side_smiles.split("."):
        comp = comp.strip()
        if not comp:
            continue
        mol = Chem.MolFromSmiles(comp)
        if mol is None:
            continue
        ring_counts.append(int(rdMolDescriptors.CalcNumRings(mol)))
    return ring_counts


def reaction_delta_weight(indexed_line: str) -> Optional[float]:
    reactants, products = parse_reaction_sides(indexed_line)
    heaviest_reactant = heaviest_component_weight(reactants)
    heaviest_product = heaviest_component_weight(products)
    if heaviest_reactant is None or heaviest_product is None:
        return None
    return heaviest_product - heaviest_reactant


def reaction_delta_rings(indexed_line: str) -> Optional[int]:
    reactants, products = parse_reaction_sides(indexed_line)
    reactant_ring_counts = component_ring_counts(reactants)
    product_ring_counts = component_ring_counts(products)
    if not reactant_ring_counts or not product_ring_counts:
        return None
    pairwise_deltas = [
        product_rings - reactant_rings
        for product_rings in product_ring_counts
        for reactant_rings in reactant_ring_counts
    ]
    return max(pairwise_deltas)


def ground_truth_count_for_combo(
    weight_deltas: list[Optional[float]],
    ring_deltas: list[Optional[int]],
    weight_threshold: int,
    ring_x: int,
) -> int:
    count = 0
    for w_delta, r_delta in zip(weight_deltas, ring_deltas):
        if w_delta is None or r_delta is None:
            continue
        if w_delta > weight_threshold and r_delta == ring_x:
            count += 1
    return count


def build_question(weight_threshold: int, ring_x: int) -> str:
    return f"""
    Context is a big string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Count how many reactions satisfy BOTH conditions:
      1) weight(heaviest product) - weight(heaviest reactant) > {weight_threshold} Da
      2) max over all pairs [rings(product_component) - rings(reactant_component)] == {ring_x}

    Guidance:
    - You may use RDKit functions if needed (for example, Chem.MolFromSmiles, Descriptors.MolWt, rdMolDescriptors.CalcNumRings).
    - "heaviest" means the largest molecular weight among dot-separated molecules on that side.
    - For ring condition, split each side by dot and compute ring count for each valid component.
    - For each reaction, compute ALL pairwise ring deltas:
      rings(product_component) - rings(reactant_component)
      and use the maximum of those deltas.
    - Ignore reagents (middle field).
    - For each side (reactants/products), ignore invalid or empty dot-separated molecules.
    - Skip a reaction only if reactant side or product side has no valid molecules left after filtering.

    Output format:
    - Report ONLY the INTEGER COUNT of matching reactions (e.g., 57)
    - No other text, quotes, labels, punctuation, or formatting.
    - If no matching reaction exists, report 0.
    """


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="rlm-tracing-dev",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLM task 5 evaluation.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for backend (default: {MODEL_NAME}).",
    )
    return parser.parse_args()


def main(model_name: str) -> None:
    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]

    weight_deltas: list[Optional[float]] = []
    ring_deltas: list[Optional[int]] = []
    for line in lines:
        weight_deltas.append(reaction_delta_weight(line))
        ring_deltas.append(reaction_delta_rings(line))

    combinations = [
        (weight_threshold, ring_x)
        for weight_threshold in WEIGHT_THRESHOLDS_DA
        for ring_x in RING_X_VALUES
    ]

    ground_truth_count_by_combo: dict[tuple[int, int], int] = {}
    print("Ground truth counts for task5 combinations:")
    for weight_threshold, ring_x in combinations:
        count = ground_truth_count_for_combo(
            weight_deltas=weight_deltas,
            ring_deltas=ring_deltas,
            weight_threshold=weight_threshold,
            ring_x=ring_x,
        )
        ground_truth_count_by_combo[(weight_threshold, ring_x)] = count
        print(f"  weight_delta>{weight_threshold} & ring_delta=={ring_x}: {count}")

    questions = [build_question(w, x) for w, x in combinations]
    context = "\n".join(lines)

    run = wandb.init(
        project="RLMs-Task5",
        config={
            "MODEL_NAME": model_name,
            "weight_thresholds_da": WEIGHT_THRESHOLDS_DA,
            "ring_x_values": RING_X_VALUES,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "num_questions": len(questions),
            "rlm_init_kwargs": rlm_init_kwargs,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    correct = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, (weight_threshold, ring_x) in enumerate(combinations):
        question = questions[i]
        print(
            f"Question {i + 1}/{len(questions)} for "
            f"weight>{weight_threshold} and ring=={ring_x}"
        )
        completion_kwargs = {"prompt": context, "root_prompt": question}
        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(questions),
                "weight_threshold_da": weight_threshold,
                "ring_x": ring_x,
            },
            tags=["run_rlms", "sample", "task5_combined"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response
        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed_count = parse_count(response)
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        ground_truth_count = ground_truth_count_by_combo[(weight_threshold, ring_x)]
        is_correct = parsed_count == ground_truth_count
        if is_correct:
            correct += 1
        else:
            print(
                f"Mismatch for weight>{weight_threshold} and ring=={ring_x}"
            )
            print(f"Predicted count: {parsed_count}")
            print(f"Ground truth count: {ground_truth_count}")
            print(f"Raw response: {response}")
            print("--------------------------------")

        for metric in iteration_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                    f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                    f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                }
            )

        if iteration_metrics:
            last_metric = iteration_metrics[-1]
            wandb.log(
                {
                    "sample_idx": i,
                    f"sample/{i}/weight_threshold_da": weight_threshold,
                    f"sample/{i}/ring_x": ring_x,
                    f"sample/{i}/final_total_input_tokens": last_metric["total_input_tokens"],
                    f"sample/{i}/final_total_output_tokens": last_metric["total_output_tokens"],
                    f"sample/{i}/final_total_tokens": last_metric["total_tokens"],
                    f"sample/{i}/iterations": len(iteration_metrics),
                    f"sample/{i}/is_correct": int(is_correct),
                    f"sample/{i}/ground_truth_count": ground_truth_count,
                    f"sample/{i}/prediction_count": parsed_count
                    if parsed_count is not None
                    else -1,
                    f"sample/{i}/response_raw": response,
                    f"sample/{i}/response_parsed_count": parsed_count
                    if parsed_count is not None
                    else -1,
                    f"sample/{i}/completion_root_prompt": question,
                    f"sample/{i}/completion_prompt_char_count": len(context),
                    **(
                        {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                        if sample_cost_usd is not None
                        else {}
                    ),
                }
            )

        wandb.log({"running_accuracy": correct / (i + 1)})

    total = len(questions)
    accuracy = (correct / total) if total else 0.0
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f}")

    for weight_threshold, ring_x in combinations:
        key = f"ground_truth/weight_gt_{weight_threshold}/ring_eq_{ring_x}/count"
        run.summary[key] = ground_truth_count_by_combo[(weight_threshold, ring_x)]

    run.summary["correct"] = correct
    run.summary["total"] = total
    run.summary["accuracy"] = accuracy
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(model_name=args.model_name)
