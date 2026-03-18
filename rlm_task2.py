import os
import uuid
from typing import Optional

import wandb
from rdkit import Chem
from rdkit.Chem import Descriptors
from rlm import RLM
from rlm.tracing import init_tracing, using_tracing_attributes

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/workspace/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
THRESHOLDS = list(range(100, 300, 10))

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 1,
}


def parse_count(response: str) -> Optional[int]:
    # Fallback parsing when the model returns extra tokens/quotes.
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


def reaction_delta_weight(indexed_line: str) -> Optional[float]:
    reactants, products = parse_reaction_sides(indexed_line)
    heaviest_reactant = heaviest_component_weight(reactants)
    heaviest_product = heaviest_component_weight(products)
    if heaviest_reactant is None or heaviest_product is None:
        return None
    return heaviest_product - heaviest_reactant


def build_question(threshold: int) -> str:
    return f"""
    Context is a big string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Count how many reactions satisfy:
      weight(heaviest product) - weight(heaviest reactant) > {threshold} Da

    Guidance:
    - You may use RDKit functions if needed (for example, Chem.MolFromSmiles and Descriptors.MolWt).
    - "heaviest" means the largest molecular weight among dot-separated molecules on that side.
    - Ignore reagents (middle field).
    - For each side (reactants/products), ignore invalid or empty dot-separated molecules instead of treating them as 0.0.
    - Compute the side's heaviest weight from the remaining valid molecules only.
    - Skip a reaction only if reactant side or product side has no valid molecules left after filtering.
    - Do NOT assign 0.0 as a fallback for missing/invalid molecules.

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


def main() -> None:
    maybe_init_tracing()
    rlm = RLM(**RLM_INIT_KWARGS)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]

    # Precompute RDKit-based deltas once, then derive ground-truth counts for each threshold.
    delta_by_index: dict[int, float] = {}
    for i, line in enumerate(lines):
        delta = reaction_delta_weight(line)
        if delta is not None:
            delta_by_index[i] = delta

    ground_truth_count_by_threshold: dict[int, int] = {}
    for x in THRESHOLDS:
        count = sum(1 for delta in delta_by_index.values() if delta > x)
        ground_truth_count_by_threshold[x] = count

    questions = [build_question(x) for x in THRESHOLDS]
    context = "\n".join(lines)

    run = wandb.init(
        project="RLMs-Task2",
        config={
            "MODEL_NAME": MODEL_NAME,
            "thresholds": THRESHOLDS,
            "backend": BACKEND,
            "model_name": MODEL_NAME,
            "dataset_path": DATASET_PATH,
            "num_questions": len(questions),
            "rlm_init_kwargs": RLM_INIT_KWARGS,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    correct = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, threshold in enumerate(THRESHOLDS):
        question = questions[i]
        print(f"Question {i + 1}/{len(questions)} for X={threshold}")
        completion_kwargs = {"prompt": context, "root_prompt": question}
        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(questions),
                "threshold": threshold,
            },
            tags=["run_rlms", "sample", "delta_weight"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response
        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed_count = parse_count(response)
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        ground_truth_count = ground_truth_count_by_threshold[threshold]
        is_correct = parsed_count == ground_truth_count
        if is_correct:
            correct += 1
        else:
            print(f"Mismatch for X={threshold}")
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
                    f"sample/{i}/threshold_x": threshold,
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

    # Persist RDKit ground-truth counts for all thresholds in run summary.
    for x in THRESHOLDS:
        run.summary[f"ground_truth/x_{x}/count"] = ground_truth_count_by_threshold[x]

    run.summary["correct"] = correct
    run.summary["total"] = total
    run.summary["accuracy"] = accuracy
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    main()
