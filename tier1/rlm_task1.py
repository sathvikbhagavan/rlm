import os
import random
import uuid

import wandb
from rlm import RLM
from rlm.codeact_helpers import build_context_pipeline, precision_recall_f1
from rlm.tracing import init_tracing, using_tracing_attributes
from task1_hardcoded_cases import (
    TASK1_HARDCODED_GROUND_TRUTH_INDICES,
    TASK1_HARDCODED_PRODUCTS,
)

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
SEED = 42
NUM_QUESTIONS = 10
ENABLE_TRACING = True
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"
MIN_SELECTED_GROUND_TRUTH = 5

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}


def parse_indices(response: str) -> list[int]:
    # This is fallback parsing when the model returns with quotes
    response = response.strip().replace('"', "").replace("'", "")
    if response == "-1":
        return []
    if response.isdigit():
        return [int(response)]
    return [int(num.strip()) for num in response.split(",") if num.strip().isdigit()]


def extract_product(indexed_line: str) -> str:
    _, reaction_smiles = indexed_line.split(" ", 1)
    return reaction_smiles.split(">")[-1].strip()


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task1",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def build_product_index_map(indexed_lines: list[str]) -> dict[str, set[int]]:
    product_to_indices: dict[str, set[int]] = {}
    for line in indexed_lines:
        idx_str, _ = line.split(" ", 1)
        product = extract_product(line)
        product_to_indices.setdefault(product, set()).add(int(idx_str))
    return product_to_indices


def build_question(product: str) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Find all the indices of the reactions for the following PRODUCT
    (and not the reactants/reagents): {product}

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If the product is not found, report: -1
"""


def main() -> None:
    maybe_init_tracing()
    rlm = RLM(**RLM_INIT_KWARGS)
    run_session_id = f"run-rlms-{uuid.uuid4()}"
    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]
    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=random.Random(SEED),
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )
    selected_products = TASK1_HARDCODED_PRODUCTS
    selected_ground_truth = TASK1_HARDCODED_GROUND_TRUTH_INDICES
    questions = [build_question(product) for product in selected_products]
    print(f"[QUESTION-SAMPLING] using_hardcoded_products={len(selected_products)}")

    run = wandb.init(
        project="RLMs-Task1",
        config={
            "MODEL_NAME": MODEL_NAME,
            "SEED": SEED,
            "NUM_QUESTIONS": NUM_QUESTIONS,
            "backend": BACKEND,
            "model_name": MODEL_NAME,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": CONTEXT_SIZE,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "min_selected_ground_truth": MIN_SELECTED_GROUND_TRUTH,
            "num_questions_requested": NUM_QUESTIONS,
            "num_questions": len(questions),
            "rlm_init_kwargs": RLM_INIT_KWARGS,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, question in enumerate(questions):
        print(f"Question {i + 1}/{len(questions)}")
        target_product = selected_products[i]
        ground_truth_index_set = set(selected_ground_truth[i])
        target_index = sorted(ground_truth_index_set)[0]
        target_line = lines[target_index]
        sample_context = context_pipeline.build_context(
            context_size=CONTEXT_SIZE,
            correct_indices=ground_truth_index_set,
            query=target_product,
        )
        context_lines = [line for line in sample_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in context_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        ground_truth_in_context_set = ground_truth_index_set & context_indices
        gt_in_context_count = len(ground_truth_in_context_set)
        print(
            f"[CONTEXT] requested_size={CONTEXT_SIZE} actual_size={len(context_lines)} "
            f"ground_truth_in_context={gt_in_context_count}/{len(ground_truth_index_set)}"
        )
        completion_kwargs = {"prompt": sample_context, "root_prompt": question}
        # Group all sample traces under one run session while keeping each sample as a distinct trace.
        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(questions),
                "target_index": target_index,
            },
            tags=["run_rlms", "sample"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response
        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed = parse_indices(response)
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        predicted_index_set = set(parsed)
        precision, recall, f1 = precision_recall_f1(predicted_index_set, ground_truth_in_context_set)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

        if f1 < 1.0:
            print(
                f"Mismatch for target_index={target_index}: "
                f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
            )
            print(f"Line in context: {target_line}")
            print(f"Product: {target_product}")
            print(f"Predicted indices: {sorted(predicted_index_set)}")
            print(f"Ground truth indices (in context): {sorted(ground_truth_in_context_set)}")
            print("--------------------------------")
        else:
            print(f'F1 is 1.0 for target_index={target_index}')

        for metric in iteration_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    # Requirement 1: token variation per iteration (sample-scoped)
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
                    # Requirement 2: final totals per sample
                    f"sample/{i}/final_total_input_tokens": last_metric["total_input_tokens"],
                    f"sample/{i}/final_total_output_tokens": last_metric["total_output_tokens"],
                    f"sample/{i}/final_total_tokens": last_metric["total_tokens"],
                    f"sample/{i}/iterations": len(iteration_metrics),
                    f"sample/{i}/precision": precision,
                    f"sample/{i}/recall": recall,
                    f"sample/{i}/f1": f1,
                    f"sample/{i}/target_index": target_index,
                    f"sample/{i}/completion_prompt_char_count": len(sample_context),
                    f"sample/{i}/context_size": CONTEXT_SIZE,
                    **(
                        {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                        if sample_cost_usd is not None
                        else {}
                    ),
                }
            )

        wandb.log(
            {
                "running_precision": precision_sum / (i + 1),
                "running_recall": recall_sum / (i + 1),
                "running_f1": f1_sum / (i + 1),
            }
        )

    total = len(questions)
    avg_precision = (precision_sum / total) if total else 0.0
    avg_recall = (recall_sum / total) if total else 0.0
    avg_f1 = (f1_sum / total) if total else 0.0
    print(f"Macro Precision: {avg_precision:.4f}")
    print(f"Macro Recall: {avg_recall:.4f}")
    print(f"Macro F1: {avg_f1:.4f}")

    run.summary["total"] = total
    run.summary["macro_precision"] = avg_precision
    run.summary["macro_recall"] = avg_recall
    run.summary["macro_f1"] = avg_f1
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    main()
