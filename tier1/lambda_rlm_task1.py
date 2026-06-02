import argparse
import random
import sys
from pathlib import Path
import os

import wandb
os.environ["WANDB_MODE"] = "disabled"


# Prefer lambda-RLM implementation (same package name: `rlm`) over local `rlm` repo.
LAMBDA_RLM_REPO_ROOT = Path(__file__).resolve().parents[2] / "lambda-RLM"
sys.path.insert(0, str(LAMBDA_RLM_REPO_ROOT))

from rlm import LambdaRLM
from rlm.codeact_helpers import build_context_pipeline, precision_recall_f1


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "x-ai/grok-4-fast"
SEED = 42
NUM_QUESTIONS = 1
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"

LAMBDA_RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
}


def parse_indices(response: str) -> list[int]:
    # Fallback parsing for outputs with accidental quotes or extra whitespace.
    response = response.strip().replace('"', "").replace("'", "")
    if response == "-1":
        return []
    if response.isdigit():
        return [int(response)]
    return [int(num.strip()) for num in response.split(",") if num.strip().isdigit()]


def extract_product(indexed_line: str) -> str:
    _, reaction_smiles = indexed_line.split(" ", 1)
    return reaction_smiles.split(">")[-1].strip()


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


def build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 1 with LambdaRLM.")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH)
    parser.add_argument("--backend", type=str, default=BACKEND)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--num-questions", type=int, default=NUM_QUESTIONS)
    parser.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_product_index_map(indexed_lines: list[str]) -> dict[str, set[int]]:
    product_to_indices: dict[str, set[int]] = {}
    for line in indexed_lines:
        idx_str, _ = line.split(" ", 1)
        product = extract_product(line)
        product_to_indices.setdefault(product, set()).add(int(idx_str))
    return product_to_indices


def main() -> None:
    args = parse_args()
    lambda_rlm = LambdaRLM(
        backend=args.backend,
        backend_kwargs={"model_name": args.model_name},
        verbose=args.verbose or LAMBDA_RLM_INIT_KWARGS["verbose"],
    )

    with open(args.dataset_path, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]
    product_index_map = build_product_index_map(lines)
    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=random.Random(args.seed),
    )

    rng = random.Random(args.seed)
    sampled_indices = rng.sample(range(len(lines)), k=min(args.num_questions, len(lines)))
    run = wandb.init(
        project="Lambda-RLM-Product-Lookup",
        config={
            "backend": args.backend,
            "model_name": args.model_name,
            "dataset_path": args.dataset_path,
            "seed": args.seed,
            "context_size": args.context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "num_questions_requested": args.num_questions,
            "num_questions": len(sampled_indices),
            "lambda_rlm_init_kwargs": {
                "backend": args.backend,
                "backend_kwargs": {"model_name": args.model_name},
                "verbose": args.verbose or LAMBDA_RLM_INIT_KWARGS["verbose"],
            },
        },
    )

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, target_index in enumerate(sampled_indices):
        print(f"Question {i + 1}/{len(sampled_indices)}")
        target_line = lines[target_index]
        target_product = extract_product(target_line)
        question = build_question(target_product)
        ground_truth_index_set = product_index_map.get(target_product, set())
        sample_context = context_pipeline.build_context(
            context_size=args.context_size,
            correct_indices=ground_truth_index_set,
            query=target_product,
        )
        context_lines = [line for line in sample_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in context_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        gt_in_context_count = len(ground_truth_index_set & context_indices)
        print(
            f"[CONTEXT] requested_size={args.context_size} actual_size={len(context_lines)} "
            f"ground_truth_in_context={gt_in_context_count}/{len(ground_truth_index_set)}"
        )
        completion_prompt = build_prompt(context=sample_context, question=question)

        completion = lambda_rlm.completion(completion_prompt)
        response = completion.response
        parsed = parse_indices(response)

        usage = completion.usage_summary
        sample_cost_usd = usage.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        predicted_index_set = set(parsed)
        precision, recall, f1 = precision_recall_f1(predicted_index_set, ground_truth_index_set)
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
            print(f"Ground truth indices: {sorted(ground_truth_index_set)}")
            print("--------------------------------")

        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/target_index": target_index,
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_size": args.context_size,
                f"sample/{i}/total_calls": usage.model_usage_summaries[
                    completion.root_model
                ].total_calls
                if completion.root_model in usage.model_usage_summaries
                else 0,
                f"sample/{i}/total_input_tokens": usage.total_input_tokens,
                f"sample/{i}/total_output_tokens": usage.total_output_tokens,
                f"sample/{i}/execution_time_s": completion.execution_time,
                **(
                    {f"sample/{i}/total_cost_usd": sample_cost_usd}
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

    total = len(sampled_indices)
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
