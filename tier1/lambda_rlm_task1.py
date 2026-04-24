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


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "x-ai/grok-4-fast"
SEED = 42
NUM_QUESTIONS = 1

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
Each line in the context contains one chemical reaction in SMILES format:
- "index reactants>reagents>products"
- "index reactants>>products"

Find the index or indices whose PRODUCT exactly matches:
{product}

Rules:
- Match only against the product side, not reactants or reagents.
- Return indices as comma-separated integers only.
- Do not include any extra text, quotes, punctuation, or formatting.
- If no product matches, return -1.
""".strip()


def build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 1 with LambdaRLM.")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH)
    parser.add_argument("--backend", type=str, default=BACKEND)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--num-questions", type=int, default=NUM_QUESTIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


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

    rng = random.Random(args.seed)
    sampled_indices = rng.sample(range(len(lines)), k=min(args.num_questions, len(lines)))
    context = "\n".join(lines)

    run = wandb.init(
        project="Lambda-RLM-Product-Lookup",
        config={
            "backend": args.backend,
            "model_name": args.model_name,
            "dataset_path": args.dataset_path,
            "seed": args.seed,
            "num_questions_requested": args.num_questions,
            "num_questions": len(sampled_indices),
            "lambda_rlm_init_kwargs": {
                "backend": args.backend,
                "backend_kwargs": {"model_name": args.model_name},
                "verbose": args.verbose or LAMBDA_RLM_INIT_KWARGS["verbose"],
            },
        },
    )

    correct = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, target_index in enumerate(sampled_indices):
        print(f"Question {i + 1}/{len(sampled_indices)}")
        target_line = lines[target_index]
        target_product = extract_product(target_line)
        question = build_question(target_product)
        completion_prompt = build_prompt(context=context, question=question)

        completion = lambda_rlm.completion(completion_prompt)
        response = completion.response
        parsed = parse_indices(response)

        usage = completion.usage_summary
        sample_cost_usd = usage.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        is_correct = target_index in parsed
        if is_correct:
            correct += 1
        else:
            print(f"Error: {target_index} not in {parsed}")
            print(f"Line in context: {target_line}")
            print(f"Product: {target_product}")
            print(f"Response: {response}")
            print("--------------------------------")

        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/is_correct": int(is_correct),
                f"sample/{i}/target_index": target_index,
                f"sample/{i}/target_product": target_product,
                f"sample/{i}/response_raw": response,
                f"sample/{i}/response_parsed": ",".join(str(x) for x in parsed),
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
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
        wandb.log({"running_accuracy": correct / (i + 1)})

    total = len(sampled_indices)
    accuracy = (correct / total) if total else 0.0
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f}")

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
