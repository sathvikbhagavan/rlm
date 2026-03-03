import os
import random

import wandb
from rlm import RLM


DATASET_PATH = "/workspace/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5.2"
SEED = 42
NUM_QUESTIONS = 200

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
}


def parse_indices(response: str) -> list[int]:
    if response.isdigit():
        return [int(response)]
    return [int(num) for num in response.split(",")]


def extract_product(indexed_line: str) -> str:
    _, reaction_smiles = indexed_line.split(" ", 1)
    return reaction_smiles.split(">")[-1].strip()


def main() -> None:
    rlm = RLM(**RLM_INIT_KWARGS)

    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]

    rng = random.Random(SEED)
    sampled_indices = rng.sample(range(len(lines)), k=min(NUM_QUESTIONS, len(lines)))
    questions: list[str] = []
    for index in sampled_indices:
        product = extract_product(lines[index])
        questions.append(
            f"""
            Context is a big string of chemical equations in SMILES format, separated by newlines.
            Find the index/indices (number at the start) of the equation for the following PRODUCT (and not the reactants/reagents): {product}.
            Report the INDICES separated by commas. DO NOT INCLUDE any other text in your response including quotes, punctuation, or formatting.
            If the product is not found, report an empty string.
            """
        )

    context = "\n".join(lines)

    run = wandb.init(
        project="RLMs-Product-Lookup",
        config={
            "MODEL_NAME": MODEL_NAME,
            "SEED": SEED,
            "NUM_QUESTIONS": NUM_QUESTIONS,
            "backend": BACKEND,
            "model_name": MODEL_NAME,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "num_questions_requested": NUM_QUESTIONS,
            "num_questions": len(questions),
            "rlm_init_kwargs": RLM_INIT_KWARGS,
        },
    )

    correct = 0
    results: list[list[int]] = []

    for i, question in enumerate(questions):
        print(f"Question {i + 1}/{len(questions)}")
        completion_kwargs = {"prompt": context, "root_prompt": question}
        response = rlm.completion(**completion_kwargs).response
        parsed = parse_indices(response)
        results.append(parsed)

        target_index = sampled_indices[i]
        target_line = lines[target_index]
        target_product = extract_product(target_line)
        is_correct = target_index in parsed
        if is_correct:
            correct += 1

        if not is_correct:
            print(f"Error: {target_index} not in {parsed}")
            print(f"Line in context: {target_line}")
            print(f"Product: {target_product}")
            print(f"Response: {response}")
            print("--------------------------------")

        wandb.log(
            {
                "question_idx": i,
                "target_index": target_index,
                "target_product": target_product,
                "response_raw": response,
                "response_parsed": ",".join(str(x) for x in parsed),
                "completion_root_prompt": question,
                "completion_prompt_char_count": len(context),
                "is_correct": int(is_correct),
                "running_accuracy": correct / (i + 1),
            }
        )

    total = len(questions)
    accuracy = (correct / total) if total else 0.0
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f}")

    run.summary["correct"] = correct
    run.summary["total"] = total
    run.summary["accuracy"] = accuracy
    wandb.finish()


if __name__ == "__main__":
    main()
