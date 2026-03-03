import os
import random

import wandb
from openai import OpenAI


DATASET_PATH = "/workspace/datasets/reactionSmilesFigShareUSPTO2023.txt"
MODEL_NAME = "google/gemini-3-flash-preview"
SEED = 42
NUM_QUESTIONS = 200
CONTEXT_SIZE = 7500

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def parse_indices(response: str) -> list[int]:
    response = response.strip()
    if not response:
        return []
    if response.isdigit():
        return [int(response)]
    return [int(num.strip()) for num in response.split(",") if num.strip().isdigit()]


def extract_product(indexed_line: str) -> str:
    _, reaction_smiles = indexed_line.split(" ", 1)
    return reaction_smiles.split(">")[-1].strip()


def load_lines():
    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        return [f"{i} {line}" for i, line in enumerate(raw_lines)]


def build_context(lines: list[str], target_index: int, rng: random.Random, k: int) -> str:
    """
    Randomly sample k equations from the dataset, always including the one at
    target_index. Returns them shuffled (so the correct equation is not always
    first) and joined as a single newline-separated string.
    """
    other_indices = [i for i in range(len(lines)) if i != target_index]
    sampled = rng.sample(other_indices, k=min(k - 1, len(other_indices)))
    sampled.append(target_index)
    rng.shuffle(sampled)
    return "\n".join(lines[i] for i in sampled)


def main():
    if not OPENROUTER_API_KEY:
        raise ValueError("Set OPENROUTER_API_KEY in your environment before running.")

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    lines = load_lines()

    rng = random.Random(SEED)
    sampled_indices = rng.sample(range(len(lines)), k=min(NUM_QUESTIONS, len(lines)))

    questions = []
    for index in sampled_indices:
        product = extract_product(lines[index])
        questions.append(
            f"""
            Context is a big string of chemical equations in SMILES format, separated by newlines.
            Find the index/indices (number at the start) of the equation for the following PRODUCT (and not the reactants/reagents): {product}.
            Report the INDICES separated by commas. DO NOT INCLUDE any other text in your response including quotes, punctuation, or formatting.
            If the product is not found, report an empty string. 
            Beware of outputting too many internal thoughts because of limit on the number of tokens. Just respond with the indices.
            """
        )

    run = wandb.init(
        project="LLMs-RAG-Product-Lookup",
        config={
            "MODEL_NAME": MODEL_NAME,
            "SEED": SEED,
            "NUM_QUESTIONS": NUM_QUESTIONS,
            "CONTEXT_SIZE": CONTEXT_SIZE,
            "dataset_path": DATASET_PATH,
            "num_questions": len(sampled_indices),
        },
    )

    correct = 0
    results = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for i, question in enumerate(questions):
        print(f"Question {i + 1}/{len(sampled_indices)}")

        target_index = sampled_indices[i]
        target_line = lines[target_index]
        target_product = extract_product(target_line)

        retrieved_context = build_context(lines, target_index, rng, CONTEXT_SIZE)

        completion_prompt = f"""
        You are given a subset of chemical reactions in SMILES format and a question.
        <context>
        {retrieved_context}
        </context>
        <question>
        {question}
        </question>
        """

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": completion_prompt}],
        )

        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        response = choice.message.content or ""
        if not response:
            print(f"  [WARNING] Empty response. finish_reason={finish_reason!r}")
        parsed = parse_indices(response)
        results.append(parsed)

        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        tokens = usage.total_tokens if usage else 0

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += tokens

        is_correct = target_index in parsed
        if is_correct:
            correct += 1
        else:
            print(f"❌ Error: {target_index} not in {parsed}")
            print(f"Line: {target_line}")
            print(f"Product: {target_product}")
            print(f"Response: {response!r}")
            print(f"finish_reason: {finish_reason!r}")
            print("-" * 60)

        wandb.log(
            {
                "question_idx": i,
                "target_index": target_index,
                "target_product": target_product,
                "response_raw": response,
                "response_parsed": ",".join(str(x) for x in parsed),
                "context_char_count": len(retrieved_context),
                "context_size": CONTEXT_SIZE,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": tokens,
                "is_correct": int(is_correct),
                "running_accuracy": correct / (i + 1),
            }
        )

    total = len(results)
    accuracy = (correct / total) if total else 0.0

    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f}")

    run.summary["correct"] = correct
    run.summary["total"] = total
    run.summary["accuracy"] = accuracy
    run.summary["total_prompt_tokens"] = total_prompt_tokens
    run.summary["total_completion_tokens"] = total_completion_tokens
    run.summary["total_tokens"] = total_tokens

    wandb.finish()


if __name__ == "__main__":
    main()