import os
import random
import uuid

import wandb
from rlm import RLM
from rlm.tracing import init_tracing, using_tracing_attributes

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/workspace/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "anthropic/claude-sonnet-4.6"
SEED = 42
NUM_QUESTIONS = 2
ENABLE_TRACING = True

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 1,
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

    rng = random.Random(SEED)
    sampled_indices = rng.sample(range(len(lines)), k=min(NUM_QUESTIONS, len(lines)))
    questions: list[str] = []
    for index in sampled_indices:
        product = extract_product(lines[index])
        questions.append(
            f"""
            Context is a big string of chemical reactions in SMILES format, separated by newlines. Each reaction is a string of the form "index reactants>reagents>products" or "index reactants>>products".
            Find the index/indices of the reaction for the following PRODUCT (and not the reactants/reagents): {product}.
            Report the INDICES separated by commas. DO NOT INCLUDE any other text in your response including quotes, punctuation, or formatting.
            If the product is not found, report -1.
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
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    correct = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, question in enumerate(questions):
        print(f"Question {i + 1}/{len(questions)}")
        completion_kwargs = {"prompt": context, "root_prompt": question}
        # Group all sample traces under one run session while keeping each sample as a distinct trace.
        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(questions),
                "target_index": sampled_indices[i],
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
                    f"sample/{i}/is_correct": int(is_correct),
                    f"sample/{i}/target_index": target_index,
                    f"sample/{i}/target_product": target_product,
                    f"sample/{i}/response_raw": response,
                    f"sample/{i}/response_parsed": ",".join(str(x) for x in parsed),
                    f"sample/{i}/completion_root_prompt": question,
                    f"sample/{i}/completion_prompt_char_count": len(context),
                    **(
                        {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                        if sample_cost_usd is not None
                        else {}
                    ),
                }
            )

        wandb.log(
            {
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
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    main()
