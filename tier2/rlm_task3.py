import argparse
import os
import random
import uuid

import wandb
from rlm import RLM
from rlm.codeact_helpers import build_context_pipeline, parse_indices, precision_recall_f1
from rlm.tracing import init_tracing, using_tracing_attributes
from task3_hardcoded_ground_truth import (
    TASK3_HARDCODED_GROUND_TRUTH_INDICES,
    TASK3_THRESHOLDS,
)

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
THRESHOLDS = TASK3_THRESHOLDS
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}

def build_question(threshold: int) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Find all the indices of the reactions that satisfy:
      max over all pairs [rings(product_component) - rings(reactant_component)] == {threshold}

    Guidance:
    - You may use RDKit functions if needed (for example, Chem.MolFromSmiles and rdMolDescriptors.CalcNumRings).
    - Split each side by dot (.) to get components and compute ring count for each valid component.
    - For each reaction, compute ALL pairwise deltas:
      rings(product_component) - rings(reactant_component)
      and use the maximum of those deltas.
    - Ignore reagents (middle field).
    - For each side (reactants/products), ignore invalid or empty dot-separated molecules.
    - Skip a reaction only if reactant side or product side has no valid molecules left after filtering.

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If no matching reaction exists, report: -1
    """


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task3",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLM task 3 evaluation.")
    parser.add_argument(
        "--context-size",
        type=int,
        default=CONTEXT_SIZE,
        help=(
            "Number of retrieved reactions to include in context "
            f"(default: {CONTEXT_SIZE}; use -1 for all lines)."
        ),
    )
    return parser.parse_args()


def main(context_size: int) -> None:
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
    )

    ground_truth_indices_by_threshold: dict[int, set[int]] = {}
    for x in THRESHOLDS:
        indices = set(TASK3_HARDCODED_GROUND_TRUTH_INDICES.get(x, []))
        ground_truth_indices_by_threshold[x] = indices

    questions = [build_question(x) for x in THRESHOLDS]

    run = wandb.init(
        project="RLMs-Task3",
        config={
            "MODEL_NAME": MODEL_NAME,
            "thresholds": THRESHOLDS,
            "backend": BACKEND,
            "model_name": MODEL_NAME,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "num_questions": len(questions),
            "rlm_init_kwargs": RLM_INIT_KWARGS,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    total_input_tokens_sum = 0
    total_output_tokens_sum = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, threshold in enumerate(THRESHOLDS):
        question = questions[i]
        print(f"Question {i + 1}/{len(questions)} for X={threshold}")
        ground_truth_index_set = ground_truth_indices_by_threshold[threshold]
        sample_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=ground_truth_index_set,
            query=str(threshold),
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
            f"[CONTEXT] requested_size={context_size} actual_size={len(context_lines)} "
            f"ground_truth_in_context={gt_in_context_count}/{len(ground_truth_index_set)}"
        )
        completion_kwargs = {"prompt": sample_context, "root_prompt": question}
        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(questions),
                "threshold": threshold,
            },
            tags=["run_rlms", "sample", "delta_rings"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response
        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed_indices = parse_indices(response)
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        predicted_index_set = set(parsed_indices)
        precision, recall, f1 = precision_recall_f1(
            predicted_index_set, ground_truth_in_context_set
        )
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        if f1 < 1.0:
            print(f"Mismatch for X={threshold}")
            print(f"Predicted indices: {sorted(predicted_index_set)}")
            print(f"Ground truth indices (in context): {sorted(ground_truth_in_context_set)}")
            print(
                f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
            )
            print("--------------------------------")
        else:
            print(f"F1 is 1.0 for X={threshold}")

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
            total_input_tokens_sum += int(last_metric["total_input_tokens"])
            total_output_tokens_sum += int(last_metric["total_output_tokens"])
            wandb.log(
                {
                    "sample_idx": i,
                    f"sample/{i}/threshold_x": threshold,
                    f"sample/{i}/final_total_input_tokens": last_metric["total_input_tokens"],
                    f"sample/{i}/final_total_output_tokens": last_metric["total_output_tokens"],
                    f"sample/{i}/final_total_tokens": last_metric["total_tokens"],
                    f"sample/{i}/iterations": len(iteration_metrics),
                    f"sample/{i}/precision": precision,
                    f"sample/{i}/recall": recall,
                    f"sample/{i}/f1": f1,
                    f"sample/{i}/ground_truth_count": len(ground_truth_in_context_set),
                    f"sample/{i}/predicted_count": len(predicted_index_set),
                    f"sample/{i}/completion_prompt_char_count": len(sample_context),
                    f"sample/{i}/context_size": context_size,
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
    avg_total_input_tokens = (total_input_tokens_sum / total) if total else 0.0
    avg_total_output_tokens = (total_output_tokens_sum / total) if total else 0.0
    print(f"Macro Precision: {avg_precision:.4f}")
    print(f"Macro Recall: {avg_recall:.4f}")
    print(f"Macro F1: {avg_f1:.4f}")
    print(f"Avg total input tokens/sample: {avg_total_input_tokens:.2f}")
    print(f"Avg total output tokens/sample: {avg_total_output_tokens:.2f}")

    for x in THRESHOLDS:
        run.summary[f"ground_truth/x_{x}/count"] = len(ground_truth_indices_by_threshold[x])

    run.summary["total"] = total
    run.summary["macro_precision"] = avg_precision
    run.summary["macro_recall"] = avg_recall
    run.summary["macro_f1"] = avg_f1
    run.summary["avg_total_input_tokens_per_sample"] = avg_total_input_tokens
    run.summary["avg_total_output_tokens_per_sample"] = avg_total_output_tokens
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(context_size=args.context_size)
