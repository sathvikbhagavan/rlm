import asyncio
import os
import random
import uuid

import wandb
from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter
from task4_hardcoded_ground_truth import (
    TASK4_HARDCODED_GROUND_TRUTH_INDICES,
    TASK4_THRESHOLDS,
)

from rlm.codeact_helpers import (
    build_context_pipeline,
    extract_response_text,
    extract_usage_metrics,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
THRESHOLDS = TASK4_THRESHOLDS
SEED = 42
CONTEXT_SIZE = 500
CONTEXT_PIPELINE_NAME = "random"
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 30_000
# os.environ["WANDB_MODE"] = "disabled"


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="LLM-Task4",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def build_question(threshold: int) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Find all the indices of the reactions that satisfy:
      new_aromatic_rings == {threshold}

    Guidance:
    - You may use RDKit functions if needed (for example, Chem.MolFromSmiles and rdMolDescriptors.CalcNumAromaticRings).
    - Split each side by dot (.) to get components and compute aromatic ring count for each valid component.
    - Compute aromatic_rings_reactants as the sum over valid reactant components.
    - Compute aromatic_rings_products as the sum over valid product components.
    - Then compute:
      new_aromatic_rings = aromatic_rings_products - aromatic_rings_reactants
    - Ignore reagents (middle field).
    - For each side (reactants/products), ignore invalid or empty dot-separated molecules.
    - Skip a reaction only if reactant side or product side has no valid molecules left after filtering.

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If no matching reaction exists, report: -1
"""


async def main() -> None:
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    rng = random.Random(SEED)
    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=rng,
    )
    run_session_id = f"llm-task4-{uuid.uuid4()}"

    llm = OpenRouter(
        model=MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
    )

    run = wandb.init(
        project="LLM-Task4",
        config={
            "MODEL_NAME": MODEL_NAME,
            "thresholds": THRESHOLDS,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": CONTEXT_SIZE,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "reasoning_effort": REASONING_EFFORT,
            "mode": "llm_baseline_no_tools",
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    ground_truth_indices_by_threshold: dict[int, set[int]] = {
        threshold: set(TASK4_HARDCODED_GROUND_TRUTH_INDICES.get(threshold, []))
        for threshold in THRESHOLDS
    }

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    retrieval_hits = 0
    total_input_tokens_sum = 0
    total_output_tokens_sum = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, threshold in enumerate(THRESHOLDS):
        print(f"Question {i + 1}/{len(THRESHOLDS)} for X={threshold}")
        question = build_question(threshold)
        ground_truth_index_set = ground_truth_indices_by_threshold[threshold]
        retrieved_context = context_pipeline.build_context(
            context_size=CONTEXT_SIZE,
            correct_indices=ground_truth_index_set,
            query=str(threshold),
        )
        retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
        retrieved_indices = {
            int(line.split(" ", 1)[0])
            for line in retrieved_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        ground_truth_in_context_set = ground_truth_index_set & retrieved_indices
        gt_in_context_count = len(ground_truth_in_context_set)
        print(
            f"[CONTEXT] requested_size={CONTEXT_SIZE} actual_size={len(retrieved_lines)} "
            f"ground_truth_in_context={gt_in_context_count}/{len(ground_truth_index_set)}"
        )
        context_has_ground_truth = bool(ground_truth_in_context_set)
        retrieval_hits += int(context_has_ground_truth)
        context_coverage = len(retrieved_lines) / len(lines) if lines else 0.0
        completion_prompt = f"""
        You are given a subset of chemical reactions in SMILES format and a question.
        <context>
        {retrieved_context}
        </context>
        <question>
        {question}
        </question>
        """

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(THRESHOLDS),
                "threshold": threshold,
                "agent": "llm_baseline",
            },
            tags=["llm-baseline", "sample", "new_aromatic_rings"],
        ):
            response = await llm.achat([ChatMessage(role="user", content=completion_prompt)])

        response_text = extract_response_text(response)
        predicted_index_set = set(parse_indices(response_text))
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

        usage_metrics = extract_usage_metrics(response)
        prompt_tokens = int(usage_metrics.get("prompt_tokens", 0))
        completion_tokens = int(usage_metrics.get("completion_tokens", 0))
        total_tokens = int(usage_metrics.get("total_tokens", 0))
        sample_cost = (
            float(usage_metrics["cost_usd"]) if "cost_usd" in usage_metrics else None
        )
        if total_tokens == 0:
            prompt_tokens = count_tokens([{"role": "user", "content": completion_prompt}], MODEL_NAME)
            completion_tokens = count_tokens(
                [{"role": "assistant", "content": response_text}],
                MODEL_NAME,
            )
            total_tokens = prompt_tokens + completion_tokens
        if sample_cost is not None:
            total_cost_usd += sample_cost
            samples_with_cost += 1
        total_input_tokens_sum += prompt_tokens
        total_output_tokens_sum += completion_tokens

        wandb.log(
            {
                "sample_iteration": 1,
                f"sample/{i}/iteration_input_tokens": prompt_tokens,
                f"sample/{i}/iteration_output_tokens": completion_tokens,
                f"sample/{i}/iteration_total_tokens": total_tokens,
                **({f"sample/{i}/iteration_cost_usd": sample_cost} if sample_cost is not None else {}),
            }
        )
        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/threshold_x": threshold,
                f"sample/{i}/final_total_input_tokens": prompt_tokens,
                f"sample/{i}/final_total_output_tokens": completion_tokens,
                f"sample/{i}/final_total_tokens": total_tokens,
                f"sample/{i}/iterations": 1,
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/ground_truth_count": len(ground_truth_in_context_set),
                f"sample/{i}/predicted_count": len(predicted_index_set),
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/context_size": CONTEXT_SIZE,
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_coverage": context_coverage,
                f"sample/{i}/context_has_ground_truth": int(context_has_ground_truth),
                **({f"sample/{i}/final_total_cost_usd": sample_cost} if sample_cost is not None else {}),
            }
        )
        wandb.log(
            {
                "running_precision": precision_sum / (i + 1),
                "running_recall": recall_sum / (i + 1),
                "running_f1": f1_sum / (i + 1),
                "running_retrieval_hit_rate": retrieval_hits / (i + 1),
            }
        )

    total = len(THRESHOLDS)
    avg_precision = (precision_sum / total) if total else 0.0
    avg_recall = (recall_sum / total) if total else 0.0
    avg_f1 = (f1_sum / total) if total else 0.0
    retrieval_hit_rate = (retrieval_hits / total) if total else 0.0
    avg_total_input_tokens = (total_input_tokens_sum / total) if total else 0.0
    avg_total_output_tokens = (total_output_tokens_sum / total) if total else 0.0
    print(f"Macro Precision: {avg_precision:.4f}")
    print(f"Macro Recall: {avg_recall:.4f}")
    print(f"Macro F1: {avg_f1:.4f}")
    print(f"Retrieval hit-rate (ground truth in context): {retrieval_hit_rate:.4f}")
    print(f"Avg total input tokens/sample: {avg_total_input_tokens:.2f}")
    print(f"Avg total output tokens/sample: {avg_total_output_tokens:.2f}")

    run.summary["total"] = total
    run.summary["macro_precision"] = avg_precision
    run.summary["macro_recall"] = avg_recall
    run.summary["macro_f1"] = avg_f1
    run.summary["retrieval_hits"] = retrieval_hits
    run.summary["retrieval_hit_rate"] = retrieval_hit_rate
    run.summary["avg_total_input_tokens_per_sample"] = avg_total_input_tokens
    run.summary["avg_total_output_tokens_per_sample"] = avg_total_output_tokens
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
