import argparse
import asyncio
import os
import random
import uuid

from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter
from task12b_hub_molecule_graph import (
    build_protected_hub_context,
    canonicalize_smiles,
    ground_truth_hub_molecules_in_context,
)
from task12b_hub_molecule_ground_truth import (
    HARDCODED_GT_HUB_MOLECULES,
    TASK12B_DAG_MODE,
    TASK12B_GROUND_TRUTH_DEFINITION,
    TASK12B_MIN_DOWNSTREAM,
    TASK12B_MIN_SELECTED_GROUND_TRUTH,
    TASK12B_TOTAL_REACTIONS,
    hubs_for_context_sampling,
    support_indices,
)

import wandb
from rlm.codeact_helpers import extract_response_text, extract_usage_metrics, load_lines, precision_recall_f1
from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
SEED = 42
CONTEXT_SIZE = 100
MAX_OUTPUT_TOKENS = 30_000
REASONING_EFFORT = "high"


def parse_smiles(response: str) -> list[str]:
    response = response.strip()
    if not response:
        return []
    if response.replace(" ", "") == "-1":
        return []

    parsed: list[str] = []
    seen: set[str] = set()
    for line in response.splitlines():
        line = line.strip()
        if not line or line == "-1":
            continue
        if line.upper().startswith("ANSWER:"):
            line = line.split(":", 1)[1].strip()
            if not line or line == "-1":
                continue
        canonical = canonicalize_smiles(line)
        if canonical is None:
            continue
        if canonical not in seen:
            seen.add(canonical)
            parsed.append(canonical)
    return sorted(parsed)


def build_question(min_downstream: int = TASK12B_MIN_DOWNSTREAM) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Each side (reactants / products) may contain multiple molecules separated by dots (.).
    Ignore reagents (middle field between the two > delimiters).

    Task:
    Find ALL molecules in the provided context that satisfy both conditions:
    1. The molecule appears as a product in exactly ONE reaction in the context.
    2. The same molecule appears as a reactant in at least {min_downstream} distinct downstream
       consumer reactions in the context.

    Downstream consumer reactions must have a strictly higher reaction index than the single
    producing reaction (index_asc DAG rule).

    Molecule identity must be determined by exact SMILES string equality on each component
    after splitting on dots (.).
    Do NOT use substructure matching — only exact equality counts as a match.
    Only consider reactions that appear in the provided context string.
    Do not infer producers or consumers from reactions outside the context.

    Guidance:
    - Split multi-component sides on dots (.).
    - Skip malformed reactions.

    Output format:
    - Return one SMILES string per line.
    - Sort lines in lexicographic (ascending) order.
    - No other text, quotes, labels, punctuation, or formatting.
    - If no molecule satisfies the criteria, return -1.
    """


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="LLM-Task12b",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM task 12b — hub molecule identification evaluation."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for OpenRouter (default: {MODEL_NAME}).",
    )
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


async def main(model_name: str, context_size: int) -> None:
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    run_session_id = f"llm-task12b-{uuid.uuid4()}"
    full_gt = list(HARDCODED_GT_HUB_MOLECULES)
    full_support = support_indices()
    print(
        f"Ground truth hub molecules={len(full_gt)} "
        f"full_support_indices={len(full_support)}"
    )

    llm = OpenRouter(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
    )

    run = wandb.init(
        project="LLM-Task12b",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": "protected_hub_ceiling",
            "min_selected_ground_truth": TASK12B_MIN_SELECTED_GROUND_TRUTH,
            "reasoning_effort": REASONING_EFFORT,
            "num_questions": 1,
            "dag_mode": TASK12B_DAG_MODE,
            "min_downstream": TASK12B_MIN_DOWNSTREAM,
            "task_description": "Identify hub molecules with one producer and many downstream consumers.",
            "ground_truth_definition": TASK12B_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK12B_TOTAL_REACTIONS,
            "mode": "llm_baseline_no_tools",
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    question = build_question()
    sampling = hubs_for_context_sampling(context_size)
    support = set(sampling.support_indices)
    retrieved_context = build_protected_hub_context(
        lines,
        support,
        context_size=context_size,
        rng=random.Random(SEED),
    )
    retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
    context_indices = {
        int(line.split(" ", 1)[0])
        for line in retrieved_lines
        if " " in line and line.split(" ", 1)[0].isdigit()
    }
    gt_molecules = ground_truth_hub_molecules_in_context(
        retrieved_lines,
        min_downstream=TASK12B_MIN_DOWNSTREAM,
        dag_mode=TASK12B_DAG_MODE,
    )
    gt_set = set(gt_molecules)
    context_coverage = len(retrieved_lines) / len(lines) if lines else 0.0
    support_in_context = len(support & context_indices)

    completion_prompt = f"""
    You are given a subset of chemical reactions in SMILES format and a question.
    <context>
    {retrieved_context}
    </context>
    <question>
    {question}
    </question>
    """

    print("\nQuestion 1/1: hub_molecule_identification")
    print(
        f"[CONTEXT] requested_size={context_size} actual_size={len(retrieved_lines)} "
        f"selected_hubs={sampling.selected_hub_count}/{len(full_gt)} "
        f"forced_count={sampling.forced_count} "
        f"support_in_context={support_in_context}/{len(support)} "
        f"full_support={len(full_support)} "
        f"gt_in_context={len(gt_molecules)}/{sampling.selected_hub_count} "
        f"coverage={context_coverage:.4f}"
    )

    with using_tracing_attributes(
        session_id=run_session_id,
        metadata={
            "sample_index": 0,
            "sample_count": 1,
            "task": "hub_molecule_identification",
            "gt_molecule_count": len(gt_molecules),
            "agent": "llm_baseline",
            "ground_truth_definition": TASK12B_GROUND_TRUTH_DEFINITION,
        },
        tags=["llm-baseline", "sample", "task12b_HUB_MOLECULE"],
    ):
        response = await llm.achat([ChatMessage(role="user", content=completion_prompt)])

    response_text = extract_response_text(response)
    parsed_molecules = parse_smiles(response_text)
    pred_set = set(parsed_molecules)
    precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
    is_exact_match = pred_set == gt_set

    print(f"Response: {response_text[:500]}{'…' if len(response_text) > 500 else ''}")
    print(f"Predicted molecules: {len(parsed_molecules)}")
    print(f"Ground truth molecules: {len(gt_molecules)}")
    print(
        f"Metrics -> precision={precision:.4f} recall={recall:.4f} "
        f"f1={f1:.4f} exact_match={is_exact_match}"
    )

    usage_metrics = extract_usage_metrics(response)
    prompt_tokens = int(usage_metrics.get("prompt_tokens", 0))
    completion_tokens = int(usage_metrics.get("completion_tokens", 0))
    total_tokens = int(usage_metrics.get("total_tokens", 0))
    sample_cost = float(usage_metrics["cost_usd"]) if "cost_usd" in usage_metrics else None
    if total_tokens == 0:
        prompt_tokens = count_tokens(
            [{"role": "user", "content": completion_prompt}],
            model_name,
        )
        completion_tokens = count_tokens(
            [{"role": "assistant", "content": response_text}],
            model_name,
        )
        total_tokens = prompt_tokens + completion_tokens

    wandb.log(
        {
            "sample_iteration": 1,
            "sample/0/iteration_input_tokens": prompt_tokens,
            "sample/0/iteration_output_tokens": completion_tokens,
            "sample/0/iteration_total_tokens": total_tokens,
            **({"sample/0/iteration_cost_usd": sample_cost} if sample_cost is not None else {}),
        }
    )
    wandb.log(
        {
            "sample_idx": 0,
            "sample/0/ground_truth_molecule_count": len(gt_molecules),
            "sample/0/ground_truth_full_count": len(full_gt),
            "sample/0/selected_hub_count": sampling.selected_hub_count,
            "sample/0/forced_reaction_count": sampling.forced_count,
            "sample/0/predicted_molecule_count": len(parsed_molecules),
            "sample/0/support_indices_in_context": support_in_context,
            "sample/0/support_indices_full_count": len(full_support),
            "sample/0/support_indices_selected_count": len(support),
            "sample/0/final_total_input_tokens": prompt_tokens,
            "sample/0/final_total_output_tokens": completion_tokens,
            "sample/0/final_total_tokens": total_tokens,
            "sample/0/iterations": 1,
            "sample/0/response_raw": response_text,
            "sample/0/response_parsed_molecules": "\n".join(parsed_molecules),
            "sample/0/ground_truth_molecules": "\n".join(gt_molecules),
            "sample/0/precision": precision,
            "sample/0/recall": recall,
            "sample/0/f1": f1,
            "sample/0/is_exact_match": int(is_exact_match),
            "sample/0/completion_prompt_char_count": len(completion_prompt),
            "sample/0/context_char_count": len(retrieved_context),
            "sample/0/retrieved_line_count": len(retrieved_lines),
            "sample/0/context_size": context_size,
            "sample/0/context_coverage": context_coverage,
            **({"sample/0/final_total_cost_usd": sample_cost} if sample_cost is not None else {}),
        }
    )

    run.summary["ground_truth/molecule_count"] = len(full_gt)
    run.summary["ground_truth/molecules"] = "\n".join(full_gt)
    run.summary["exact_match_correct"] = int(is_exact_match)
    run.summary["total"] = 1
    run.summary["exact_match_accuracy"] = float(is_exact_match)
    run.summary["macro_precision"] = precision
    run.summary["macro_recall"] = recall
    run.summary["macro_f1"] = f1
    run.summary["avg_total_input_tokens_per_sample"] = float(prompt_tokens)
    run.summary["avg_total_output_tokens_per_sample"] = float(completion_tokens)
    if sample_cost is not None:
        run.summary["total_cost_usd"] = sample_cost
        run.summary["avg_cost_per_sample_usd"] = sample_cost
        run.summary["samples_with_cost"] = 1
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            model_name=args.model_name,
            context_size=args.context_size,
        )
    )
