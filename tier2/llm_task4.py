import asyncio
import os
import random
import uuid
from typing import Optional

import wandb
from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from rlm.codeact_helpers import (
    build_retriever,
    extract_response_text,
    extract_usage_metrics,
    load_lines,
    parse_count,
    parse_reaction_sides,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
THRESHOLDS = [1, 2, 3, 4, 5]
SEED = 42
CONTEXT_SIZE = 100
RETRIEVER_NAME = "random"
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


def component_aromatic_ring_counts(side_smiles: str) -> list[int]:
    if not side_smiles:
        return []

    aromatic_ring_counts: list[int] = []
    for comp in side_smiles.split("."):
        comp = comp.strip()
        if not comp:
            continue
        mol = Chem.MolFromSmiles(comp)
        if mol is None:
            continue
        aromatic_ring_counts.append(int(rdMolDescriptors.CalcNumAromaticRings(mol)))
    return aromatic_ring_counts


def reaction_delta_aromatic_rings(indexed_line: str) -> Optional[int]:
    reactants, products = parse_reaction_sides(indexed_line)
    reactant_counts = component_aromatic_ring_counts(reactants)
    product_counts = component_aromatic_ring_counts(products)
    if not reactant_counts or not product_counts:
        return None
    aromatic_rings_reactants = sum(reactant_counts)
    aromatic_rings_products = sum(product_counts)
    return aromatic_rings_products - aromatic_rings_reactants


def count_matches_exact(lines: list[str], x: int) -> int:
    total = 0
    for line in lines:
        delta = reaction_delta_aromatic_rings(line)
        if delta is not None and delta == x:
            total += 1
    return total


def build_question(threshold: int) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Count how many reactions satisfy:
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
    - Final response must be exactly: ANSWER: <integer>
      Example: ANSWER: 57
    - Do not include additional prose in the final response.
    - If no matching reaction exists, return: ANSWER: 0
"""


async def main() -> None:
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    rng = random.Random(SEED)
    retriever = build_retriever(name=RETRIEVER_NAME, lines=lines, rng=rng)
    run_session_id = f"llm-task4-{uuid.uuid4()}"

    llm = OpenRouter(
        model=MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
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
            "retriever_name": RETRIEVER_NAME,
            "mode": "llm_baseline_no_tools",
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    correct = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, threshold in enumerate(THRESHOLDS):
        print(f"Question {i + 1}/{len(THRESHOLDS)} for X={threshold}")
        question = build_question(threshold)
        retrieved_context = retriever.build_context(
            query=f"new_aromatic_rings_eq_{threshold}",
            target_index=-1,
            k=CONTEXT_SIZE,
        )
        retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
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
        parsed_count = parse_count(response_text)
        ground_truth_count = count_matches_exact(retrieved_lines, int(threshold))
        is_correct = parsed_count == ground_truth_count
        if is_correct:
            correct += 1

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
                f"sample/{i}/is_correct": int(is_correct),
                f"sample/{i}/ground_truth_count": ground_truth_count,
                f"sample/{i}/prediction_count": parsed_count if parsed_count is not None else -1,
                f"sample/{i}/response_raw": response_text,
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/context_size": CONTEXT_SIZE,
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_coverage": context_coverage,
                **({f"sample/{i}/final_total_cost_usd": sample_cost} if sample_cost is not None else {}),
            }
        )
        wandb.log(
            {
                "running_accuracy": correct / (i + 1),
                "running_context_coverage": context_coverage,
            }
        )

    total = len(THRESHOLDS)
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
    asyncio.run(main())
