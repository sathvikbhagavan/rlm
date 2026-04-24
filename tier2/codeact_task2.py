from typing import Any, Optional
import argparse
import asyncio
import random
import os
import uuid

import wandb
from rdkit import Chem
from rdkit.Chem import Descriptors
from llama_index.core.workflow import Context
from llama_index.llms.openrouter import OpenRouter

from rlm.codeact_core import CodeActAgent, make_simple_code_executor, run_agent_verbose
from rlm.codeact_helpers import (
    build_retriever,
    extract_response_text,
    load_lines,
    parse_count,
    parse_reaction_sides,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
MODEL_NAME = "openai/gpt-5-mini"  # or try something simpler first
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 600.0
THRESHOLDS = list(range(100, 300, 10))
SEED = 42
CONTEXT_SIZE = 10
RETRIEVER_NAME = "random"
MAX_OUTPUT_TOKENS = 30_000
MAX_ITERATIONS = 8
# os.environ["WANDB_MODE"] = "disabled"


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="codeact-delta-weight-count",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def heaviest_component_weight(side_smiles: str) -> Optional[float]:
    if not side_smiles:
        return None

    weights: list[float] = []
    for comp in side_smiles.split("."):
        comp = comp.strip()
        if not comp:
            continue
        mol = Chem.MolFromSmiles(comp)
        if mol is None:
            continue
        weights.append(Descriptors.MolWt(mol))

    if not weights:
        return None
    return max(weights)


def reaction_delta_weight(indexed_line: str) -> Optional[float]:
    reactants, products = parse_reaction_sides(indexed_line)
    heaviest_reactant = heaviest_component_weight(reactants)
    heaviest_product = heaviest_component_weight(products)
    if heaviest_reactant is None or heaviest_product is None:
        return None
    return heaviest_product - heaviest_reactant


def count_matches(lines: list[str], threshold: float) -> int:
    total = 0
    for line in lines:
        delta = reaction_delta_weight(line)
        if delta is not None and delta > threshold:
            total += 1
    return total


def _extract_usage_metrics(response: Any) -> dict[str, float | int]:
    def _from_usage_obj(usage: Any) -> tuple[int, int, int, float | None]:
        if usage is None:
            return 0, 0, 0, None
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
            completion = int(
                usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
            )
            total = int(usage.get("total_tokens", 0) or (prompt + completion))
            cost = usage.get("cost")
            if cost is None:
                cost = (
                    usage.get("cost_details", {}).get("upstream_inference_cost")
                    if isinstance(usage.get("cost_details"), dict)
                    else None
                )
            return prompt, completion, total, (float(cost) if cost is not None else None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
        completion = int(
            getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
        )
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
        cost = getattr(usage, "cost", None)
        extra = getattr(usage, "model_extra", None)
        if cost is None and isinstance(extra, dict):
            cost = extra.get("cost")
            if cost is None and isinstance(extra.get("cost_details"), dict):
                cost = extra["cost_details"].get("upstream_inference_cost")
        return prompt, completion, total, (float(cost) if cost is not None else None)

    candidates: list[Any] = []
    raw = getattr(response, "raw", None)
    if raw is not None:
        candidates.append(raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None))
    candidates.append(getattr(response, "usage", None))
    msg = getattr(response, "message", None)
    if msg is not None:
        msg_kwargs = getattr(msg, "additional_kwargs", None)
        if isinstance(msg_kwargs, dict):
            candidates.append(msg_kwargs.get("usage"))
    add_kwargs = getattr(response, "additional_kwargs", None)
    if isinstance(add_kwargs, dict):
        candidates.append(add_kwargs.get("usage"))

    for usage in candidates:
        p, c, t, cost = _from_usage_obj(usage)
        if p or c or t or cost is not None:
            result: dict[str, float | int] = {
                "prompt_tokens": p,
                "completion_tokens": c,
                "total_tokens": t,
            }
            if cost is not None:
                result["cost_usd"] = cost
            return result

    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _extract_finish_reason(response: Any) -> Optional[str]:
    raw = getattr(response, "raw", None)
    if raw is None:
        return None

    choices = raw.get("choices") if isinstance(raw, dict) else getattr(raw, "choices", None)
    if not choices:
        return None

    first = choices[0]
    if isinstance(first, dict):
        reason = first.get("finish_reason")
    else:
        reason = getattr(first, "finish_reason", None)
    return str(reason) if reason is not None else None


def build_question(threshold: int) -> str:
    return f"""
    Above is a list of chemical reactions in SMILES format. Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Count how many reactions satisfy:
      weight(heaviest product) - weight(heaviest reactant) > {threshold} Da

    Guidance:
    - You may use RDKit functions if needed (for example, Chem.MolFromSmiles and Descriptors.MolWt).
    - "heaviest" means the largest molecular weight among dot-separated molecules on that side.
    - Ignore reagents (middle field).
    - For each side (reactants/products), ignore invalid or empty dot-separated molecules instead of treating them as 0.0.
    - Compute the side's heaviest weight from the remaining valid molecules only.
    - Skip a reaction only if reactant side or product side has no valid molecules left after filtering.
    - Do NOT assign 0.0 as a fallback for missing/invalid molecules.
    - If you copy SMILES text into a Python string literal, handle backslashes safely:
      use a raw string (for example, r\"\"\"...\"\"\") or escape backslashes as "\\\\".

    Output format:
    - Final response must be exactly: ANSWER: <integer>
      Example: ANSWER: 57
    - Do not include additional prose in the final response.
    - If no matching reaction exists, return: ANSWER: 0
"""


def build_code_executor(lines: list[str]):
    del lines
    return make_simple_code_executor(
        extra_globals={
            "np": __import__("numpy"),
            "rdkit": __import__("rdkit"),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAct task 2 evaluation.")
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
    lines = load_lines()
    rng = random.Random(SEED)
    retriever = build_retriever(name=RETRIEVER_NAME, lines=lines, rng=rng)
    run_session_id = f"codeact-task2-{uuid.uuid4()}"

    run = wandb.init(
        project="CodeAct-Task2",
        config={
            "MODEL_NAME": model_name,
            "thresholds": THRESHOLDS,
            "dataset_path": DATASET_PATH,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": SEED,
            "context_size": context_size,
            "retriever_name": RETRIEVER_NAME,
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
            query=f"delta_weight_gt_{threshold}",
            target_index=-1,
            k=context_size,
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
        executor = build_code_executor(lines=retrieved_lines)
        agent = CodeActAgent(
            code_execute_fn=executor.execute,
            llm=OpenRouter(
                model=model_name,
                api_key=OPENROUTER_API_KEY,
                max_tokens=MAX_OUTPUT_TOKENS,
                additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
            ),
            max_iterations=MAX_ITERATIONS,
            timeout=WORKFLOW_TIMEOUT_S,
        )
        ctx = Context(agent)

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(THRESHOLDS),
                "threshold": threshold,
                "agent": "codeact",
            },
            tags=["codeact", "sample", "delta_weight"],
        ):
            print(f"Prompt: {completion_prompt!r}")
            response = await run_agent_verbose(agent, ctx, completion_prompt)
            

        response_text = extract_response_text(response)
        print(f"Raw response: {response_text!r}")
        print("-" * 60)
        llm_turn_metrics = await ctx.store.get("llm_turn_metrics", default=[])
        if not llm_turn_metrics:
            estimated_prompt_tokens = count_tokens(
                [{"role": "user", "content": completion_prompt}],
                model_name,
            )
            estimated_completion_tokens = count_tokens(
                [{"role": "assistant", "content": response_text}],
                model_name,
            )
            llm_turn_metrics = [
                {
                    "iteration": 1,
                    "iteration_input_tokens": estimated_prompt_tokens,
                    "iteration_output_tokens": estimated_completion_tokens,
                    "iteration_total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
                }
            ]

        parsed_count = parse_count(response_text)
        ground_truth_count = count_matches(retrieved_lines, float(threshold))
        is_correct = parsed_count == ground_truth_count
        if is_correct:
            correct += 1
        else:
            print(f"Mismatch for X={threshold}")
            print(f"Predicted count: {parsed_count}")
            print(f"Ground truth count (retrieved context): {ground_truth_count}")
            print(f"Raw response: {response_text!r}")
            print("-" * 60)

        for metric in llm_turn_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                    f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                    f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                    **(
                        {f"sample/{i}/iteration_cost_usd": metric["iteration_cost_usd"]}
                        if "iteration_cost_usd" in metric
                        else {}
                    ),
                }
            )

        final_input_tokens = sum(
            int(metric.get("iteration_input_tokens", 0)) for metric in llm_turn_metrics
        )
        final_output_tokens = sum(
            int(metric.get("iteration_output_tokens", 0)) for metric in llm_turn_metrics
        )
        final_total_tokens = sum(
            int(metric.get("iteration_total_tokens", 0)) for metric in llm_turn_metrics
        )
        final_cost = sum(float(metric.get("iteration_cost_usd", 0.0)) for metric in llm_turn_metrics)
        has_cost = any("iteration_cost_usd" in metric for metric in llm_turn_metrics)
        if has_cost:
            total_cost_usd += final_cost
            samples_with_cost += 1

        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/threshold_x": threshold,
                f"sample/{i}/final_total_input_tokens": final_input_tokens,
                f"sample/{i}/final_total_output_tokens": final_output_tokens,
                f"sample/{i}/final_total_tokens": final_total_tokens,
                f"sample/{i}/iterations": len(llm_turn_metrics),
                f"sample/{i}/is_correct": int(is_correct),
                f"sample/{i}/ground_truth_count": ground_truth_count,
                f"sample/{i}/prediction_count": parsed_count if parsed_count is not None else -1,
                f"sample/{i}/response_raw": response_text,
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/context_size": context_size,
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_coverage": context_coverage,
                **({f"sample/{i}/final_total_cost_usd": final_cost} if has_cost else {}),
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
    args = parse_args()
    asyncio.run(main(model_name=args.model_name, context_size=args.context_size))
