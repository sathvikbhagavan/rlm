from typing import Optional
import argparse
import asyncio
import os
import random
import uuid

import wandb
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from llama_index.core.workflow import Context
from llama_index.llms.openrouter import OpenRouter

from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens

from codeact_task2 import (
    CodeActAgent,
    build_retriever,
    extract_response_text,
    load_lines,
    parse_count,
    run_agent_verbose,
)


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 600.0
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
        project_name="codeact-new-aromatic-rings-count",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_reaction_sides(indexed_line: str) -> tuple[str, str]:
    _, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    reactant_side = parts[0].strip()
    product_side = parts[-1].strip()
    return reactant_side, product_side


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
    new_aromatic_rings = aromatic_rings_products - aromatic_rings_reactants
    return new_aromatic_rings


def count_matches_exact(lines: list[str], x: int) -> int:
    total = 0
    for line in lines:
        delta = reaction_delta_aromatic_rings(line)
        if delta is not None and delta == x:
            total += 1
    return total


def build_question(threshold: int) -> str:
    return f"""
    Above is a list of chemical reactions in SMILES format. Each reaction is in one of these forms:
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
    - If you copy SMILES text into a Python string literal, handle backslashes safely:
      use a raw string (for example, r\"\"\"...\"\"\") or escape backslashes as "\\\\".

    Output format:
    - Final response must be exactly: ANSWER: <integer>
      Example: ANSWER: 57
    - Do not include additional prose in the final response.
    - If no matching reaction exists, return: ANSWER: 0
"""


def build_code_executor() -> object:
    namespace = {
        "__builtins__": __builtins__,
        "np": __import__("numpy"),
        "rdkit": __import__("rdkit"),
    }

    class _Executor:
        def __init__(self) -> None:
            self.namespace = dict(namespace)

        def execute(self, code: str) -> str:
            import ast
            import contextlib
            import io
            import traceback

            stdout = io.StringIO()
            stderr = io.StringIO()
            output = ""
            return_value = None
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    try:
                        tree = ast.parse(code)
                        last_node = tree.body[-1] if tree.body else None
                        if isinstance(last_node, ast.Expr):
                            last_line = code.rstrip().split("\n")[-1]
                            exec_code = code[: -len(last_line)] + "\n__result__ = " + last_line
                            exec(exec_code, self.namespace, self.namespace)
                            return_value = self.namespace.get("__result__")
                        else:
                            exec(code, self.namespace, self.namespace)
                    except Exception:
                        exec(code, self.namespace, self.namespace)
                output = stdout.getvalue()
                if stderr.getvalue():
                    output += "\n" + stderr.getvalue()
            except Exception as e:
                output = f"Error: {type(e).__name__}: {str(e)}\n"
                output += traceback.format_exc()

            if return_value is not None:
                output += "\n\n" + str(return_value)
            return output

    return _Executor()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAct task 4 evaluation.")
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
    run_session_id = f"codeact-task4-{uuid.uuid4()}"

    run = wandb.init(
        project="CodeAct-Task4",
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
            query=f"new_aromatic_rings_eq_{threshold}",
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
        executor = build_code_executor()
        agent = CodeActAgent(
            code_execute_fn=executor.execute,
            llm=OpenRouter(
                model=model_name,
                api_key=OPENROUTER_API_KEY,
                max_tokens=MAX_OUTPUT_TOKENS,
                additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
            ),
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
            tags=["codeact", "sample", "new_aromatic_rings"],
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
        ground_truth_count = count_matches_exact(retrieved_lines, int(threshold))
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
