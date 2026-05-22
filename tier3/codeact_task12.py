import argparse
import asyncio
import os
import random
import uuid
from dataclasses import dataclass

import wandb
from rdkit import Chem
from llama_index.core.workflow import Context
from llama_index.llms.openrouter import OpenRouter

from rlm.codeact_core import (
    CodeActAgent,
    INDEX_CODEACT_SYSTEM_PROMPT,
    INDEX_FORCE_LOOP_MESSAGE,
    INDEX_OBSERVATION_FOLLOWUP,
    make_simple_code_executor,
    run_agent_verbose,
)
from rlm.codeact_helpers import (
    build_retriever,
    extract_response_text,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import get_tracer, init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 1200.0
SEED = 42
CONTEXT_SIZE = 500
GROUND_TRUTH_FRACTION_PER_CONTEXT = 0.2
RETRIEVER_NAME = "random"
MAX_OUTPUT_TOKENS = 50_000
REASONING_EFFORT = "high"
MAX_ITERATIONS = 8

REACTION_KEY = "cn_bond_formation_connectivity_delta"
REACTION_LABEL = "Net C-N bond formation by connectivity delta"
REACTION_DESCRIPTION = (
    "A reaction forms a C-N bond when the total number of carbon-nitrogen connections "
    "in products is greater than in reactants. Count connectivity only: single, "
    "double, triple, and aromatic C-N bonds each count as one C-N connection. "
    "Ignore reagents."
)


@dataclass
class CNConnectivityResult:
    index: int
    reactant_cn_count: int
    product_cn_count: int
    delta: int
    is_valid: bool


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="CodeAct-Task12",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAct task 12 evaluation.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH)
    parser.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    parser.add_argument(
        "--ground-truth-fraction-per-context",
        type=float,
        default=GROUND_TRUTH_FRACTION_PER_CONTEXT,
    )
    return parser.parse_args()


def parse_reaction_sides(indexed_line: str) -> tuple[int, list[str], list[str]]:
    idx_str, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    if len(parts) != 3:
        raise ValueError("Reaction must have reactants>reagents>products format.")
    reactant_smiles = [s.strip() for s in parts[0].split(".") if s.strip()]
    product_smiles = [s.strip() for s in parts[2].split(".") if s.strip()]
    return int(idx_str), reactant_smiles, product_smiles


def mols_from_smiles(smiles_list: list[str]) -> list[Chem.Mol]:
    mols: list[Chem.Mol] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        mols.append(mol)
    return mols


def count_cn_connections(mols: list[Chem.Mol]) -> int:
    count = 0
    for mol in mols:
        for bond in mol.GetBonds():
            atom_nums = {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()}
            if atom_nums == {6, 7}:
                count += 1
    return count


def analyze_cn_connectivity(indexed_line: str) -> CNConnectivityResult:
    idx = -1
    try:
        idx, reactant_smiles, product_smiles = parse_reaction_sides(indexed_line)
        reactant_mols = mols_from_smiles(reactant_smiles)
        product_mols = mols_from_smiles(product_smiles)
        reactant_cn_count = count_cn_connections(reactant_mols)
        product_cn_count = count_cn_connections(product_mols)
        delta = product_cn_count - reactant_cn_count
        return CNConnectivityResult(
            index=idx,
            reactant_cn_count=reactant_cn_count,
            product_cn_count=product_cn_count,
            delta=delta,
            is_valid=True,
        )
    except Exception:
        return CNConnectivityResult(
            index=idx,
            reactant_cn_count=0,
            product_cn_count=0,
            delta=0,
            is_valid=False,
        )


def ground_truth_indices(lines: list[str]) -> list[int]:
    indices: list[int] = []
    for line in lines:
        result = analyze_cn_connectivity(line)
        if result.is_valid and result.delta > 0:
            indices.append(result.index)
    indices.sort()
    return indices


def build_question() -> str:
    return f"""
    Context: You are given a large string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Task:
    Return all reaction indices that match this reaction type:
    - {REACTION_LABEL}

    Description:
    - {REACTION_DESCRIPTION}

    Guidance:
    - Use RDKit for parsing and bond counting.
    - A C-N connection is any bond where one endpoint atom is carbon and the other endpoint atom is nitrogen.
    - Count all C-N bond types (single/double/triple/aromatic) as one connection each.
    - Ignore reagents (middle field).
    - A reaction matches when product-side C-N connections minus reactant-side C-N connections is > 0.
    - Skip malformed reactions.
    - DO NOT assume/simulate output of the code. Wait for code execution and then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.
    - DO NOT do `exit()` in the code in any case.

    Output format:
    - Return ONLY the matching reaction INDICES.
    - Format must be a comma-separated list of integers in ascending order (e.g., 3,8,21).
    - No other text, quotes, labels, punctuation, or formatting.
    - If no matching reaction exists, return -1.
    """


def build_code_executor(lines: list[str]):
    return make_simple_code_executor(
        extra_locals={"lines": lines},
        extra_globals={
            "np": __import__("numpy"),
            "rdkit": __import__("rdkit"),
        },
    )


async def main(
    model_name: str,
    dataset_path: str,
    context_size: int,
    ground_truth_fraction_per_context: float,
) -> None:
    maybe_init_tracing()
    tracer = get_tracer("codeact-task12")
    lines = load_lines(dataset_path)
    context = "\n".join(lines)
    rng = random.Random(SEED)
    run_session_id = f"codeact-task12-{uuid.uuid4()}"

    full_gt_indices = ground_truth_indices(lines)
    full_gt_by_reaction = {REACTION_KEY: full_gt_indices}
    retriever = build_retriever(
        name=RETRIEVER_NAME,
        lines=lines,
        rng=rng,
        ground_truth_indices_by_reaction=full_gt_by_reaction,
        ground_truth_fraction_per_context=ground_truth_fraction_per_context,
    )
    retriever_name = RETRIEVER_NAME if context_size >= 0 else "all_lines"

    run = wandb.init(
        project="CodeAct-Task12",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": dataset_path,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": SEED,
            "context_size": context_size,
            "ground_truth_fraction_per_context": ground_truth_fraction_per_context,
            "retriever_name": retriever_name,
            "reasoning_effort": REASONING_EFFORT,
            "task_label": REACTION_LABEL,
            "task_description": REACTION_DESCRIPTION,
            "full_ground_truth_indices_by_reaction": full_gt_by_reaction,
            "full_ground_truth_count": len(full_gt_indices),
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    question = build_question()
    if context_size < 0:
        retrieved_context = context
        retrieved_lines = lines
    else:
        retrieved_context = retriever.build_context(query=REACTION_KEY, target_index=-1, k=context_size)
        retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
    context_coverage = len(retrieved_lines) / len(lines) if lines else 0.0
    gt_indices_in_context = ground_truth_indices(retrieved_lines)
    gt_set = set(gt_indices_in_context)

    completion_prompt = f"""
    You are given a subset of chemical reactions in SMILES format and a question.
    <context>
    {retrieved_context}
    </context>
    <question>
    {question}
    </question>
    """
    print("Question 1/1 task=cn_bond_formation_connectivity_delta")
    print(
        f"Ground truth count (full dataset): {len(full_gt_indices)}, "
        f"in-context: {len(gt_indices_in_context)} (context lines: {len(retrieved_lines)})"
    )

    executor = build_code_executor(lines=retrieved_lines)
    agent = CodeActAgent(
        code_execute_fn=executor.execute,
        llm=OpenRouter(
            model=model_name,
            api_key=OPENROUTER_API_KEY,
            max_tokens=MAX_OUTPUT_TOKENS,
            reasoning_effort=REASONING_EFFORT,
            additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
        ),
        system_prompt=INDEX_CODEACT_SYSTEM_PROMPT,
        max_iterations=MAX_ITERATIONS,
        force_loop_message=INDEX_FORCE_LOOP_MESSAGE,
        observation_followup=INDEX_OBSERVATION_FOLLOWUP,
        timeout=WORKFLOW_TIMEOUT_S,
    )
    ctx = Context(agent)

    with tracer.start_as_current_span("codeact_task12_sample_0") as sample_span:
        sample_span.set_attributes(
            {
                "sample.index": 0,
                "sample.count": 1,
                "reaction.key": REACTION_KEY,
                "agent.name": "codeact",
            }
        )
        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": 0,
                "sample_count": 1,
                "reaction_key": REACTION_KEY,
                "agent": "codeact",
            },
            tags=["codeact", "sample", "task12_cn_connectivity"],
        ):
            response = await run_agent_verbose(agent, ctx, completion_prompt)

    response_text = extract_response_text(response)
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

    parsed_indices = parse_indices(response_text)
    pred_set = set(parsed_indices)
    precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
    is_exact_match = pred_set == gt_set

    for metric in llm_turn_metrics:
        wandb.log(
            {
                "sample_iteration": metric["iteration"],
                "sample/0/iteration_input_tokens": metric["iteration_input_tokens"],
                "sample/0/iteration_output_tokens": metric["iteration_output_tokens"],
                "sample/0/iteration_total_tokens": metric["iteration_total_tokens"],
                **(
                    {"sample/0/iteration_cost_usd": metric["iteration_cost_usd"]}
                    if "iteration_cost_usd" in metric
                    else {}
                ),
            }
        )

    final_input_tokens = sum(int(m.get("iteration_input_tokens", 0)) for m in llm_turn_metrics)
    final_output_tokens = sum(int(m.get("iteration_output_tokens", 0)) for m in llm_turn_metrics)
    final_total_tokens = sum(int(m.get("iteration_total_tokens", 0)) for m in llm_turn_metrics)
    final_cost = sum(float(m.get("iteration_cost_usd", 0.0)) for m in llm_turn_metrics)
    has_cost = any("iteration_cost_usd" in m for m in llm_turn_metrics)

    wandb.log(
        {
            "sample_idx": 0,
            "sample/0/reaction_key": REACTION_KEY,
            "sample/0/final_total_input_tokens": final_input_tokens,
            "sample/0/final_total_output_tokens": final_output_tokens,
            "sample/0/final_total_tokens": final_total_tokens,
            "sample/0/iterations": len(llm_turn_metrics),
            "sample/0/is_exact_match": int(is_exact_match),
            "sample/0/precision": precision,
            "sample/0/recall": recall,
            "sample/0/f1": f1,
            "sample/0/ground_truth_count": len(gt_indices_in_context),
            "sample/0/ground_truth_in_context_count": len(gt_indices_in_context),
            "sample/0/ground_truth_full_count": len(full_gt_indices),
            "sample/0/prediction_count": len(parsed_indices),
            "sample/0/ground_truth_indices": ",".join(str(x) for x in gt_indices_in_context),
            "sample/0/predicted_indices": ",".join(str(x) for x in parsed_indices),
            "sample/0/response_raw": response_text,
            "sample/0/completion_prompt_char_count": len(completion_prompt),
            "sample/0/context_char_count": len(retrieved_context),
            "sample/0/retrieved_line_count": len(retrieved_lines),
            "sample/0/context_coverage": context_coverage,
            **({"sample/0/final_total_cost_usd": final_cost} if has_cost else {}),
        }
    )
    wandb.log(
        {
            "running_exact_match_accuracy": float(is_exact_match),
            "running_macro_precision": precision,
            "running_macro_recall": recall,
            "running_macro_f1": f1,
        }
    )

    print(f"Exact match: {int(is_exact_match)}/1")
    print(f"Exact match accuracy: {float(is_exact_match):.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")

    run.summary["exact_match_correct"] = int(is_exact_match)
    run.summary["total"] = 1
    run.summary["exact_match_accuracy"] = float(is_exact_match)
    run.summary["macro_precision"] = precision
    run.summary["macro_recall"] = recall
    run.summary["macro_f1"] = f1
    run.summary["samples_with_cost"] = int(has_cost)
    if has_cost:
        run.summary["total_cost_usd"] = final_cost
        run.summary["avg_cost_per_sample_usd"] = final_cost
    run.summary["full_ground_truth/cn_bond_formation/count"] = len(full_gt_indices)
    run.summary["full_ground_truth/cn_bond_formation/indices"] = ",".join(str(x) for x in full_gt_indices)
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            context_size=args.context_size,
            ground_truth_fraction_per_context=args.ground_truth_fraction_per_context,
        )
    )
