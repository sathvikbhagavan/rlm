import argparse
import asyncio
import os
import random
import uuid

from llama_index.core.workflow import Context
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
from rlm.codeact_core import (
    INDEX_CODEACT_SYSTEM_PROMPT,
    INDEX_FORCE_LOOP_MESSAGE,
    INDEX_OBSERVATION_FOLLOWUP,
    CodeActAgent,
    make_simple_code_executor,
    run_agent_verbose,
)
from rlm.codeact_helpers import extract_response_text, load_lines, precision_recall_f1
from rlm.tracing import get_tracer, init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 900.0
SEED = 42
CONTEXT_SIZE = 100
MAX_OUTPUT_TOKENS = 30_000
REASONING_EFFORT = "high"
MAX_ITERATIONS = 8
LLM_TIMEOUT_RETRIES = 2
LLM_TIMEOUT_RETRY_BACKOFF_S = 2.0
LLM_REQUEST_TIMEOUT_S = 300.0
CODE_EXECUTION_TIMEOUT_S = 300.0


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
    Context: You are given a large string of chemical reactions in SMILES format, separated by newlines.
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

    Molecule identity must use canonical SMILES.
    Do NOT use substructure matching — only exact canonical SMILES equality counts as a match.
    Only consider reactions that appear in the provided context string.
    Do not infer producers or consumers from reactions outside the context.

    Guidance:
    - Use RDKit for SMILES canonicalization.
    - Split multi-component sides on dots (.) and canonicalize each component independently.
    - Skip malformed reactions or malformed molecules.
    - If you copy SMILES text into a Python string literal, handle backslashes safely:
      use a raw string (for example, r\"\"\"...\"\"\") or escape backslashes as "\\\\".

    Output format:
    - Return one canonical SMILES string per line.
    - Sort lines in lexicographic (ascending) order.
    - No other text, quotes, labels, punctuation, or formatting.
    - If no molecule satisfies the criteria, return -1.
    """


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="CodeAct-Task12b",
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
        description="Run CodeAct task 12b — hub molecule identification evaluation."
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
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


def build_code_executor():
    return make_simple_code_executor(
        extra_globals={
            "np": __import__("numpy"),
            "rdkit": __import__("rdkit"),
        },
    )


async def main(model_name: str, context_size: int) -> None:
    maybe_init_tracing()
    tracer = get_tracer("codeact-task12b")
    lines = load_lines(DATASET_PATH)
    run_session_id = f"codeact-task12b-{uuid.uuid4()}"
    full_gt = list(HARDCODED_GT_HUB_MOLECULES)
    full_support = support_indices()
    print(
        f"Ground truth hub molecules={len(full_gt)} "
        f"full_support_indices={len(full_support)}"
    )

    run = wandb.init(
        project="CodeAct-Task12b",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": "protected_hub_ceiling",
            "min_selected_ground_truth": TASK12B_MIN_SELECTED_GROUND_TRUTH,
            "reasoning_effort": REASONING_EFFORT,
            "llm_timeout_retries": LLM_TIMEOUT_RETRIES,
            "llm_timeout_retry_backoff_s": LLM_TIMEOUT_RETRY_BACKOFF_S,
            "llm_request_timeout_s": LLM_REQUEST_TIMEOUT_S,
            "code_execution_timeout_s": CODE_EXECUTION_TIMEOUT_S,
            "num_questions": 1,
            "dag_mode": TASK12B_DAG_MODE,
            "min_downstream": TASK12B_MIN_DOWNSTREAM,
            "task_description": "Identify hub molecules with one producer and many downstream consumers.",
            "ground_truth_definition": TASK12B_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK12B_TOTAL_REACTIONS,
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

    executor = build_code_executor()
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
        llm_timeout_retries=LLM_TIMEOUT_RETRIES,
        llm_timeout_retry_backoff_s=LLM_TIMEOUT_RETRY_BACKOFF_S,
        llm_request_timeout_s=LLM_REQUEST_TIMEOUT_S,
        code_execution_timeout_s=CODE_EXECUTION_TIMEOUT_S,
    )
    ctx = Context(agent)

    with tracer.start_as_current_span("codeact_task12b_sample_0") as sample_span:
        sample_span.set_attributes(
            {
                "sample.index": 0,
                "sample.count": 1,
                "agent.name": "codeact",
            }
        )
        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": 0,
                "sample_count": 1,
                "task": "hub_molecule_identification",
                "agent": "codeact",
                "ground_truth_definition": TASK12B_GROUND_TRUTH_DEFINITION,
            },
            tags=["codeact", "sample", "task12b_HUB_MOLECULE"],
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

    wandb.log(
        {
            "sample_idx": 0,
            "sample/0/final_total_input_tokens": final_input_tokens,
            "sample/0/final_total_output_tokens": final_output_tokens,
            "sample/0/final_total_tokens": final_total_tokens,
            "sample/0/iterations": len(llm_turn_metrics),
            "sample/0/response_raw": response_text,
            "sample/0/response_parsed_molecules": "\n".join(parsed_molecules),
            "sample/0/response_parsed_count": len(parsed_molecules),
            "sample/0/ground_truth_molecules": "\n".join(gt_molecules),
            "sample/0/ground_truth_count": len(gt_molecules),
            "sample/0/ground_truth_full_count": len(full_gt),
            "sample/0/selected_hub_count": sampling.selected_hub_count,
            "sample/0/forced_reaction_count": sampling.forced_count,
            "sample/0/support_indices_in_context": support_in_context,
            "sample/0/support_indices_full_count": len(full_support),
            "sample/0/support_indices_selected_count": len(support),
            "sample/0/is_exact_match": int(is_exact_match),
            "sample/0/precision": precision,
            "sample/0/recall": recall,
            "sample/0/f1": f1,
            "sample/0/completion_prompt_char_count": len(completion_prompt),
            "sample/0/context_char_count": len(retrieved_context),
            "sample/0/retrieved_line_count": len(retrieved_lines),
            "sample/0/context_size": context_size,
            "sample/0/context_coverage": context_coverage,
            **({"sample/0/final_total_cost_usd": final_cost} if has_cost else {}),
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
    run.summary["avg_total_input_tokens_per_sample"] = float(final_input_tokens)
    run.summary["avg_total_output_tokens_per_sample"] = float(final_output_tokens)
    if has_cost:
        run.summary["total_cost_usd"] = final_cost
        run.summary["avg_cost_per_sample_usd"] = final_cost
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
