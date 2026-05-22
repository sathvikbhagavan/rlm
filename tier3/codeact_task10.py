import argparse
import asyncio
import json
import os
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass

import wandb
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
from rlm.codeact_helpers import extract_response_text, parse_indices, precision_recall_f1
from rlm.tracing import get_tracer, init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
NEGATIVE_REACTIONS_PATH = "/home/bhagavan/rlms/rlm/tier3/negative_reactions.jsonl"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 1200.0
SEED = 42
NEGATIVES_PER_QUESTION = 10
CONTEXT_SIZE = 500
MAX_OUTPUT_TOKENS = 50_000
REASONING_EFFORT = "high"
MAX_ITERATIONS = 8
# os.environ["WANDB_MODE"] = "disabled"

NEGATIVE_TYPE_LABELS: dict[str, str] = {
    "structurally_impossible": "Structurally impossible reactions",
    "conservation_violation": "Conservation-violating reactions",
    "product_swapping": "Product-swapped reactions",
    "handcrafted_wrong_reagent_or_product": "Hand-crafted wrong reagent/product reactions",
}

NEGATIVE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "structurally_impossible": (
        "Flag any reaction whose product-side molecule is invalid: SMILES cannot be parsed, "
        "sanitization fails, or chemistry is impossible (e.g., valence violations). "
        "Do not flag a reaction only because it is uncommon; it must be structurally invalid."
    ),
    "conservation_violation": (
        "Flag reactions where at least one element symbol present in products is absent from "
        "the combined reactants+reagents side. Compare element identity only (not atom counts), "
        "because stoichiometric imbalance is common in USPTO data."
    ),
    "product_swapping": (
        "Flag reactions whose reactants/reagents and products are semantically mismatched, "
        "as if the product side belongs to a different source reaction. The molecules may be "
        "valid individually, but the transformation is inconsistent as a pair."
    ),
    "handcrafted_wrong_reagent_or_product": (
        "Flag manually corrupted reactions such as irrelevant reagent injection, product "
        "replacement with an inappropriate reagent/reactant, or other edits that break "
        "reaction plausibility without necessarily breaking SMILES syntax."
    ),
}


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="CodeAct-Task10",
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
        description="Run CodeAct task 10 with injected incorrect reaction equations."
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--negatives-per-question", type=int, default=NEGATIVES_PER_QUESTION)
    parser.add_argument(
        "--context-size",
        type=int,
        default=CONTEXT_SIZE,
        help="Number of reaction lines to include in context; use -1 for full context.",
    )
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH)
    parser.add_argument("--negative-reactions-path", type=str, default=NEGATIVE_REACTIONS_PATH)
    return parser.parse_args()


def load_raw_reactions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


@dataclass
class NegativeRecord:
    source_index_1: int | None
    corrupted_reaction: str


def load_negatives_by_type(path: str) -> dict[str, list[NegativeRecord]]:
    grouped: dict[str, list[NegativeRecord]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            ctype = payload.get("corruption_type")
            source_index_1 = payload.get("source_index_1")
            rxn = payload.get("corrupted_reaction")
            if isinstance(ctype, str) and isinstance(rxn, str) and ctype in NEGATIVE_TYPE_LABELS:
                grouped[ctype].append(
                    NegativeRecord(
                        source_index_1=source_index_1 if isinstance(source_index_1, int) else None,
                        corrupted_reaction=rxn,
                    )
                )
    return grouped


def build_question(reaction_label: str, reaction_description: str) -> str:
    return f"""
    Context: You are given a list of chemical reactions in SMILES format, separated by newlines.
    Each reaction line is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Task:
    Return all indices that correspond to:
    - {reaction_label}

    Description:
    - {reaction_description}

    Guidance:
    - Use RdKit for all analysis and counting.
    - Only identify reactions matching the target corruption type above.
    - Ignore valid reactions.
    - The index to return is the integer at the start of each line in this context.
    - DO NOT assume/simulate output of the code. Wait for the code to get executed and only then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.
    - DO NOT do `exit()` in the code in any case.

    Output format:
    - Return ONLY the matching reaction INDICES.
    - Format must be a comma-separated list of integers in ascending order (e.g., 3,8,21).
    - No other text, quotes, labels, punctuation, or formatting.
    - If no matching reaction exists, return -1.
    """


def build_context(
    dataset_reactions: list[str],
    negative_pool: list[NegativeRecord],
    negatives_per_question: int,
    context_size: int,
    rng: random.Random,
) -> tuple[str, list[str], list[int]]:
    if len(negative_pool) < negatives_per_question:
        raise ValueError("Not enough negatives for requested negatives_per_question.")
    selected = negative_pool[:negatives_per_question]

    selected_indices = {
        r.source_index_1
        for r in selected
        if isinstance(r.source_index_1, int) and r.source_index_1 >= 0
    }
    all_detected_indices = {
        r.source_index_1
        for r in negative_pool
        if isinstance(r.source_index_1, int) and r.source_index_1 >= 0
    }
    excluded_indices = all_detected_indices - selected_indices

    selected_overrides = {
        r.source_index_1: r.corrupted_reaction
        for r in selected
        if isinstance(r.source_index_1, int) and r.source_index_1 >= 0
    }

    all_context_rows: list[tuple[int, str, bool]] = []
    for idx, base_reaction in enumerate(dataset_reactions):
        if idx in excluded_indices:
            continue
        reaction = selected_overrides.get(idx, base_reaction)
        all_context_rows.append((idx, f"{idx} {reaction}", idx in selected_indices))

    if context_size >= 0:
        if context_size < len(selected_indices):
            raise ValueError(
                f"context_size={context_size} is smaller than injected negatives "
                f"({len(selected_indices)}). Increase context_size or reduce negatives_per_question."
            )
        if context_size < len(all_context_rows):
            negative_rows = [row for row in all_context_rows if row[2]]
            non_negative_rows = [row for row in all_context_rows if not row[2]]
            keep_non_negative = context_size - len(negative_rows)
            sampled_non_negative_rows = (
                rng.sample(non_negative_rows, k=keep_non_negative)
                if keep_non_negative < len(non_negative_rows)
                else non_negative_rows
            )
            trimmed_rows = negative_rows + sampled_non_negative_rows
            trimmed_rows.sort(key=lambda row: row[0])
            context_lines = [row[1] for row in trimmed_rows]
        else:
            context_lines = [row[1] for row in all_context_rows]
    else:
        context_lines = [row[1] for row in all_context_rows]

    gt_indices = sorted(selected_indices)
    return "\n".join(context_lines), context_lines, gt_indices


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
    negatives_per_question: int,
    context_size: int,
    dataset_path: str,
    negative_reactions_path: str,
) -> None:
    maybe_init_tracing()
    tracer = get_tracer("codeact-task10")
    rng = random.Random(SEED)
    reaction_keys = list(NEGATIVE_TYPE_LABELS.keys())
    run_session_id = f"codeact-task10-{uuid.uuid4()}"

    valid_reactions = load_raw_reactions(dataset_path)
    negatives_by_type = load_negatives_by_type(negative_reactions_path)

    for reaction_key in reaction_keys:
        available = len(negatives_by_type.get(reaction_key, []))
        if available < negatives_per_question:
            raise ValueError(
                f"Not enough negatives for {reaction_key}: "
                f"required={negatives_per_question}, available={available}"
            )

    run = wandb.init(
        project="CodeAct-Task10",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": dataset_path,
            "negative_reactions_path": negative_reactions_path,
            "num_questions": len(reaction_keys),
            "context_size": context_size,
            "negatives_per_question": negatives_per_question,
            "seed": SEED,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "reasoning_effort": REASONING_EFFORT,
            "task_description": "Find injected incorrect reactions by corruption type.",
            "full_ground_truth_indices_by_reaction": {
                k: ",".join(
                    str(r.source_index_1)
                    for r in negatives_by_type.get(k, [])
                    if isinstance(r.source_index_1, int) and r.source_index_1 >= 0
                )
                for k in reaction_keys
            },
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    exact_match_count = 0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, reaction_key in enumerate(reaction_keys):
        reaction_label = NEGATIVE_TYPE_LABELS[reaction_key]
        reaction_description = NEGATIVE_TYPE_DESCRIPTIONS[reaction_key]
        question = build_question(reaction_label=reaction_label, reaction_description=reaction_description)
        context, context_lines, gt_indices = build_context(
            dataset_reactions=valid_reactions,
            negative_pool=negatives_by_type[reaction_key],
            negatives_per_question=negatives_per_question,
            context_size=context_size,
            rng=rng,
        )
        gt_set = set(gt_indices)
        context_coverage = (len(context_lines) / len(valid_reactions)) if valid_reactions else 0.0

        completion_prompt = f"""
        You are given a subset of chemical reactions in SMILES format and a question.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """
        print(f"Question {i + 1}/{len(reaction_keys)} type={reaction_key}")
        print(f"Injected negatives in context: {len(gt_indices)} (context lines: {len(context_lines)})")

        executor = build_code_executor(lines=context_lines)
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

        with tracer.start_as_current_span(f"codeact_task10_sample_{i}") as sample_span:
            sample_span.set_attributes(
                {
                    "sample.index": i,
                    "sample.count": len(reaction_keys),
                    "task.name": "injected_negative_detection",
                    "corruption_type": reaction_key,
                    "agent.name": "codeact",
                }
            )
            with using_tracing_attributes(
                session_id=run_session_id,
                metadata={
                    "sample_index": i,
                    "sample_count": len(reaction_keys),
                    "task": "injected_negative_detection",
                    "corruption_type": reaction_key,
                    "agent": "codeact",
                },
                tags=["codeact", "sample", "task10_negative_injection"],
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

        parsed = parse_indices(response_text)
        pred_set = set(parsed)
        precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
        is_exact_match = pred_set == gt_set
        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

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

        final_input_tokens = sum(int(m.get("iteration_input_tokens", 0)) for m in llm_turn_metrics)
        final_output_tokens = sum(int(m.get("iteration_output_tokens", 0)) for m in llm_turn_metrics)
        final_total_tokens = sum(int(m.get("iteration_total_tokens", 0)) for m in llm_turn_metrics)
        final_cost = sum(float(m.get("iteration_cost_usd", 0.0)) for m in llm_turn_metrics)
        has_cost = any("iteration_cost_usd" in m for m in llm_turn_metrics)
        if has_cost:
            total_cost_usd += final_cost
            samples_with_cost += 1

        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/corruption_type": reaction_key,
                f"sample/{i}/final_total_input_tokens": final_input_tokens,
                f"sample/{i}/final_total_output_tokens": final_output_tokens,
                f"sample/{i}/final_total_tokens": final_total_tokens,
                f"sample/{i}/iterations": len(llm_turn_metrics),
                f"sample/{i}/response_raw": response_text,
                f"sample/{i}/response_parsed_indices": ",".join(str(x) for x in parsed),
                f"sample/{i}/response_parsed_count": len(parsed),
                f"sample/{i}/ground_truth_indices": ",".join(str(x) for x in gt_indices),
                f"sample/{i}/ground_truth_count": len(gt_indices),
                f"sample/{i}/prediction_count": len(parsed),
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/is_exact_match": int(is_exact_match),
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(context),
                f"sample/{i}/retrieved_line_count": len(context_lines),
                f"sample/{i}/context_coverage": context_coverage,
                **({f"sample/{i}/final_total_cost_usd": final_cost} if has_cost else {}),
            }
        )
        wandb.log(
            {
                "running_exact_match_accuracy": exact_match_count / (i + 1),
                "running_macro_precision": macro_precision / (i + 1),
                "running_macro_recall": macro_recall / (i + 1),
                "running_macro_f1": macro_f1 / (i + 1),
            }
        )

    total = len(reaction_keys)
    exact_match_accuracy = (exact_match_count / total) if total else 0.0
    macro_precision = (macro_precision / total) if total else 0.0
    macro_recall = (macro_recall / total) if total else 0.0
    macro_f1 = (macro_f1 / total) if total else 0.0
    print(f"Exact match: {exact_match_count}/{total}")
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    run.summary["exact_match_correct"] = exact_match_count
    run.summary["total"] = total
    run.summary["exact_match_accuracy"] = exact_match_accuracy
    run.summary["macro_precision"] = macro_precision
    run.summary["macro_recall"] = macro_recall
    run.summary["macro_f1"] = macro_f1
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    for reaction_key in reaction_keys:
        run.summary[f"full_ground_truth/{reaction_key}/count"] = len(
            [
                r
                for r in negatives_by_type.get(reaction_key, [])
                if isinstance(r.source_index_1, int) and r.source_index_1 >= 0
            ]
        )
        run.summary[f"full_ground_truth/{reaction_key}/indices"] = ",".join(
            str(r.source_index_1)
            for r in negatives_by_type.get(reaction_key, [])
            if isinstance(r.source_index_1, int) and r.source_index_1 >= 0
        )
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            model_name=args.model_name,
            negatives_per_question=args.negatives_per_question,
            context_size=args.context_size,
            dataset_path=args.dataset_path,
            negative_reactions_path=args.negative_reactions_path,
        )
    )
