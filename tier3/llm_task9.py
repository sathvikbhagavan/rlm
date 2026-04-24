import argparse
import asyncio
import os
import random
import uuid

import wandb
from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter

from rlm.codeact_helpers import (
    build_reaction_index_question as build_question,
    build_reaction_query,
    build_retriever,
    extract_response_text,
    extract_usage_metrics,
    ground_truth_indices,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
SEED = 42
CONTEXT_SIZE = 10
GROUND_TRUTH_FRACTION_PER_CONTEXT = 0.1
RETRIEVER_NAME = "random"
MAX_OUTPUT_TOKENS = 30_000
REASONING_EFFORT = "high"


NAMED_REACTIONS_SMIRKS: dict[str, str] = {
    "suzuki_coupling_with_boronic_acids": "[#6;$([#6]:[#6]),$([#6]=[#6]),$([#6]#[#6]);+0:1]-[B;H0;D3;+0](-[O;H1;D1;+0])-[O;H1;D1;+0].[#6;$([#6]=[#6]),$([#6]~[#6]:[#6]),$([#6]~n);+0:2][Cl,Br,I]>>[#6;$([#6]:[#6]),$([#6]=[#6]),$([#6]#[#6]);+0:1]-[#6;$([#6]=[#6]),$([#6]~[#6]:[#6]),$([#6]~n);+0:2]",
}

NAMED_REACTIONS_LABELS: dict[str, str] = {
    "suzuki_coupling_with_boronic_acids": "Suzuki coupling with boronic acids",
}

NAMED_REACTIONS_DESCRIPTIONS: dict[str, str] = {
    "suzuki_coupling_with_boronic_acids": "A Suzuki cross-coupling reaction in which a boronic acid reacts with an organohalide under palladium catalysis to form a new carbon-carbon bond. The boronic acid partner is restricted to aryl, vinyl, or alkynyl carbons attached to a B(OH)2 group, while the halide partner carries a chlorine, bromine, or iodide leaving group on an aryl, vinyl, or heteroaryl carbon. In the product, the boron moiety and halide are both lost, and a direct C-C bond forms between the two coupling partners. Both sides of the coupling are restricted to sp2 or sp carbons, consistent with the mechanistic requirements of oxidative addition and transmetalation in the Suzuki catalytic cycle.",
}


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="LLM-Task9",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM task 9 evaluation.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    parser.add_argument(
        "--ground-truth-fraction-per-context",
        type=float,
        default=GROUND_TRUTH_FRACTION_PER_CONTEXT,
    )
    return parser.parse_args()


async def main(
    model_name: str,
    context_size: int,
    ground_truth_fraction_per_context: float,
) -> None:
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    rng = random.Random(SEED)
    reaction_keys = list(NAMED_REACTIONS_SMIRKS.keys())
    run_session_id = f"llm-task9-{uuid.uuid4()}"

    full_gt_indices_by_reaction: dict[str, list[int]] = {}
    for reaction_key in reaction_keys:
        query_reaction = build_reaction_query(NAMED_REACTIONS_SMIRKS[reaction_key])
        full_gt_indices_by_reaction[reaction_key] = ground_truth_indices(lines, query_reaction)

    retriever = build_retriever(
        name=RETRIEVER_NAME,
        lines=lines,
        rng=rng,
        ground_truth_indices_by_reaction=full_gt_indices_by_reaction,
        ground_truth_fraction_per_context=ground_truth_fraction_per_context,
    )
    llm = OpenRouter(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
    )

    run = wandb.init(
        project="LLM-Task9",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "ground_truth_fraction_per_context": ground_truth_fraction_per_context,
            "retriever_name": RETRIEVER_NAME,
            "reasoning_effort": REASONING_EFFORT,
            "task_description": "Return reaction indices for named reaction patterns.",
            "NAMED_REACTIONS_SMIRKS": NAMED_REACTIONS_SMIRKS,
            "full_ground_truth_indices_by_reaction": full_gt_indices_by_reaction,
            "mode": "llm_baseline_no_tools",
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
        reaction_label = NAMED_REACTIONS_LABELS[reaction_key]
        reaction_description = NAMED_REACTIONS_DESCRIPTIONS[reaction_key]
        reaction_smirks = NAMED_REACTIONS_SMIRKS[reaction_key]
        query_reaction = build_reaction_query(reaction_smirks)
        question = build_question(reaction_label=reaction_label, reaction_description=reaction_description)
        retrieved_context = retriever.build_context(query=reaction_key, target_index=-1, k=context_size)
        retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
        context_coverage = len(retrieved_lines) / len(lines) if lines else 0.0
        gt_indices = ground_truth_indices(retrieved_lines, query_reaction)
        gt_set = set(gt_indices)
        total_gt_count = len(full_gt_indices_by_reaction[reaction_key])

        completion_prompt = f"""
        You are given a subset of chemical reactions in SMILES format and a question.
        <context>
        {retrieved_context}
        </context>
        <question>
        {question}
        </question>
        """
        print(f"Question {i + 1}/{len(reaction_keys)} for reaction_key={reaction_key}")
        print(
            f"Ground truth present in context: {len(gt_indices)}/{total_gt_count} "
            f"(context lines: {len(retrieved_lines)})"
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(reaction_keys),
                "reaction_key": reaction_key,
                "reaction_smirks": reaction_smirks,
                "agent": "llm_baseline",
            },
            tags=["llm-baseline", "sample", "task9_named_reactions"],
        ):
            response = await llm.achat([ChatMessage(role="user", content=completion_prompt)])

        response_text = extract_response_text(response)
        parsed_indices = parse_indices(response_text)
        pred_set = set(parsed_indices)
        precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
        is_exact_match = pred_set == gt_set
        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        usage_metrics = extract_usage_metrics(response)
        prompt_tokens = int(usage_metrics.get("prompt_tokens", 0))
        completion_tokens = int(usage_metrics.get("completion_tokens", 0))
        total_tokens = int(usage_metrics.get("total_tokens", 0))
        sample_cost = (
            float(usage_metrics["cost_usd"]) if "cost_usd" in usage_metrics else None
        )
        if total_tokens == 0:
            prompt_tokens = count_tokens([{"role": "user", "content": completion_prompt}], model_name)
            completion_tokens = count_tokens(
                [{"role": "assistant", "content": response_text}],
                model_name,
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
                f"sample/{i}/reaction_key": reaction_key,
                f"sample/{i}/reaction_smirks": reaction_smirks,
                f"sample/{i}/final_total_input_tokens": prompt_tokens,
                f"sample/{i}/final_total_output_tokens": completion_tokens,
                f"sample/{i}/final_total_tokens": total_tokens,
                f"sample/{i}/iterations": 1,
                f"sample/{i}/is_exact_match": int(is_exact_match),
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/ground_truth_count": len(gt_indices),
                f"sample/{i}/prediction_count": len(parsed_indices),
                f"sample/{i}/ground_truth_indices": ",".join(str(x) for x in gt_indices),
                f"sample/{i}/predicted_indices": ",".join(str(x) for x in parsed_indices),
                f"sample/{i}/response_raw": response_text,
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_coverage": context_coverage,
                **({f"sample/{i}/final_total_cost_usd": sample_cost} if sample_cost is not None else {}),
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
            full_gt_indices_by_reaction[reaction_key]
        )
        run.summary[f"full_ground_truth/{reaction_key}/indices"] = ",".join(
            str(x) for x in full_gt_indices_by_reaction[reaction_key]
        )
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            model_name=args.model_name,
            context_size=args.context_size,
            ground_truth_fraction_per_context=args.ground_truth_fraction_per_context,
        )
    )
