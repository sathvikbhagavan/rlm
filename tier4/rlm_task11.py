import argparse
import random
import os
import re
import uuid

from task11_synthetic_chain_graph import ground_truth_chains_in_context
from task11_synthetic_chain_ground_truth import (
    FIXED_QUESTIONS,
    HARDCODED_GT_CHAINS,
    TASK11_GROUND_TRUTH_DEFINITION,
    TASK11_MIN_SELECTED_GROUND_TRUTH,
    TASK11_TOTAL_REACTIONS,
    chain_indices_for_question,
    chains_for_context_sampling,
)

import wandb

from rxnhaystack.campaign_metrics import install_campaign_metrics

from rlm import RLM
from rxnhaystack.worker import instrument_rlm_from_environment
from rlm.codeact_helpers import build_context_pipeline, load_lines, precision_recall_f1
from rlm.tracing import init_tracing, using_tracing_attributes

install_campaign_metrics(wandb)

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = __import__("os").environ.get("RXNHAYSTACK_CLEANED_DATASET", __import__("os").path.expanduser("~/datasets/rxnhaystack/reactionSmilesFigShareUSPTO2023_cleaned.txt"))
BACKEND = "openrouter"
MODEL_NAME = __import__("os").environ.get("RXNHAYSTACK_MODEL", "openai/gpt-5-mini")
ENABLE_TRACING = True
SEED = int(__import__("os").environ.get("RXNHAYSTACK_SEED", "42"))
CONTEXT_SIZE = int(__import__("os").environ.get("RXNHAYSTACK_CONTEXT_SIZE", "100"))
CONTEXT_PIPELINE_NAME = "random"

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}


def parse_chains(response: str, chain_length: int) -> list[tuple[int, ...]]:
    response = response.strip()
    if not response:
        return []
    if response.replace(" ", "") == "-1":
        return []

    chains: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    for line in response.splitlines():
        line = line.strip()
        if not line or line == "-1":
            continue
        nums = re.findall(r"\d+", line)
        if len(nums) < chain_length:
            continue
        chain = tuple(int(n) for n in nums[:chain_length])
        if chain not in seen:
            seen.add(chain)
            chains.append(chain)

    return chains


def build_question(start_index: int, chain_length: int) -> str:
    return f"""
    Context: You are given a large string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Each side (reactants / products) may contain multiple molecules separated by dots (.).
    Ignore reagents (middle field between the two > delimiters).

    Task:
    Starting from reaction index {start_index}, find ALL valid synthetic chains of exactly {chain_length} reactions.

    A synthetic chain of length {chain_length} is an ordered sequence of {chain_length} distinct reaction indices
    [r_0, r_1, ..., r_{chain_length - 1}] where r_0 = {start_index} and for every consecutive pair (r_k, r_{{k+1}}),
    at least one PRODUCT of reaction r_k is identical to at least one REACTANT of reaction r_{{k+1}}.

    Molecule identity must be determined by canonical SMILES.
    Do NOT use substructure matching — only exact canonical SMILES equality counts as a match.
    A reaction must NOT appear more than once in the same chain.
    Only consider reactions that appear in the provided context string.
    Do not infer links through molecules from reactions outside the context.

    Guidance:
    - Use RDKit for all SMILES canonicalization and parsing.
    - Split multi-component sides on dots (.) and canonicalize each component independently.
    - Skip malformed reactions or molecules that RDKit cannot parse.
    - Systematically build the product-to-reactant connections from reaction {start_index} outward.
    - DO NOT assume/simulate output of the code. Wait for the code to get executed and only then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.

    Output format:
    - Return each chain as a comma-separated sequence of {chain_length} reaction indices, one chain per line.
    - Sort chains in lexicographic (ascending) order.
    - No other text, quotes, labels, punctuation, or formatting.
    - If no valid chain exists, return -1.
    """


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task11",
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
        description="Run RLM task 11 — synthetic chain evaluation."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for backend (default: {MODEL_NAME}).",
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


def _fmt_chains(chains: list[tuple[int, ...]], limit: int = 20) -> str:
    shown = [",".join(str(x) for x in chain) for chain in chains[:limit]]
    suffix = f" … +{len(chains) - limit} more" if len(chains) > limit else ""
    return "; ".join(shown) + suffix


def main(model_name: str, context_size: int) -> None:
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)

    for start_idx, q_chain_length in FIXED_QUESTIONS:
        full_gt = HARDCODED_GT_CHAINS[(start_idx, q_chain_length)]
        print(
            f"Ground truth [start={start_idx}, chain_length={q_chain_length}] "
            f"full_dataset={len(full_gt)} chains "
            f"support_indices={len(chain_indices_for_question(start_idx, q_chain_length))}"
        )

    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**instrument_rlm_from_environment(rlm_init_kwargs))
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    run = wandb.init(
        project="RLMs-Task11",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "min_selected_ground_truth": TASK11_MIN_SELECTED_GROUND_TRUTH,
            "num_questions": len(FIXED_QUESTIONS),
            "fixed_questions": FIXED_QUESTIONS,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Synthetic chain identification — pairwise reaction analysis.",
            "ground_truth_definition": TASK11_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK11_TOTAL_REACTIONS,
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
    total_input_tokens = 0
    total_output_tokens = 0
    samples_run = 0

    for i, (start_idx, q_chain_length) in enumerate(FIXED_QUESTIONS):
        question = build_question(start_index=start_idx, chain_length=q_chain_length)
        full_gt_chains = HARDCODED_GT_CHAINS[(start_idx, q_chain_length)]
        full_support_indices = chain_indices_for_question(start_idx, q_chain_length)
        sampling = chains_for_context_sampling(start_idx, q_chain_length, context_size)
        support_indices = set(sampling.support_indices)

        context_pipeline = build_context_pipeline(
            name=CONTEXT_PIPELINE_NAME,
            lines=lines,
            rng=random.Random(SEED + i),
            min_selected_ground_truth=TASK11_MIN_SELECTED_GROUND_TRUTH,
        )
        sample_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=support_indices,
            query=f"synthetic_chain_{start_idx}_L{q_chain_length}",
        )
        context_lines = [line for line in sample_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in context_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        gt_chains = ground_truth_chains_in_context(
            context_lines,
            start_index=start_idx,
            chain_length=q_chain_length,
        )
        gt_set = set(gt_chains)
        context_coverage = len(context_lines) / len(lines) if lines else 0.0
        support_in_context = len(support_indices & context_indices)

        print(
            f"\nQuestion {i + 1}/{len(FIXED_QUESTIONS)}: start_idx={start_idx}, "
            f"chain_length={q_chain_length}"
        )
        print(
            f"[CONTEXT] requested_size={context_size} actual_size={len(context_lines)} "
            f"selected_chains={sampling.selected_chain_count}/{len(full_gt_chains)} "
            f"forced_count={sampling.forced_count} "
            f"support_in_context={support_in_context}/{len(support_indices)} "
            f"full_support={len(full_support_indices)} "
            f"gt_in_context={len(gt_chains)}/{sampling.selected_chain_count} "
            f"coverage={context_coverage:.4f}"
        )

        completion_kwargs = {"prompt": sample_context, "root_prompt": question}

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(FIXED_QUESTIONS),
                "task": "synthetic_chain",
                "start_idx": start_idx,
                "chain_length": q_chain_length,
                "ground_truth_definition": TASK11_GROUND_TRUTH_DEFINITION,
            },
            tags=["run_rlms", "sample", "task11_SYNTHETIC_CHAIN"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed_chains = parse_chains(response, q_chain_length)
        pred_set = set(parsed_chains)
        precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
        sample_cost_usd = completion.usage_summary.total_cost
        is_exact_match = pred_set == gt_set

        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        print(
            f"Response [start={start_idx}]: "
            f"{response[:500]}{'…' if len(response) > 500 else ''}"
        )
        print(f"Predicted [start={start_idx}]: {len(parsed_chains)} chains")
        print(f"Ground truth [start={start_idx}]: {len(gt_chains)} chains")
        print(
            f"Metrics [start={start_idx}] -> precision={precision:.4f} "
            f"recall={recall:.4f} f1={f1:.4f} exact_match={is_exact_match}"
        )

        for metric in iteration_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                    f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                    f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                }
            )

        final_input_tokens = 0
        final_output_tokens = 0
        final_total_tokens = 0
        if iteration_metrics:
            last_metric = iteration_metrics[-1]
            final_input_tokens = int(last_metric["total_input_tokens"])
            final_output_tokens = int(last_metric["total_output_tokens"])
            final_total_tokens = int(last_metric["total_tokens"])
            total_input_tokens += final_input_tokens
            total_output_tokens += final_output_tokens
        samples_run += 1

        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/start_idx": start_idx,
                f"sample/{i}/chain_length": q_chain_length,
                f"sample/{i}/final_total_input_tokens": final_input_tokens,
                f"sample/{i}/final_total_output_tokens": final_output_tokens,
                f"sample/{i}/final_total_tokens": final_total_tokens,
                f"sample/{i}/iterations": len(iteration_metrics),
                f"sample/{i}/response_raw": response,
                f"sample/{i}/response_parsed_chains": _fmt_chains(parsed_chains),
                f"sample/{i}/response_parsed_count": len(parsed_chains),
                f"sample/{i}/ground_truth_chains": _fmt_chains(gt_chains),
                f"sample/{i}/ground_truth_count": len(gt_chains),
                f"sample/{i}/ground_truth_full_count": len(full_gt_chains),
                f"sample/{i}/selected_chain_count": sampling.selected_chain_count,
                f"sample/{i}/forced_reaction_count": sampling.forced_count,
                f"sample/{i}/support_indices_in_context": support_in_context,
                f"sample/{i}/support_indices_full_count": len(full_support_indices),
                f"sample/{i}/support_indices_selected_count": len(support_indices),
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/is_exact_match": int(is_exact_match),
                f"sample/{i}/completion_root_prompt": question,
                f"sample/{i}/completion_prompt_char_count": len(sample_context),
                f"sample/{i}/context_size": context_size,
                f"sample/{i}/context_coverage": context_coverage,
                f"sample/{i}/retrieved_line_count": len(context_lines),
                **(
                    {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                    if sample_cost_usd is not None
                    else {}
                ),
            }
        )
        wandb.log(
            {
                "running_exact_match_accuracy": exact_match_count / samples_run,
                "running_macro_precision": macro_precision / samples_run,
                "running_macro_recall": macro_recall / samples_run,
                "running_macro_f1": macro_f1 / samples_run,
            }
        )

    total = samples_run
    exact_match_accuracy = (exact_match_count / total) if total else 0.0
    macro_precision = (macro_precision / total) if total else 0.0
    macro_recall = (macro_recall / total) if total else 0.0
    macro_f1 = (macro_f1 / total) if total else 0.0

    print(f"\n{'=' * 60}")
    print(f"Exact match: {exact_match_count}/{total}")
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    for start_idx, q_chain_length in FIXED_QUESTIONS:
        full_gt = HARDCODED_GT_CHAINS[(start_idx, q_chain_length)]
        run.summary[f"ground_truth/start_{start_idx}/count"] = len(full_gt)
        run.summary[f"ground_truth/start_{start_idx}/chain_length"] = q_chain_length

    run.summary["exact_match_correct"] = exact_match_count
    run.summary["total"] = total
    run.summary["exact_match_accuracy"] = exact_match_accuracy
    run.summary["macro_precision"] = macro_precision
    run.summary["macro_recall"] = macro_recall
    run.summary["macro_f1"] = macro_f1
    run.summary["avg_total_input_tokens_per_sample"] = (
        total_input_tokens / total if total else 0.0
    )
    run.summary["avg_total_output_tokens_per_sample"] = (
        total_output_tokens / total if total else 0.0
    )
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        context_size=args.context_size,
    )
