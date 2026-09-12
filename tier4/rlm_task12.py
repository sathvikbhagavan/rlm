import argparse
import random
import os
import re
import uuid

from task12_longest_chain_graph import ground_truth_longest_chain_in_context
from task12_longest_chain_ground_truth import (
    FIXED_TARGET_PRODUCTS,
    HARDCODED_GT_LONGEST_CHAIN,
    TASK12_DAG_MODE,
    TASK12_GROUND_TRUTH_DEFINITION,
    TASK12_TOTAL_REACTIONS,
    chain_indices_for_product,
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


def parse_chain(response: str) -> tuple[int, ...]:
    response = response.strip()
    if not response:
        return tuple()
    if response.replace(" ", "") == "-1":
        return tuple()

    for line in response.splitlines():
        line = line.strip()
        if not line or line == "-1":
            continue
        nums = re.findall(r"\d+", line)
        if nums:
            return tuple(int(n) for n in nums)
    return tuple()


def common_prefix_len(pred: tuple[int, ...], gt: tuple[int, ...]) -> int:
    prefix_len = 0
    for pred_idx, gt_idx in zip(pred, gt, strict=False):
        if pred_idx != gt_idx:
            break
        prefix_len += 1
    return prefix_len


def position_accuracy(pred: tuple[int, ...], gt: tuple[int, ...]) -> float:
    denom = max(len(pred), len(gt), 1)
    matches = sum(1 for pred_idx, gt_idx in zip(pred, gt, strict=False) if pred_idx == gt_idx)
    return matches / denom


def lcs_length(pred: tuple[int, ...], gt: tuple[int, ...]) -> int:
    n = len(pred)
    m = len(gt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        pred_idx = pred[i - 1]
        for j in range(1, m + 1):
            if pred_idx == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def normalized_edit_distance(pred: tuple[int, ...], gt: tuple[int, ...]) -> float:
    n = len(pred)
    m = len(gt)
    if n == 0 and m == 0:
        return 0.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[n][m] / max(n, m)


def build_question(target_product_smiles: str) -> str:
    return f"""
    Context: You are given a large string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Each side (reactants / products) may contain multiple molecules separated by dots (.).
    Ignore reagents (middle field between the two > delimiters).

    Task:
    Find the SINGLE longest synthetic chain of reactions that produces target product:
    {target_product_smiles}

    A synthetic chain is an ordered sequence of DISTINCT reaction indices [r_0, r_1, ..., r_k]
    where for every consecutive pair (r_i, r_{{i+1}}), at least one PRODUCT of reaction r_i
    is identical to at least one REACTANT of reaction r_{{i+1}} by canonical SMILES equality.

    A chain "produces the target product" if the final reaction r_k has the target product
    among its products (canonical SMILES match).

    IMPORTANT DAG RULE:
    - To avoid cycles, only allow links from lower index to higher index (index_asc):
      r_i < r_{{i+1}}.

    Molecule identity must use canonical SMILES.
    Do NOT use substructure matching — only exact canonical SMILES equality counts as a match.
    Only consider reactions that appear in the provided context string.
    Do not infer links through molecules from reactions outside the context.

    If multiple longest chains have the same maximum length, return the lexicographically
    smallest chain of indices.

    Guidance:
    - Use RDKit for SMILES canonicalization.
    - Skip malformed reactions or malformed molecules.
    - DO NOT assume/simulate output of the code. Wait for the code to get executed and only then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.

    Output format:
    - Return ONLY one comma-separated sequence of reaction indices for the longest chain.
    - No other text, labels, punctuation, or formatting.
    - If no producing chain exists, return -1.
    """


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task12",
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
        description="Run RLM task 12 — longest product chain (DAG) evaluation."
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


def main(model_name: str, context_size: int) -> None:
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)

    for target_product in FIXED_TARGET_PRODUCTS:
        full_gt = HARDCODED_GT_LONGEST_CHAIN[target_product]
        print(
            f"Ground truth [product={target_product}] "
            f"full_dataset_len={len(full_gt)} "
            f"support_indices={len(chain_indices_for_product(target_product))}"
        )

    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**instrument_rlm_from_environment(rlm_init_kwargs))
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    run = wandb.init(
        project="RLMs-Task12",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "num_questions": len(FIXED_TARGET_PRODUCTS),
            "fixed_target_products": FIXED_TARGET_PRODUCTS,
            "dag_mode": TASK12_DAG_MODE,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Find longest DAG chain that produces a target product.",
            "ground_truth_definition": TASK12_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK12_TOTAL_REACTIONS,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    exact_match_count = 0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    macro_prefix_ratio = 0.0
    macro_position_accuracy = 0.0
    macro_lcs_ratio = 0.0
    macro_norm_edit_distance = 0.0
    total_cost_usd = 0.0
    samples_with_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    samples_run = 0

    for i, target_product in enumerate(FIXED_TARGET_PRODUCTS):
        question = build_question(target_product_smiles=target_product)
        full_gt_chain = HARDCODED_GT_LONGEST_CHAIN[target_product]
        support_indices = chain_indices_for_product(target_product)

        context_pipeline = build_context_pipeline(
            name=CONTEXT_PIPELINE_NAME,
            lines=lines,
            rng=random.Random(SEED + i),
            min_selected_ground_truth=len(support_indices),
        )
        sample_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=support_indices,
            query=f"longest_product_chain_{i}",
        )
        context_lines = [line for line in sample_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in context_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        gt_chain = ground_truth_longest_chain_in_context(
            context_lines,
            target_product,
            dag_mode=TASK12_DAG_MODE,
        )
        context_coverage = len(context_lines) / len(lines) if lines else 0.0
        support_in_context = len(support_indices & context_indices)

        print(
            f"\nQuestion {i + 1}/{len(FIXED_TARGET_PRODUCTS)}: "
            f"target_product={target_product}"
        )
        print(
            f"[CONTEXT] requested_size={context_size} actual_size={len(context_lines)} "
            f"support_in_context={support_in_context}/{len(support_indices)} "
            f"gt_in_context={len(gt_chain)}/{len(full_gt_chain)} "
            f"coverage={context_coverage:.4f}"
        )

        completion_kwargs = {"prompt": sample_context, "root_prompt": question}

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(FIXED_TARGET_PRODUCTS),
                "task": "longest_product_chain_dag",
                "target_product": target_product,
                "gt_chain_length": len(gt_chain),
                "ground_truth_definition": TASK12_GROUND_TRUTH_DEFINITION,
            },
            tags=["run_rlms", "sample", "task12_LONGEST_PRODUCT_CHAIN"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        pred_chain = parse_chain(response)
        pred_set = set(pred_chain)
        gt_set = set(gt_chain)
        precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
        prefix_len = common_prefix_len(pred_chain, gt_chain)
        prefix_ratio = prefix_len / len(gt_chain) if gt_chain else 0.0
        pos_acc = position_accuracy(pred_chain, gt_chain)
        lcs_len = lcs_length(pred_chain, gt_chain)
        lcs_ratio = lcs_len / len(gt_chain) if gt_chain else 0.0
        norm_edit_distance = normalized_edit_distance(pred_chain, gt_chain)
        sample_cost_usd = completion.usage_summary.total_cost
        is_exact_match = pred_chain == gt_chain

        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        macro_prefix_ratio += prefix_ratio
        macro_position_accuracy += pos_acc
        macro_lcs_ratio += lcs_ratio
        macro_norm_edit_distance += norm_edit_distance
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        print(
            f"Response [target={target_product}]: "
            f"{response[:500]}{'…' if len(response) > 500 else ''}"
        )
        print(f"Predicted chain length [target={target_product}]: {len(pred_chain)}")
        print(f"Ground truth chain length [target={target_product}]: {len(gt_chain)}")
        print(
            f"Metrics [target={target_product}] -> precision={precision:.4f} "
            f"recall={recall:.4f} f1={f1:.4f} exact_match={is_exact_match} "
            f"prefix_ratio={prefix_ratio:.4f} position_acc={pos_acc:.4f} "
            f"lcs_ratio={lcs_ratio:.4f} norm_edit_dist={norm_edit_distance:.4f}"
        )

        gt_str = ",".join(str(x) for x in gt_chain)
        pred_str = ",".join(str(x) for x in pred_chain)

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
                f"sample/{i}/target_product": target_product,
                f"sample/{i}/ground_truth_chain_length": len(gt_chain),
                f"sample/{i}/ground_truth_full_length": len(full_gt_chain),
                f"sample/{i}/predicted_chain_length": len(pred_chain),
                f"sample/{i}/support_indices_in_context": support_in_context,
                f"sample/{i}/support_indices_full_count": len(support_indices),
                f"sample/{i}/final_total_input_tokens": final_input_tokens,
                f"sample/{i}/final_total_output_tokens": final_output_tokens,
                f"sample/{i}/final_total_tokens": final_total_tokens,
                f"sample/{i}/iterations": len(iteration_metrics),
                f"sample/{i}/response_raw": response,
                f"sample/{i}/response_parsed_chain": pred_str,
                f"sample/{i}/ground_truth_chain": gt_str,
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/prefix_match_length": prefix_len,
                f"sample/{i}/prefix_match_ratio": prefix_ratio,
                f"sample/{i}/position_accuracy": pos_acc,
                f"sample/{i}/lcs_length": lcs_len,
                f"sample/{i}/lcs_ratio": lcs_ratio,
                f"sample/{i}/normalized_edit_distance": norm_edit_distance,
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
                "running_macro_prefix_match_ratio": macro_prefix_ratio / samples_run,
                "running_macro_position_accuracy": macro_position_accuracy / samples_run,
                "running_macro_lcs_ratio": macro_lcs_ratio / samples_run,
                "running_macro_normalized_edit_distance": macro_norm_edit_distance / samples_run,
            }
        )

    total = samples_run
    exact_match_accuracy = (exact_match_count / total) if total else 0.0
    macro_precision = (macro_precision / total) if total else 0.0
    macro_recall = (macro_recall / total) if total else 0.0
    macro_f1 = (macro_f1 / total) if total else 0.0
    macro_prefix_ratio = (macro_prefix_ratio / total) if total else 0.0
    macro_position_accuracy = (macro_position_accuracy / total) if total else 0.0
    macro_lcs_ratio = (macro_lcs_ratio / total) if total else 0.0
    macro_norm_edit_distance = (macro_norm_edit_distance / total) if total else 0.0

    print(f"\n{'=' * 60}")
    print(f"Exact match: {exact_match_count}/{total}")
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro prefix match ratio: {macro_prefix_ratio:.4f}")
    print(f"Macro position accuracy: {macro_position_accuracy:.4f}")
    print(f"Macro LCS ratio: {macro_lcs_ratio:.4f}")
    print(f"Macro normalized edit distance: {macro_norm_edit_distance:.4f}")

    for target_product in FIXED_TARGET_PRODUCTS:
        gt_chain = HARDCODED_GT_LONGEST_CHAIN[target_product]
        run.summary[f"ground_truth/product_{target_product}/chain"] = ",".join(
            str(x) for x in gt_chain
        )
        run.summary[f"ground_truth/product_{target_product}/length"] = len(gt_chain)

    run.summary["exact_match_correct"] = exact_match_count
    run.summary["total"] = total
    run.summary["exact_match_accuracy"] = exact_match_accuracy
    run.summary["macro_precision"] = macro_precision
    run.summary["macro_recall"] = macro_recall
    run.summary["macro_f1"] = macro_f1
    run.summary["macro_prefix_match_ratio"] = macro_prefix_ratio
    run.summary["macro_position_accuracy"] = macro_position_accuracy
    run.summary["macro_lcs_ratio"] = macro_lcs_ratio
    run.summary["macro_normalized_edit_distance"] = macro_norm_edit_distance
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
