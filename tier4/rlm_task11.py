import argparse
import re
import uuid
import os
import wandb
from rlm import RLM
from rlm.tracing import init_tracing, using_tracing_attributes

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
SEED = 42

NUM_QUESTIONS = 5
MAX_MOLECULE_FREQ = 200

# Fixed evaluation set: 2 starts each for chain lengths 2, 3, and 4.
# All were chosen using find_good_starts.py so GT count is in (10, 20).
FIXED_QUESTIONS: list[tuple[int, int]] = [
    (99, 2),
    (1052, 2),
    (888, 3),
    (1462, 3),
    (216, 4),
    (2306, 4),
]

# Hardcoded ground truth chains for FIXED_QUESTIONS.
# This avoids recomputing chain enumeration at every run.
HARDCODED_GT_CHAINS: dict[tuple[int, int], list[tuple[int, ...]]] = {
    (99, 2): [
        (99, 18128), (99, 22184), (99, 24614), (99, 31268), (99, 44044),
        (99, 44046), (99, 61474), (99, 66157), (99, 79344), (99, 89733),
        (99, 90192), (99, 117904), (99, 135720),
    ],
    (1052, 2): [
        (1052, 1055), (1052, 1058), (1052, 1064), (1052, 1071), (1052, 1073),
        (1052, 1076), (1052, 1079), (1052, 1096), (1052, 1109), (1052, 1141),
        (1052, 1144), (1052, 1149), (1052, 1151), (1052, 1159), (1052, 1170),
    ],
    (888, 3): [
        (888, 889, 919), (888, 889, 927), (888, 889, 1010), (888, 889, 1014),
        (888, 889, 25152), (888, 918, 919), (888, 918, 927), (888, 918, 1010),
        (888, 918, 1014), (888, 918, 25152), (888, 81788, 919), (888, 81788, 927),
        (888, 81788, 1010), (888, 81788, 1014), (888, 81788, 25152),
    ],
    (1462, 3): [
        (1462, 1463, 1464), (1462, 1463, 1469), (1462, 1463, 1531), (1462, 1463, 1532),
        (1462, 1463, 88170), (1462, 1463, 88175), (1462, 1463, 88248), (1462, 1463, 88249),
        (1462, 88169, 1464), (1462, 88169, 1469), (1462, 88169, 1531), (1462, 88169, 1532),
        (1462, 88169, 88170), (1462, 88169, 88175), (1462, 88169, 88248), (1462, 88169, 88249),
    ],
    (216, 4): [
        (216, 217, 218, 219), (216, 217, 218, 69293), (216, 217, 218, 69314),
        (216, 217, 69292, 219), (216, 217, 69292, 69293), (216, 217, 69292, 69314),
        (216, 69291, 218, 219), (216, 69291, 218, 69293), (216, 69291, 218, 69314),
        (216, 69291, 69292, 219), (216, 69291, 69292, 69293), (216, 69291, 69292, 69314),
    ],
    (2306, 4): [
        (2306, 2307, 23069, 23070), (2306, 2307, 99087, 99088), (2306, 22791, 22886, 23104),
        (2306, 22791, 22889, 22911), (2306, 22791, 22889, 23105), (2306, 22791, 22946, 23184),
        (2306, 22791, 22948, 22949), (2306, 23068, 23069, 23070), (2306, 23068, 99087, 99088),
        (2306, 99056, 99070, 99072), (2306, 99056, 99071, 99072),
    ],
}

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def precision_recall_f1(
    predicted: set[tuple[int, ...]], ground_truth: set[tuple[int, ...]]
) -> tuple[float, float, float]:
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_question(start_index: int, chain_length: int) -> str:
    return f"""
    Context: You are given a large set of chemical reactions in SMILES format, separated by newlines.
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

    Molecule identity must be determined by canonical SMILES (use RDKit Chem.CanonSmiles for comparison).
    Do NOT use substructure matching — only exact canonical SMILES equality counts as a match.
    A reaction must NOT appear more than once in the same chain.
    Exclude trivially common molecules that appear in more than {MAX_MOLECULE_FREQ} reactions as products or reactants
    (these are typically solvents, salts, or byproducts and should not count as meaningful synthetic links).

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


# ---------------------------------------------------------------------------
# Tracing / CLI
# ---------------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(description="Run RLM task 11 — synthetic chain evaluation.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for backend (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=NUM_QUESTIONS,
        help=f"Number of starting reactions to evaluate (default: {NUM_QUESTIONS}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_name: str, num_questions: int) -> None:
    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]

    print(f"Loaded {len(lines)} reactions from {DATASET_PATH}")

    if num_questions != len(FIXED_QUESTIONS):
        print(
            f"num_questions={num_questions} ignored; using fixed set of "
            f"{len(FIXED_QUESTIONS)} questions."
        )
    fixed_questions = list(FIXED_QUESTIONS)
    print(
        f"Using {len(fixed_questions)} fixed questions "
        f"(2xL2, 2xL3, 2xL4): {fixed_questions}"
    )

    gt_chains_by_question: list[list[tuple[int, ...]]] = []
    for start_idx, q_chain_length in fixed_questions:
        key = (start_idx, q_chain_length)
        chains = HARDCODED_GT_CHAINS.get(key)
        if chains is None:
            raise ValueError(
                f"Missing hardcoded GT chains for start={start_idx}, "
                f"chain_length={q_chain_length}"
            )
        gt_chains_by_question.append(chains)
        print(
            f"Ground truth [start={start_idx}, chain_length={q_chain_length}] "
            f"(hardcoded): {len(chains)} chains"
        )

    context = "\n".join(lines)

    run = wandb.init(
        project="RLMs-Task11",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "chain_length": "mixed(2,3,4)",
            "num_questions": len(fixed_questions),
            "fixed_questions": fixed_questions,
            "max_molecule_freq": MAX_MOLECULE_FREQ,
            "seed": SEED,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Synthetic chain identification — pairwise reaction analysis.",
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

    for i, ((start_idx, q_chain_length), gt_chains) in enumerate(
        zip(fixed_questions, gt_chains_by_question)
    ):
        question = build_question(start_index=start_idx, chain_length=q_chain_length)
        gt_set = set(gt_chains)
        completion_kwargs = {"prompt": context, "root_prompt": question}

        print(f"\nQuestion {i + 1}/{len(fixed_questions)}: start_idx={start_idx}, "
              f"chain_length={q_chain_length}, "
              f"gt_chains={len(gt_chains)}")

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(fixed_questions),
                "task": "synthetic_chain",
                "start_idx": start_idx,
                "chain_length": q_chain_length,
            },
            tags=["run_rlms", "sample", "task10_SYNTHETIC_CHAIN"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed_chains = parse_chains(response, q_chain_length)
        pred_set = set(parsed_chains)
        precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        is_exact_match = pred_set == gt_set
        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        print(f"Response [start={start_idx}]: {response[:500]}{'…' if len(response) > 500 else ''}")
        print(f"Predicted [start={start_idx}]: {len(parsed_chains)} chains")
        print(f"Ground truth [start={start_idx}]: {len(gt_chains)} chains")
        print(
            f"Metrics [start={start_idx}] -> precision={precision:.4f} "
            f"recall={recall:.4f} f1={f1:.4f} exact_match={is_exact_match}"
        )

        def _fmt_chains(chains: list[tuple[int, ...]], limit: int = 20) -> str:
            shown = [",".join(str(x) for x in c) for c in chains[:limit]]
            suffix = f" … +{len(chains) - limit} more" if len(chains) > limit else ""
            return "; ".join(shown) + suffix

        for metric in iteration_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                    f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                    f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                }
            )

        if iteration_metrics:
            last_metric = iteration_metrics[-1]
            wandb.log(
                {
                    "sample_idx": i,
                    f"sample/{i}/start_idx": start_idx,
                    f"sample/{i}/chain_length": q_chain_length,
                    f"sample/{i}/final_total_input_tokens": last_metric["total_input_tokens"],
                    f"sample/{i}/final_total_output_tokens": last_metric["total_output_tokens"],
                    f"sample/{i}/final_total_tokens": last_metric["total_tokens"],
                    f"sample/{i}/iterations": len(iteration_metrics),
                    f"sample/{i}/response_raw": response,
                    f"sample/{i}/response_parsed_chains": _fmt_chains(parsed_chains),
                    f"sample/{i}/response_parsed_count": len(parsed_chains),
                    f"sample/{i}/ground_truth_chains": _fmt_chains(gt_chains),
                    f"sample/{i}/ground_truth_count": len(gt_chains),
                    f"sample/{i}/precision": precision,
                    f"sample/{i}/recall": recall,
                    f"sample/{i}/f1": f1,
                    f"sample/{i}/is_exact_match": int(is_exact_match),
                    f"sample/{i}/completion_root_prompt": question,
                    f"sample/{i}/completion_prompt_char_count": len(context),
                    **(
                        {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                        if sample_cost_usd is not None
                        else {}
                    ),
                }
            )

    total = len(fixed_questions)
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

    for (start_idx, q_chain_length), gt in zip(fixed_questions, gt_chains_by_question):
        run.summary[f"ground_truth/start_{start_idx}/count"] = len(gt)
        run.summary[f"ground_truth/start_{start_idx}/chain_length"] = q_chain_length

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
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        num_questions=args.num_questions,
    )
