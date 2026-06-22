import argparse
import asyncio
import os
import random
import uuid

import wandb
from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter
from task8_hardcoded_ground_truth import (
    TASK8_GROUND_TRUTH_DEFINITION,
    TASK8_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION,
    TASK8_POSITIVE_REACTIONS_BY_KEY,
    TASK8_SKIPPED_REACTIONS,
    TASK8_TOTAL_REACTIONS,
    TASK8_VALID_REACTIONS,
)

from rlm.codeact_helpers import (
    build_context_pipeline,
    extract_response_text,
    extract_usage_metrics,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = False
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"
MIN_SELECTED_GROUND_TRUTH = 5
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 30_000
# os.environ["WANDB_MODE"] = "disabled"

SKIPPED_REACTION_KEYS = frozenset({
    "boc_primary_amine_deprotection",
    "boc_amine_protection_of_secondary_amine",
})

BOC_LABELS: dict[str, str] = {
    # "boc_primary_amine_deprotection": "BOC deprotection of primary amine",
    "boc_secondary_amine_deprotection": "BOC deprotection of secondary amine",
    # "boc_amine_protection_of_secondary_amine": "BOC amine protection of secondary amine",
    "boc_amine_protection_of_primary_amine": "BOC amine protection of primary amine",
}

BOC_DESCRIPTIONS: dict[str, str] = {
    # "boc_primary_amine_deprotection": "Acid-mediated removal of a tert-butyloxycarbonyl (Boc) protecting group from a nitrogen atom, regenerating a primary amine. The reactant contains the full Boc group — a tert-butyl moiety consisting of a quaternary carbon with no hydrogens and four connections bearing three methyl groups, connected through a neutral oxygen with no hydrogens and two connections to a carbonyl carbon with no hydrogens and three connections, double-bonded to an oxygen with no hydrogens and one connection. The carbonyl is bonded to the protected nitrogen which has one hydrogen and is neutral. The entire Boc group is unmapped and fully cleaved off, and the nitrogen gains a hydrogen (going from one to two), regenerating a free primary amine.",
    "boc_secondary_amine_deprotection": "Acid-mediated removal of a tert-butyloxycarbonyl (Boc) protecting group from a nitrogen atom, regenerating a secondary amine. The reactant contains the full Boc group — a tert-butyl moiety consisting of a quaternary carbon with no hydrogens and four connections bearing three methyl groups, connected through a neutral oxygen with no hydrogens and two connections to a carbonyl carbon with no hydrogens and three connections, double-bonded to an oxygen with no hydrogens and one connection. The carbonyl is bonded to the protected nitrogen which has no hydrogens and is neutral. The entire Boc group is unmapped and fully cleaved off, and the nitrogen gains a hydrogen (going from zero to one), regenerating a free secondary amine.",
    # "boc_amine_protection_of_secondary_amine": "Protection of a secondary amine with a tert-butyloxycarbonyl (Boc) group. The reactant side contains a Boc reagent — a tert-butyl moiety (quaternary carbon bearing three methyl groups) connected through an oxygen to a carbonyl, which has a second oxygen acting as the leaving group (unmapped). The secondary amine, bearing one hydrogen on nitrogen, attacks the carbonyl carbon, displacing the leaving group oxygen. In the product, the nitrogen loses its hydrogen (going from H1 to H0) and is now bonded to the Boc carbonyl, forming a carbamate linkage. The full Boc group — tert-butyl, oxygen, and carbonyl — is preserved intact in the product.",
    "boc_amine_protection_of_primary_amine": "Protection of a primary amine with a tert-butyloxycarbonyl (Boc) group. The reactant side contains a Boc reagent — a tert-butyl moiety consisting of a quaternary carbon with no hydrogens and four connections, bearing three methyl groups each with three hydrogens and one connection — connected through a neutral oxygen with no hydrogens and two connections to a carbonyl carbon with no hydrogens and three connections, which is double-bonded to an oxygen with no hydrogens and one connection. A second unmapped oxygen on the carbonyl acts as the leaving group. The primary amine nitrogen, bearing two hydrogens and neutral charge, attacks the carbonyl carbon, displacing the leaving group oxygen. In the product, the nitrogen loses one hydrogen (going from two to one), forming a new bond to the Boc carbonyl, while the full Boc group is preserved intact.",
}


def build_question(reaction_label: str, reaction_description: str) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Return all reaction indices that match this reaction type:
      - {reaction_label}

    Description:
      - {reaction_description}

    Guidance:
    - Examine reactant and product molecules; ignore reagents.
    - Identify reactions that perform the full bond-forming transformation described above, not merely related functional-group changes on unrelated scaffolds.
    - Handle multi-component sides separated by dots (.).
    - Skip malformed reactions.

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If no matching reaction exists, report: -1
    """


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="LLM-Task8",
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
        description="Run LLM task 8 BOC reaction index evaluation."
    )
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
    lines = load_lines(DATASET_PATH)
    reaction_keys = list(TASK8_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION.keys())
    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=random.Random(SEED),
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )
    run_session_id = f"llm-task8-{uuid.uuid4()}"

    for reaction_key in reaction_keys:
        if reaction_key in SKIPPED_REACTION_KEYS:
            continue
        print(
            f"Ground truth [{reaction_key}] "
            f"count={TASK8_POSITIVE_REACTIONS_BY_KEY[reaction_key]} "
            f"valid={TASK8_VALID_REACTIONS} "
            f"skipped={TASK8_SKIPPED_REACTIONS} "
            f"definition={TASK8_GROUND_TRUTH_DEFINITION}"
        )

    llm = OpenRouter(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
    )

    run = wandb.init(
        project="LLM-Task8",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "min_selected_ground_truth": MIN_SELECTED_GROUND_TRUTH,
            "reasoning_effort": REASONING_EFFORT,
            "num_questions": len(reaction_keys),
            "task_description": "Return reaction indices for BOC protection/deprotection subtypes.",
            "ground_truth_definition": TASK8_GROUND_TRUTH_DEFINITION,
            "ground_truth_positive_reactions_by_key": TASK8_POSITIVE_REACTIONS_BY_KEY,
            "ground_truth_total_reactions": TASK8_TOTAL_REACTIONS,
            "ground_truth_valid_reactions": TASK8_VALID_REACTIONS,
            "ground_truth_skipped_reactions": TASK8_SKIPPED_REACTIONS,
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
    total_input_tokens = 0
    total_output_tokens = 0
    samples_run = 0

    for i, reaction_key in enumerate(reaction_keys):
        if reaction_key in SKIPPED_REACTION_KEYS:
            continue
        reaction_label = BOC_LABELS[reaction_key]
        reaction_description = BOC_DESCRIPTIONS[reaction_key]
        question = build_question(
            reaction_label=reaction_label,
            reaction_description=reaction_description,
        )
        full_gt_set = set(TASK8_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION[reaction_key])

        retrieved_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=full_gt_set,
            query=reaction_key,
        )
        retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
        retrieved_indices = {
            int(line.split(" ", 1)[0])
            for line in retrieved_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        ground_truth_in_context_set = full_gt_set & retrieved_indices
        ground_truth_count = len(ground_truth_in_context_set)
        context_has_ground_truth = bool(ground_truth_in_context_set)
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

        print(f"Question {i + 1}/{len(reaction_keys)} task={reaction_key}")
        print(
            f"[CONTEXT] requested_size={context_size} actual_size={len(retrieved_lines)} "
            f"ground_truth_in_context={ground_truth_count}/{len(full_gt_set)} "
            f"coverage={context_coverage:.4f}"
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(reaction_keys),
                "reaction_key": reaction_key,
                "agent": "llm_baseline",
                "ground_truth_definition": TASK8_GROUND_TRUTH_DEFINITION,
            },
            tags=["llm-baseline", "sample", "task8_boc"],
        ):
            response = await llm.achat([ChatMessage(role="user", content=completion_prompt)])

        response_text = extract_response_text(response)
        parsed_indices = parse_indices(response_text)
        pred_set = set(parsed_indices)
        precision, recall, f1 = precision_recall_f1(pred_set, ground_truth_in_context_set)
        predicted_count = len(pred_set)
        count_error = abs(predicted_count - ground_truth_count)
        count_exact = int(predicted_count == ground_truth_count)
        is_exact_match = pred_set == ground_truth_in_context_set

        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        print(f"Predicted [{reaction_key}] count: {predicted_count}")
        print(f"Ground truth [{reaction_key}] count: {ground_truth_count}")
        print(
            f"Metrics [{reaction_key}] -> "
            f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
            f"exact_match={is_exact_match} count_error={count_error} count_exact={count_exact}"
        )

        usage_metrics = extract_usage_metrics(response)
        prompt_tokens = int(usage_metrics.get("prompt_tokens", 0))
        completion_tokens = int(usage_metrics.get("completion_tokens", 0))
        total_tokens = int(usage_metrics.get("total_tokens", 0))
        sample_cost = float(usage_metrics["cost_usd"]) if "cost_usd" in usage_metrics else None
        if total_tokens == 0:
            prompt_tokens = count_tokens(
                [{"role": "user", "content": completion_prompt}],
                model_name,
            )
            completion_tokens = count_tokens(
                [{"role": "assistant", "content": response_text}],
                model_name,
            )
            total_tokens = prompt_tokens + completion_tokens
        total_input_tokens += prompt_tokens
        total_output_tokens += completion_tokens
        if sample_cost is not None:
            total_cost_usd += sample_cost
            samples_with_cost += 1
        samples_run += 1

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
                f"sample/{i}/final_total_input_tokens": prompt_tokens,
                f"sample/{i}/final_total_output_tokens": completion_tokens,
                f"sample/{i}/final_total_tokens": total_tokens,
                f"sample/{i}/iterations": 1,
                f"sample/{i}/is_exact_match": int(is_exact_match),
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/predicted_count": predicted_count,
                f"sample/{i}/ground_truth_count": ground_truth_count,
                f"sample/{i}/ground_truth_full_count": len(full_gt_set),
                f"sample/{i}/count_error": count_error,
                f"sample/{i}/count_exact": count_exact,
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_size": context_size,
                f"sample/{i}/context_coverage": context_coverage,
                f"sample/{i}/context_has_ground_truth": int(context_has_ground_truth),
                **({f"sample/{i}/final_total_cost_usd": sample_cost} if sample_cost is not None else {}),
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
    run.summary["avg_total_input_tokens_per_sample"] = (
        total_input_tokens / total if total else 0.0
    )
    run.summary["avg_total_output_tokens_per_sample"] = (
        total_output_tokens / total if total else 0.0
    )
    run.summary["ground_truth/total_reactions"] = TASK8_TOTAL_REACTIONS
    run.summary["ground_truth/valid_reactions"] = TASK8_VALID_REACTIONS
    run.summary["ground_truth/skipped_reactions"] = TASK8_SKIPPED_REACTIONS
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost

    for reaction_key in reaction_keys:
        run.summary[f"full_ground_truth/{reaction_key}/count"] = (
            TASK8_POSITIVE_REACTIONS_BY_KEY[reaction_key]
        )

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            model_name=args.model_name,
            context_size=args.context_size,
        )
    )
