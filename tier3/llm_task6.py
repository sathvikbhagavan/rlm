import argparse
import asyncio
import os
import random
import uuid

import wandb
from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter
from task6_hardcoded_ground_truth import (
    TASK6_GROUND_TRUTH_DEFINITION,
    TASK6_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION,
    TASK6_POSITIVE_REACTIONS_BY_KEY,
    TASK6_SKIPPED_REACTIONS,
    TASK6_TOTAL_REACTIONS,
    TASK6_VALID_REACTIONS,
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
ENABLE_TRACING = True
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"
MIN_SELECTED_GROUND_TRUTH = 1
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 30_000
# os.environ["WANDB_MODE"] = "disabled"

AMIDE_COUPLING_LABELS: dict[str, str] = {
    "acyl_chloride_with_primary_amine": "Acyl chloride with primary amine to amide (Schotten-Baumann)",
    # "acyl_chloride_with_secondary_amine": "Acyl chloride with secondary amine to amide",
    "carboxylic_acid_with_primary_amine": "Carboxylic acid with primary amine to amide",
    "ester_with_primary_amine": "Ester with primary amine to amide",
    "ester_with_secondary_amine": "Ester with secondary amine to amide",
}

AMIDE_COUPLING_DESCRIPTIONS: dict[str, str] = {
    "acyl_chloride_with_primary_amine": "An amide bond formation through a nucleophilic acyl substitution between an acyl chloride and a primary amine. In this reaction, the nitrogen of the primary amine attacks the electrophilic carbonyl carbon of the acyl chloride, which is highly activated due to the electron-withdrawing chlorine substituent. The chloride ion acts as the leaving group and is displaced during the process, departing along with a proton as HCl. The product is an amide featuring a new carbon-nitrogen bond, while the original carbonyl (C=O) double bond is preserved. One of the two hydrogens on the amine nitrogen is consumed in forming the new bond, resulting in a secondary-type nitrogen in the product. The primary amine is restricted to simple amines whose nitrogen is not directly bonded to another nitrogen or oxygen, thereby excluding hydrazines, hydroxylamines, and similar heteroatom-substituted amines.",
    # "acyl_chloride_with_secondary_amine": "An amide bond formation through a nucleophilic acyl substitution between an acyl chloride and a secondary amine. The nitrogen of the secondary amine, which already bears two carbon substituents and one hydrogen, attacks the electrophilic carbonyl carbon of the acyl chloride. The chloride ion is displaced as the leaving group, departing along with a proton as HCl. The product is a tertiary amide in which the nitrogen has lost its only hydrogen and now forms three bonds to carbon-containing groups, while the carbonyl (C=O) double bond remains intact. Because acyl chlorides are highly activated electrophiles, this reaction proceeds readily and is a common method for constructing sterically hindered or N,N-disubstituted amides in synthetic chemistry.",
    "carboxylic_acid_with_primary_amine": "An amide bond formation through a condensation reaction between a carboxylic acid and a primary amine. In this reaction, the nucleophilic nitrogen of the amine attacks the electrophilic carbonyl carbon of the carboxylic acid. The hydroxyl group on the carboxylic acid acts as the leaving group and is displaced during the process, ultimately departing as water. The product is an amide, featuring a new carbon-nitrogen bond while the original carbonyl (C=O) double bond is preserved. One of the two hydrogens on the amine nitrogen is consumed in forming the new bond, leaving a secondary-type nitrogen in the product. This reaction is one of the most fundamental transformations in both organic chemistry and biology, serving as the basis for peptide bond formation during protein synthesis.",
    "ester_with_primary_amine": "An amide bond formation through aminolysis of an ester by a primary amine. In this reaction, the nucleophilic nitrogen of the primary amine attacks the electrophilic carbonyl carbon of the ester, displacing the alkoxy group as an alcohol leaving group. The carbonyl (C=O) double bond is preserved in the product, while a new carbon-nitrogen amide bond is formed. One of the two hydrogens on the amine nitrogen is consumed during bond formation, yielding a secondary-type nitrogen in the resulting amide. The substituent on the carbonyl carbon is restricted to a carbon-based group, confirming this is an organic ester. This transformation is commonly used in synthetic chemistry when milder conditions than acyl chloride chemistry are desired, though it typically requires heating or catalytic activation.",
    "ester_with_secondary_amine": "An amide bond formation through aminolysis of an ester by a secondary amine. The ester consists of a carbon-based substituent bonded to a neutral carbonyl carbon with no hydrogens and three connections, bearing a double-bonded oxygen and a single-bonded ester oxygen. The ester oxygen is neutral, has no hydrogens, two connections, and specifically excludes tert-butyl esters where the oxygen is bonded to a quaternary carbon bearing three methyl groups, to avoid competing Boc-deprotection pathways. The secondary amine nitrogen is neutral, carries one hydrogen and two connections. In the product, the ester oxygen is displaced as an alcohol leaving group, and the nitrogen replaces it — losing its hydrogen (going from one to zero) and gaining a connection (going from two to three), forming a tertiary amide. The carbonyl double bond to oxygen is preserved unchanged throughout the transformation.",
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
        project_name="LLM-Task6",
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
        description="Run LLM task 6 amide acylation index evaluation."
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
    reaction_keys = list(TASK6_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION.keys())
    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=random.Random(SEED),
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )
    run_session_id = f"llm-task6-{uuid.uuid4()}"

    for reaction_key in reaction_keys:
        if reaction_key == "acyl_chloride_with_secondary_amine":
            continue
        print(
            f"Ground truth [{reaction_key}] "
            f"count={TASK6_POSITIVE_REACTIONS_BY_KEY[reaction_key]} "
            f"valid={TASK6_VALID_REACTIONS} "
            f"skipped={TASK6_SKIPPED_REACTIONS} "
            f"definition={TASK6_GROUND_TRUTH_DEFINITION}"
        )

    llm = OpenRouter(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
    )

    run = wandb.init(
        project="LLM-Task6",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "min_selected_ground_truth": MIN_SELECTED_GROUND_TRUTH,
            "reasoning_effort": REASONING_EFFORT,
            "num_questions": len(reaction_keys),
            "task_description": "Return reaction indices for amide-acylation subtypes.",
            "ground_truth_definition": TASK6_GROUND_TRUTH_DEFINITION,
            "ground_truth_positive_reactions_by_key": TASK6_POSITIVE_REACTIONS_BY_KEY,
            "ground_truth_total_reactions": TASK6_TOTAL_REACTIONS,
            "ground_truth_valid_reactions": TASK6_VALID_REACTIONS,
            "ground_truth_skipped_reactions": TASK6_SKIPPED_REACTIONS,
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
        if reaction_key == "acyl_chloride_with_secondary_amine":
            continue
        reaction_label = AMIDE_COUPLING_LABELS[reaction_key]
        reaction_description = AMIDE_COUPLING_DESCRIPTIONS[reaction_key]
        question = build_question(
            reaction_label=reaction_label,
            reaction_description=reaction_description,
        )
        full_gt_set = set(TASK6_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION[reaction_key])

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
                "ground_truth_definition": TASK6_GROUND_TRUTH_DEFINITION,
            },
            tags=["llm-baseline", "sample", "task6_amide_acylation"],
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
    run.summary["ground_truth/total_reactions"] = TASK6_TOTAL_REACTIONS
    run.summary["ground_truth/valid_reactions"] = TASK6_VALID_REACTIONS
    run.summary["ground_truth/skipped_reactions"] = TASK6_SKIPPED_REACTIONS
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost

    for reaction_key in reaction_keys:
        run.summary[f"full_ground_truth/{reaction_key}/count"] = (
            TASK6_POSITIVE_REACTIONS_BY_KEY[reaction_key]
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
