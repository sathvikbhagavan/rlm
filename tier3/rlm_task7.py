import argparse
import os
import random
import uuid

import wandb
from rlm import RLM
from rlm.codeact_helpers import (
    build_context_pipeline,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from task7_hardcoded_ground_truth import (
    TASK7_GROUND_TRUTH_DEFINITION,
    TASK7_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION,
    TASK7_POSITIVE_REACTIONS_BY_KEY,
    TASK7_SKIPPED_REACTIONS,
    TASK7_TO_FG_SMIRKS,
    TASK7_TOTAL_REACTIONS,
    TASK7_VALID_REACTIONS,
)

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"
MIN_SELECTED_GROUND_TRUTH = 5

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}

SKIPPED_REACTION_KEYS = frozenset({"grignard_aldehyde_to_secondary_alcohol"})

TO_FG_LABELS: dict[str, str] = {
    "grignard_ketone_to_tertiary_alcohol": "Grignard from ketone to alcohol",
    # "grignard_aldehyde_to_secondary_alcohol": "Grignard from aldehyde to secondary alcohol",
    "nitrile_to_amine": "Reduction of nitrile to amine",
    "nitro_groups_to_amines": "Reduction of nitro groups to amines",
    "alcohol_to_azide": "Alcohol to azide",
    "alcohol_to_carboxylic_acid": "Oxidation of alcohol to carboxylic acid",
}

TO_FG_DESCRIPTIONS: dict[str, str] = {
    "grignard_ketone_to_tertiary_alcohol": "A Grignard reaction in which an organomagnesium halide adds to a ketone to form a tertiary alcohol. The carbon nucleophile of the Grignard reagent, bonded to magnesium which in turn bears a halide (bromide, iodide, or chloride), attacks the electrophilic carbonyl carbon of the ketone. The carbonyl carbon transitions from trigonal planar (three substituents) to tetrahedral (four substituents), gaining a new carbon-carbon bond to the Grignard carbon. The carbonyl oxygen is reduced from a double bond to a single bond, gaining a hydrogen to become a hydroxyl group in the product. Since the ketone carbonyl carbon already bears two carbon substituents, the addition of the Grignard carbon yields a tertiary alcohol with no hydrogens on the central carbon. This reaction is one of the most important carbon-carbon bond-forming reactions in organic synthesis, enabling the construction of complex molecular architectures from simpler precursors.",
    # "grignard_aldehyde_to_secondary_alcohol": "A Grignard reaction in which an organomagnesium halide adds to an aldehyde to form a secondary alcohol. The carbon nucleophile of the Grignard reagent attacks the electrophilic carbonyl carbon of the aldehyde, forming a new carbon-carbon bond. The carbonyl carbon transitions from two substituents to three, and the carbonyl oxygen is reduced from a double bond to a hydroxyl group. Since the aldehyde carbon originally bears one hydrogen and one substituent, the addition of the Grignard carbon yields a secondary alcohol with one hydrogen remaining on the central carbon.",
    "nitrile_to_amine": "Reduction of a nitrile to a primary amine. The reactant contains a carbon-based substituent bonded to a neutral nitrile carbon with no hydrogens and two connections, which is triple-bonded to a nitrogen with no hydrogens and one connection. In the product, the triple bond is fully reduced to a single bond — the carbon gains two hydrogens (going from zero to two) while retaining two connections, and the nitrogen also gains two hydrogens (going from zero to two) while remaining singly connected. The result is a primary amine, with four hydrogen atoms added overall across the carbon and nitrogen.",
    "nitro_groups_to_amines": "Reduction of a nitro group to a primary amine. The reactant contains a nitrogen with no hydrogens, three connections, and a positive formal charge, double-bonded to one oxygen and single-bonded to another oxygen carrying a negative formal charge — the canonical representation of a nitro group (-NO₂). Both oxygens are unmapped and are completely removed during the transformation. In the product, the nitrogen loses all its oxygen substituents, drops from three connections to one, changes from a positive to neutral charge, and gains two hydrogens, becoming a free primary amine.",
    "alcohol_to_azide": "Conversion of an alcohol to an azide via substitution of the hydroxyl group. The reactant contains a carbon-based substituent bonded to a neutral oxygen with one hydrogen and one connection — a simple hydroxyl group. In the product, the hydroxyl is replaced by an azide moiety consisting of three nitrogen atoms in a linear arrangement: the first nitrogen has no hydrogens, two connections, and is neutral; the middle nitrogen has no hydrogens, two connections, and carries a positive formal charge; and the terminal nitrogen has no hydrogens, one connection, and carries a negative formal charge. The oxygen is unmapped and fully removed, while the entire azide group is unmapped and newly introduced.",
    "alcohol_to_carboxylic_acid": "Oxidation of a primary alcohol to a carboxylic acid. The reactant contains a neutral carbon with two hydrogens and two connections, bonded to a neutral hydroxyl oxygen with one hydrogen and one connection — a primary alcohol (R-CH₂-OH). In the product, the carbon loses both hydrogens (going from two to zero) and gains an additional connection (going from two to three). The original oxygen loses its hydrogen (going from one to zero) and becomes double-bonded to the carbon, forming a carbonyl. A new hydroxyl oxygen, unmapped in the reactant and freshly introduced, appears single-bonded to the carbon with one hydrogen and one connection. The result is a carboxylic acid (R-C(=O)-OH), representing a four-electron oxidation.",
}


def build_question(reaction_label: str, reaction_description: str) -> str:
    return f"""
    Context: You are given a large string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Task:
    Return all reaction indices that match this reaction type:
    - {reaction_label}

    Description:
    - {reaction_description}

    Guidance:
    - Use RDKit for parsing reactions and programmatic classification.
    - Represent the transformation as a reaction-level pattern (for example, SMIRKS or reaction SMARTS) that encodes reactants and products together.
    - Pattern matching and substructure checks on mapped reaction templates are appropriate ways to decide membership.
    - Ignore reagents (middle field).
    - Handle multi-component sides separated by dots (.).
    - Skip malformed reactions and matching failures.
    - DO NOT assume/simulate output of the code. Wait for the code to get executed and only then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If no matching reaction exists, return -1.
    """


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task7",
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
        description="Run RLM task 7 functional-group transformation index evaluation."
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
    reaction_keys = list(TASK7_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION.keys())

    for reaction_key in reaction_keys:
        if reaction_key in SKIPPED_REACTION_KEYS:
            continue
        print(
            f"Ground truth [{reaction_key}] "
            f"count={TASK7_POSITIVE_REACTIONS_BY_KEY[reaction_key]} "
            f"valid={TASK7_VALID_REACTIONS} "
            f"skipped={TASK7_SKIPPED_REACTIONS} "
            f"definition={TASK7_GROUND_TRUTH_DEFINITION}"
        )

    context_pipeline = build_context_pipeline(
        name=CONTEXT_PIPELINE_NAME,
        lines=lines,
        rng=random.Random(SEED),
        min_selected_ground_truth=MIN_SELECTED_GROUND_TRUTH,
    )

    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    run = wandb.init(
        project="RLMs-Task7",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "min_selected_ground_truth": MIN_SELECTED_GROUND_TRUTH,
            "num_questions": len(reaction_keys),
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Return reaction indices for functional-group transformation subtypes.",
            "ground_truth_definition": TASK7_GROUND_TRUTH_DEFINITION,
            "to_fg_smirks": TASK7_TO_FG_SMIRKS,
            "ground_truth_positive_reactions_by_key": TASK7_POSITIVE_REACTIONS_BY_KEY,
            "ground_truth_total_reactions": TASK7_TOTAL_REACTIONS,
            "ground_truth_valid_reactions": TASK7_VALID_REACTIONS,
            "ground_truth_skipped_reactions": TASK7_SKIPPED_REACTIONS,
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
        reaction_label = TO_FG_LABELS[reaction_key]
        reaction_description = TO_FG_DESCRIPTIONS[reaction_key]
        reaction_smirks = TASK7_TO_FG_SMIRKS[reaction_key]
        question = build_question(
            reaction_label=reaction_label,
            reaction_description=reaction_description,
        )
        full_gt_set = set(TASK7_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION[reaction_key])

        sample_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=full_gt_set,
            query=reaction_key,
        )
        context_lines = [line for line in sample_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in context_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        ground_truth_in_context_set = full_gt_set & context_indices
        ground_truth_count = len(ground_truth_in_context_set)
        context_coverage = len(context_lines) / len(lines) if lines else 0.0

        print(f"Question {i + 1}/{len(reaction_keys)} task={reaction_key}")
        print(
            f"[CONTEXT] requested_size={context_size} actual_size={len(context_lines)} "
            f"ground_truth_in_context={ground_truth_count}/{len(full_gt_set)} "
            f"coverage={context_coverage:.4f}"
        )

        completion_kwargs = {"prompt": sample_context, "root_prompt": question}

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(reaction_keys),
                "task": "to_fg_transformation",
                "reaction_key": reaction_key,
                "ground_truth_definition": TASK7_GROUND_TRUTH_DEFINITION,
            },
            tags=["run_rlms", "sample", "task7_to_fg"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed_indices = parse_indices(response)
        pred_set = set(parsed_indices)
        precision, recall, f1 = precision_recall_f1(pred_set, ground_truth_in_context_set)
        predicted_count = len(pred_set)
        count_error = abs(predicted_count - ground_truth_count)
        count_exact = int(predicted_count == ground_truth_count)
        sample_cost_usd = completion.usage_summary.total_cost
        is_exact_match = pred_set == ground_truth_in_context_set

        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        print(f"Predicted [{reaction_key}] count: {predicted_count}")
        print(f"Ground truth [{reaction_key}] count: {ground_truth_count}")
        print(
            f"Metrics [{reaction_key}] -> "
            f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
            f"exact_match={is_exact_match} count_error={count_error} count_exact={count_exact}"
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
                f"sample/{i}/reaction_key": reaction_key,
                f"sample/{i}/reaction_smirks": reaction_smirks,
                f"sample/{i}/final_total_input_tokens": final_input_tokens,
                f"sample/{i}/final_total_output_tokens": final_output_tokens,
                f"sample/{i}/final_total_tokens": final_total_tokens,
                f"sample/{i}/iterations": len(iteration_metrics),
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/is_exact_match": int(is_exact_match),
                f"sample/{i}/predicted_count": predicted_count,
                f"sample/{i}/ground_truth_count": ground_truth_count,
                f"sample/{i}/ground_truth_full_count": len(full_gt_set),
                f"sample/{i}/count_error": count_error,
                f"sample/{i}/count_exact": count_exact,
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

    print(f"Exact match: {exact_match_count}/{total}")
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    for reaction_key in reaction_keys:
        run.summary[f"full_ground_truth/{reaction_key}/count"] = (
            TASK7_POSITIVE_REACTIONS_BY_KEY[reaction_key]
        )

    run.summary["exact_match_correct"] = exact_match_count
    run.summary["total"] = total
    run.summary["exact_match_accuracy"] = exact_match_accuracy
    run.summary["macro_precision"] = macro_precision
    run.summary["macro_recall"] = macro_recall
    run.summary["macro_f1"] = macro_f1
    run.summary["ground_truth/total_reactions"] = TASK7_TOTAL_REACTIONS
    run.summary["ground_truth/valid_reactions"] = TASK7_VALID_REACTIONS
    run.summary["ground_truth/skipped_reactions"] = TASK7_SKIPPED_REACTIONS
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
