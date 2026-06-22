import argparse
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
from task9_hardcoded_ground_truth import (
    TASK9_GROUND_TRUTH_DEFINITION,
    TASK9_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION,
    TASK9_NAMED_REACTIONS_SMIRKS,
    TASK9_POSITIVE_REACTIONS_BY_KEY,
    TASK9_SKIPPED_REACTIONS,
    TASK9_TOTAL_REACTIONS,
    TASK9_VALID_REACTIONS,
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

SKIPPED_REACTION_KEYS = frozenset({
    "sonogashira_coupling_terminal_alkyne_with_aryl_halide",
    "wittig_with_phosphonium",
})

NAMED_REACTIONS_LABELS: dict[str, str] = {
    "suzuki_coupling_with_boronic_acids": "Suzuki coupling with boronic acids",
    "mitsunobu_sulfonamide": "Mitsunobu sulfonamide",
    # "sonogashira_coupling_terminal_alkyne_with_aryl_halide": "Sonogashira coupling of terminal alkyne with aryl halide",
    "buchwald_hartwig_n_arylation_primary_amine": "Buchwald-Hartwig Ullmann-Goldberg N-arylation primary amine",
    "stille_reaction_aryl": "Stille reaction aryl",
    # "wittig_with_phosphonium": "Wittig with Phosphonium",
}

NAMED_REACTIONS_DESCRIPTIONS: dict[str, str] = {
    "suzuki_coupling_with_boronic_acids": "A Suzuki cross-coupling reaction in which a boronic acid reacts with an organohalide under palladium catalysis to form a new carbon-carbon bond. The boronic acid partner is restricted to aryl, vinyl, or alkynyl carbons attached to a B(OH)₂ group, while the halide partner carries a chlorine, bromine, or iodide leaving group on an aryl, vinyl, or heteroaryl carbon. In the product, the boron moiety and halide are both lost, and a direct C-C bond forms between the two coupling partners. Both sides of the coupling are restricted to sp2 or sp carbons, consistent with the mechanistic requirements of oxidative addition and transmetalation in the Suzuki catalytic cycle.",
    "mitsunobu_sulfonamide": "A Mitsunobu reaction in which a sulfonamide nitrogen displaces a hydroxyl group on a primary or secondary alcohol, forming a new carbon-nitrogen bond with inversion of stereochemistry. The alcohol carbon is restricted to either a secondary carbon with one hydrogen and two carbon neighbors, or a primary carbon with two hydrogens and one carbon neighbor — excluding tertiary alcohols and methanol. The nitrogen nucleophile is a sulfonamide bearing one hydrogen, bonded to a carbon substituent and a sulfonyl group (S(=O)=O). In the product, the hydroxyl group is lost and a direct C-N bond forms between the alcohol carbon and the sulfonamide nitrogen. This reaction is mediated by a phosphine (typically triphenylphosphine) and a dialkyl azodicarboxylate (DIAD or DEAD), which together activate the alcohol as a leaving group and enable the SN2 displacement.",
    # "sonogashira_coupling_terminal_alkyne_with_aryl_halide": "A Sonogashira cross-coupling reaction in which a terminal alkyne couples with an aryl halide to form an aryl-alkyne (C-C) bond. The aryl halide consists of an aromatic carbon bearing a chlorine, bromine, or iodine leaving group. The terminal alkyne has a substituted carbon with no hydrogens and two connections (one to the substituent, one to the triple bond) and a terminal carbon with one hydrogen and one connection. In the product, the halide is displaced and the aromatic carbon forms a new bond to the substituted alkyne carbon, which retains its two connections and gains no hydrogens, while the terminal alkyne carbon remains unchanged with its hydrogen intact. This reaction is typically catalyzed by a palladium complex with a copper(I) co-catalyst and a base, and is widely used for introducing alkyne functionality onto aromatic rings.",
    "buchwald_hartwig_n_arylation_primary_amine": "A palladium- or copper-catalyzed N-arylation in which a primary amine couples with an aryl halide to form a new aryl carbon-nitrogen bond. The aryl halide consists of a neutral aromatic carbon with no hydrogens and three connections, bearing a fluorine, chlorine, bromine, or iodine leaving group. The primary amine has a neutral nitrogen with two hydrogens and one connection, bonded to a carbon-based substituent. In the product, the halide is displaced and the nitrogen forms a direct bond to the aromatic carbon, losing one hydrogen (going from two to one) and gaining one connection (going from one to two), yielding a secondary arylamine. This transformation encompasses several named reactions including Buchwald-Hartwig amination, Ullmann-Goldberg coupling, and nucleophilic aromatic substitution, depending on the catalyst and conditions employed.",
    "stille_reaction_aryl": "A Stille cross-coupling reaction in which an aryl group is transferred from an organostannane to an organohalide under palladium catalysis, forming a new carbon-carbon bond. The organostannane consists of a tin center with no hydrogens and four connections — three alkyl substituents (methyl or longer chain, with two or three hydrogens on the carbon directly bonded to tin) and one aromatic carbon with no hydrogens and three connections that serves as the transferred group. The coupling partner is a neutral carbon bearing a fluorine, chlorine, bromine, or iodine leaving group. In the product, the tin moiety and halide are both lost, and a direct bond forms between the electrophilic carbon and the aromatic carbon from the stannane. This reaction is valued for its tolerance of diverse functional groups and mild reaction conditions.",
    # "wittig_with_phosphonium": "A Wittig olefination in which a phosphonium ylide reacts with an aldehyde or ketone to form a new carbon-carbon double bond. The carbonyl component has a neutral carbon bonded to a carbon substituent and a double-bonded oxygen. The phosphonium salt consists of a positively charged phosphorus bonded to a methylene carbon with two hydrogens and two connections, carrying one substituent. In the product, the carbonyl oxygen and phosphorus are both lost, and the carbonyl carbon forms a double bond to the ylide carbon, which loses one hydrogen (going from two to one) while retaining two connections. The resulting alkene bridges the two original substituents. This template specifically covers monosubstituted phosphonium ylides and is one of the most widely used methods for constructing alkenes with defined geometry.",
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
        project_name="RLMs-Task9",
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
        description="Run RLM task 9 named reaction index evaluation."
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
    reaction_keys = list(TASK9_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION.keys())

    for reaction_key in reaction_keys:
        if reaction_key in SKIPPED_REACTION_KEYS:
            continue
        print(
            f"Ground truth [{reaction_key}] "
            f"count={TASK9_POSITIVE_REACTIONS_BY_KEY[reaction_key]} "
            f"valid={TASK9_VALID_REACTIONS} "
            f"skipped={TASK9_SKIPPED_REACTIONS} "
            f"definition={TASK9_GROUND_TRUTH_DEFINITION}"
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
        project="RLMs-Task9",
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
            "task_description": "Return reaction indices for named reaction subtypes.",
            "ground_truth_definition": TASK9_GROUND_TRUTH_DEFINITION,
            "named_reactions_smirks": TASK9_NAMED_REACTIONS_SMIRKS,
            "ground_truth_positive_reactions_by_key": TASK9_POSITIVE_REACTIONS_BY_KEY,
            "ground_truth_total_reactions": TASK9_TOTAL_REACTIONS,
            "ground_truth_valid_reactions": TASK9_VALID_REACTIONS,
            "ground_truth_skipped_reactions": TASK9_SKIPPED_REACTIONS,
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
        reaction_label = NAMED_REACTIONS_LABELS[reaction_key]
        reaction_description = NAMED_REACTIONS_DESCRIPTIONS[reaction_key]
        reaction_smirks = TASK9_NAMED_REACTIONS_SMIRKS[reaction_key]
        question = build_question(
            reaction_label=reaction_label,
            reaction_description=reaction_description,
        )
        full_gt_set = set(TASK9_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION[reaction_key])

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
                "task": "named_reactions",
                "reaction_key": reaction_key,
                "ground_truth_definition": TASK9_GROUND_TRUTH_DEFINITION,
            },
            tags=["run_rlms", "sample", "task9_named_reactions"],
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
            TASK9_POSITIVE_REACTIONS_BY_KEY[reaction_key]
        )

    run.summary["exact_match_correct"] = exact_match_count
    run.summary["total"] = total
    run.summary["exact_match_accuracy"] = exact_match_accuracy
    run.summary["macro_precision"] = macro_precision
    run.summary["macro_recall"] = macro_recall
    run.summary["macro_f1"] = macro_f1
    run.summary["ground_truth/total_reactions"] = TASK9_TOTAL_REACTIONS
    run.summary["ground_truth/valid_reactions"] = TASK9_VALID_REACTIONS
    run.summary["ground_truth/skipped_reactions"] = TASK9_SKIPPED_REACTIONS
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
