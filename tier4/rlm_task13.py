import argparse
import os
import random
import uuid

import wandb
from oracle_predicates import (
    ORACLE_PREDICATE_SHA256,
    ORACLE_PREDICATE_VERSION,
    task13_oracle_guidance,
)
from task13_fg_chain_graph import (
    MAX_HEAVY_ATOMS,
    MAX_MOLECULE_FREQ_REFERENCE,
    MIN_HEAVY_ATOMS,
    MIN_LOCAL_MOLECULE_FREQ,
    PATH_LENGTH,
    build_rlm_question,
    ground_truth_fg_path_in_context,
    parse_chains,
    parse_records_from_lines,
    score_chain_predictions,
)
from task13_fg_chain_ground_truth import (
    FIXED_QUESTIONS,
    TASK13_GROUND_TRUTH_DEFINITION,
    TASK13_MIN_SELECTED_GROUND_TRUTH,
    TASK13_TOTAL_REACTIONS,
    build_task13_wandb_sample_log,
    chains_for_context_sampling,
    full_support_indices_for_question,
    print_task13_run_summary,
    print_task13_sample_context,
    print_task13_sample_metrics,
    print_task13_startup_banner,
    update_task13_run_summary,
)

from rlm import RLM
from rlm.codeact_helpers import build_context_pipeline, load_lines
from rlm.tracing import init_tracing, using_tracing_attributes
from rxnhaystack.campaign_metrics import install_campaign_metrics
from rxnhaystack.worker import instrument_rlm_from_environment

install_campaign_metrics(wandb)

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = __import__("os").environ.get(
    "RXNHAYSTACK_CLEANED_DATASET",
    __import__("os").path.expanduser(
        "~/datasets/rxnhaystack/reactionSmilesFigShareUSPTO2023_cleaned.txt"
    ),
)
BACKEND = "openrouter"
MODEL_NAME = __import__("os").environ.get("RXNHAYSTACK_MODEL", "openai/gpt-5-mini")
ENABLE_TRACING = True
SEED = int(__import__("os").environ.get("RXNHAYSTACK_SEED", "42"))
CONTEXT_SIZE = int(__import__("os").environ.get("RXNHAYSTACK_CONTEXT_SIZE", "-1"))
CONTEXT_PIPELINE_NAME = "random"
ORACLE_PREDICATE = os.environ.get("RXNHAYSTACK_ORACLE_PREDICATE") == "1"

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": False,
    "max_depth": 2,
}


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task13",
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
        description="Run RLM task 13 — functional-group transformation chains."
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
    print_task13_startup_banner()

    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**instrument_rlm_from_environment(rlm_init_kwargs))
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    run = wandb.init(
        project="RLMs-Task13",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "num_questions": len(FIXED_QUESTIONS),
            "fixed_fg_queries": [(q.source_fg, q.target_fg) for q in FIXED_QUESTIONS],
            "path_length": PATH_LENGTH,
            "min_selected_ground_truth": TASK13_MIN_SELECTED_GROUND_TRUTH,
            "max_molecule_freq_reference": MAX_MOLECULE_FREQ_REFERENCE,
            "min_local_molecule_freq": MIN_LOCAL_MOLECULE_FREQ,
            "min_heavy_atoms": MIN_HEAVY_ATOMS,
            "max_heavy_atoms": MAX_HEAVY_ATOMS,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Functional-group transformation chains via RDKit SMARTS.",
            "ground_truth_definition": TASK13_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK13_TOTAL_REACTIONS,
            "oracle_predicate": ORACLE_PREDICATE,
            "oracle_predicate_version": ORACLE_PREDICATE_VERSION if ORACLE_PREDICATE else None,
            "oracle_predicate_sha256": ORACLE_PREDICATE_SHA256 if ORACLE_PREDICATE else None,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    exact_match_count = 0
    total_cost_usd = 0.0
    samples_with_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    samples_run = 0

    for i, question in enumerate(FIXED_QUESTIONS):
        full_support_indices = full_support_indices_for_question(question)
        sampling = chains_for_context_sampling(question, context_size)
        support_indices = set(sampling.support_indices)

        context_pipeline = build_context_pipeline(
            name=CONTEXT_PIPELINE_NAME,
            lines=lines,
            rng=random.Random(SEED + i),
            min_selected_ground_truth=TASK13_MIN_SELECTED_GROUND_TRUTH,
        )
        sample_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=support_indices,
            query=f"fg_chain_{question.source_fg}_{question.target_fg}",
        )
        context_lines = [line for line in sample_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in context_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        support_in_context = len(support_indices & context_indices)
        context_coverage = len(context_lines) / len(lines) if lines else 0.0

        gt, filters = ground_truth_fg_path_in_context(
            context_lines,
            question.source_fg,
            question.target_fg,
        )
        if gt is None:
            raise ValueError(
                f"No 7-reaction ground-truth path in context for "
                f"{question.source_fg}->{question.target_fg}"
            )

        records = parse_records_from_lines(context_lines)

        gt_chains = sorted(gt.accepted_reaction_indices or (gt.reaction_indices,))

        prompt_question = build_rlm_question(
            source_fg=question.source_fg,
            target_fg=question.target_fg,
            context_reaction_count=filters.context_reaction_count,
            molecule_freq_cap=filters.molecule_freq_cap,
            oracle_guidance=task13_oracle_guidance() if ORACLE_PREDICATE else None,
        )

        print_task13_sample_context(
            sample_index=i,
            question=question,
            gt=gt,
            sampling=sampling,
            full_support_indices=full_support_indices,
            support_indices=support_indices,
            support_in_context=support_in_context,
            context_size=context_size,
            context_line_count=len(context_lines),
            context_coverage=context_coverage,
            filters=filters,
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(FIXED_QUESTIONS),
                "task": "functional_group_chain",
                "path_length": PATH_LENGTH,
                "source_fg": question.source_fg,
                "target_fg": question.target_fg,
                "gt_chain_count": len(gt_chains),
                "molecule_freq_cap": filters.molecule_freq_cap,
                "frequent_molecule_count": len(filters.frequent_molecules),
                "oracle_predicate": ORACLE_PREDICATE,
                "oracle_predicate_sha256": ORACLE_PREDICATE_SHA256 if ORACLE_PREDICATE else None,
            },
            tags=["run_rlms", "sample", "task13_FUNCTIONAL_GROUP_CHAIN"],
        ):
            completion = rlm.completion(
                prompt=sample_context,
                root_prompt=prompt_question,
            )
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed_chains = parse_chains(response)
        scores = score_chain_predictions(
            pred_chains=parsed_chains,
            gt=gt,
            records=records,
            filters=filters,
        )
        is_exact_match = bool(scores["is_exact_match"])
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        if is_exact_match:
            exact_match_count += 1
        macro_precision += float(scores["precision"])
        macro_recall += float(scores["recall"])
        macro_f1 += float(scores["f1"])
        samples_run += 1

        print_task13_sample_metrics(
            sample_index=i,
            question=question,
            response=response,
            parsed_chains=parsed_chains,
            gt_chains=list(gt_chains),
            scores=scores,
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

        wandb.log(
            build_task13_wandb_sample_log(
                sample_index=i,
                question=question,
                gt=gt,
                sampling=sampling,
                full_support_indices=full_support_indices,
                support_indices=support_indices,
                support_in_context=support_in_context,
                filters=filters,
                parsed_chains=parsed_chains,
                gt_chains=list(gt_chains),
                scores=scores,
                response=response,
                context_size=context_size,
                context_coverage=context_coverage,
                context_line_count=len(context_lines),
                completion_prompt_char_count=len(sample_context),
                final_input_tokens=final_input_tokens,
                final_output_tokens=final_output_tokens,
                final_total_tokens=final_total_tokens,
                iterations=len(iteration_metrics),
                sample_cost_usd=sample_cost_usd,
            )
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
    macro_precision = macro_precision / total if total else 0.0
    macro_recall = macro_recall / total if total else 0.0
    macro_f1 = macro_f1 / total if total else 0.0

    print_task13_run_summary(
        total=total,
        exact_match_count=exact_match_count,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
    )

    update_task13_run_summary(
        run,
        total=total,
        exact_match_count=exact_match_count,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        samples_with_cost=samples_with_cost,
        total_cost_usd=total_cost_usd,
    )
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        context_size=args.context_size,
    )
