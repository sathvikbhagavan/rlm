import argparse
import random
import uuid

from task15_ring_chain_graph import (
    MAX_HEAVY_ATOMS,
    MIN_HEAVY_ATOMS,
    PATH_LENGTH,
    build_rlm_question,
    ground_truth_ring_path_in_context,
    parse_response,
    parse_selected_records_from_lines,
    score_prediction,
)
from task15_ring_chain_ground_truth import (
    FIXED_QUESTIONS,
    TASK15_GROUND_TRUTH_DEFINITION,
    TASK15_MIN_SELECTED_GROUND_TRUTH,
    TASK15_TOTAL_REACTIONS,
    build_task15_wandb_sample_log,
    chains_for_context_sampling,
    full_dataset_chain_count,
    full_support_indices_for_question,
    print_task15_run_summary,
    print_task15_sample_context,
    print_task15_sample_metrics,
    print_task15_startup_banner,
    ring_spec_for_question,
    update_task15_run_summary,
)

import wandb
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
CONTEXT_SIZE = int(__import__("os").environ.get("RXNHAYSTACK_CONTEXT_SIZE", "100"))
CONTEXT_PIPELINE_NAME = "random"

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task15",
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
        description="Run RLM task 15 — ring-construction chains from acyclic precursors."
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
    evaluated_questions = [
        question for question in FIXED_QUESTIONS if full_dataset_chain_count(question) > 0
    ]
    if not evaluated_questions:
        raise ValueError("No ring-system questions have non-empty ground truth.")

    print_task15_startup_banner()

    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**instrument_rlm_from_environment(rlm_init_kwargs))
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    run = wandb.init(
        project="RLMs-Task15",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "num_questions": len(evaluated_questions),
            "ring_questions": [q.ring_system for q in evaluated_questions],
            "path_length": PATH_LENGTH,
            "min_selected_ground_truth": TASK15_MIN_SELECTED_GROUND_TRUTH,
            "min_heavy_atoms": MIN_HEAVY_ATOMS,
            "max_heavy_atoms": MAX_HEAVY_ATOMS,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Ring-construction chains from acyclic precursors via RDKit SMARTS.",
            "ground_truth_definition": TASK15_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK15_TOTAL_REACTIONS,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    macro_accuracy = 0.0
    macro_valid_path = 0.0
    macro_objective_length = 0.0
    macro_reaction_f1 = 0.0
    index_match_count = 0
    total_cost_usd = 0.0
    samples_with_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    samples_run = 0

    for i, question in enumerate(evaluated_questions):
        spec = ring_spec_for_question(question)
        full_support_indices = full_support_indices_for_question(question)
        sampling = chains_for_context_sampling(question, context_size)
        support_indices = set(sampling.support_indices)

        context_pipeline = build_context_pipeline(
            name=CONTEXT_PIPELINE_NAME,
            lines=lines,
            rng=random.Random(SEED + i),
            min_selected_ground_truth=max(
                TASK15_MIN_SELECTED_GROUND_TRUTH,
                len(support_indices),
            ),
        )
        sample_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=support_indices,
            query=f"ring_chain_{question.ring_system}",
        )
        context_lines = [line for line in sample_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in context_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        support_in_context = len(support_indices & context_indices)
        context_coverage = len(context_lines) / len(lines) if lines else 0.0

        gt, filters = ground_truth_ring_path_in_context(
            context_lines,
            question.ring_system,
        )
        if gt is None:
            raise ValueError(
                f"No {PATH_LENGTH}-reaction ground-truth path in context for "
                f"ring_system={question.ring_system}"
            )

        prompt_question = build_rlm_question(
            spec=spec,
            context_reaction_count=filters.context_reaction_count,
            molecule_freq_cap=filters.molecule_freq_cap,
        )

        print_task15_sample_context(
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
                "sample_count": len(evaluated_questions),
                "task": "ring_construction_chain",
                "ring_system": question.ring_system,
                "path_length": PATH_LENGTH,
                "gt_chain_count": len(gt.accepted_reaction_indices or (gt.reaction_indices,)),
            },
            tags=["run_rlms", "sample", "task15_RING_CONSTRUCTION_CHAIN"],
        ):
            completion = rlm.completion(
                prompt=sample_context,
                root_prompt=prompt_question,
            )
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        pred_rxns = parse_response(response)
        records = parse_selected_records_from_lines(context_lines, pred_rxns)
        scores = score_prediction(
            pred_rxns=pred_rxns,
            gt=gt,
            records=records,
            filters=filters,
            min_path_reactions=PATH_LENGTH,
        )
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        macro_accuracy += float(scores["is_correct"])
        macro_valid_path += float(scores["valid_path"])
        macro_objective_length += float(scores["objective_length_match"])
        macro_reaction_f1 += float(scores["reaction_f1"])
        index_match_count += int(scores["index_match"])
        samples_run += 1

        print_task15_sample_metrics(
            sample_index=i,
            question=question,
            response=response,
            pred_rxns=pred_rxns,
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
            build_task15_wandb_sample_log(
                sample_index=i,
                question=question,
                gt=gt,
                sampling=sampling,
                full_support_indices=full_support_indices,
                support_indices=support_indices,
                support_in_context=support_in_context,
                filters=filters,
                pred_rxns=pred_rxns,
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
                "running_macro_accuracy": macro_accuracy / samples_run,
                "running_macro_valid_path": macro_valid_path / samples_run,
                "running_macro_objective_length": macro_objective_length / samples_run,
                "running_macro_reaction_f1": macro_reaction_f1 / samples_run,
            }
        )

    total = samples_run
    macro_accuracy = macro_accuracy / total if total else 0.0
    macro_valid_path = macro_valid_path / total if total else 0.0
    macro_objective_length = macro_objective_length / total if total else 0.0
    macro_reaction_f1 = macro_reaction_f1 / total if total else 0.0

    print_task15_run_summary(
        total=total,
        index_match_count=index_match_count,
        macro_accuracy=macro_accuracy,
        macro_valid_path=macro_valid_path,
        macro_objective_length=macro_objective_length,
        macro_reaction_f1=macro_reaction_f1,
    )

    update_task15_run_summary(
        run,
        total=total,
        index_match_count=index_match_count,
        macro_accuracy=macro_accuracy,
        macro_valid_path=macro_valid_path,
        macro_objective_length=macro_objective_length,
        macro_reaction_f1=macro_reaction_f1,
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
