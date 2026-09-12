import argparse
import random
import os
import uuid

import wandb

from rxnhaystack.campaign_metrics import install_campaign_metrics

from rlm import RLM
from rxnhaystack.worker import instrument_rlm_from_environment
from rlm.codeact_helpers import build_context_pipeline, load_lines
from rlm.tracing import init_tracing, using_tracing_attributes

from task14_protecting_group_graph import (
    MAX_HEAVY_ATOMS,
    MIN_HEAVY_ATOMS,
    build_rlm_question,
    ground_truth_pairs_in_context,
    parse_response,
    precision_recall_f1,
)
from task14_protecting_group_ground_truth import (
    FIXED_QUESTIONS,
    TASK14_GROUND_TRUTH_DEFINITION,
    TASK14_MIN_SELECTED_GROUND_TRUTH,
    TASK14_TOTAL_REACTIONS,
    build_task14_wandb_sample_log,
    full_dataset_pair_count,
    full_support_indices_for_question,
    pairs_for_context_sampling,
    print_task14_run_summary,
    print_task14_sample_context,
    print_task14_sample_metrics,
    print_task14_startup_banner,
    update_task14_run_summary,
)

install_campaign_metrics(wandb)

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = __import__("os").environ.get("RXNHAYSTACK_CLEANED_DATASET", __import__("os").path.expanduser("~/datasets/rxnhaystack/reactionSmilesFigShareUSPTO2023_cleaned.txt"))
BACKEND = "openrouter"
MODEL_NAME = __import__("os").environ.get("RXNHAYSTACK_MODEL", "openai/gpt-5-mini")
ENABLE_TRACING = True
SEED = int(__import__("os").environ.get("RXNHAYSTACK_SEED", "42"))
CONTEXT_SIZE = int(__import__("os").environ.get("RXNHAYSTACK_CONTEXT_SIZE", "100"))
CONTEXT_PIPELINE_NAME = "random"
MAX_PAIRS_PER_GROUP = 0

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
        project_name="RLMs-Task14",
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
        description="Run RLM task 14 — protecting-group install/remove pairs."
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
    parser.add_argument(
        "--max-pairs-per-group",
        type=int,
        default=MAX_PAIRS_PER_GROUP,
        help=(
            "Maximum ground-truth pairs per protecting group; use 0 for all "
            f"pairs in context (default: {MAX_PAIRS_PER_GROUP})."
        ),
    )
    return parser.parse_args()


def main(model_name: str, context_size: int, max_pairs_per_group: int) -> None:
    if max_pairs_per_group < 0:
        raise ValueError("--max-pairs-per-group must be non-negative.")

    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    evaluated_specs = [
        spec for spec in FIXED_QUESTIONS if full_dataset_pair_count(spec) > 0
    ]
    if not evaluated_specs:
        raise ValueError("No protecting-group questions have non-empty ground truth.")

    print_task14_startup_banner(max_pairs_per_group=max_pairs_per_group)

    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**instrument_rlm_from_environment(rlm_init_kwargs))
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    run = wandb.init(
        project="RLMs-Task14",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "num_questions": len(evaluated_specs),
            "protecting_groups": [spec.label for spec in evaluated_specs],
            "max_pairs_per_group": max_pairs_per_group,
            "min_selected_ground_truth": TASK14_MIN_SELECTED_GROUND_TRUTH,
            "min_heavy_atoms": MIN_HEAVY_ATOMS,
            "max_heavy_atoms": MAX_HEAVY_ATOMS,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Protecting-group install/remove pair discovery via RDKit SMARTS.",
            "ground_truth_definition": TASK14_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK14_TOTAL_REACTIONS,
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

    for i, spec in enumerate(evaluated_specs):
        full_support_indices = full_support_indices_for_question(spec)
        sampling = pairs_for_context_sampling(spec, context_size)
        support_indices = set(sampling.support_indices)

        context_pipeline = build_context_pipeline(
            name=CONTEXT_PIPELINE_NAME,
            lines=lines,
            rng=random.Random(SEED + i),
            min_selected_ground_truth=TASK14_MIN_SELECTED_GROUND_TRUTH,
        )
        sample_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=support_indices,
            query=f"pg_pairs_{spec.label}",
        )
        context_lines = [line for line in sample_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in context_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        support_in_context = len(support_indices & context_indices)
        context_coverage = len(context_lines) / len(lines) if lines else 0.0

        gt_pairs = ground_truth_pairs_in_context(
            context_lines,
            spec.label,
            max_pairs_per_group=max_pairs_per_group,
        )
        if not gt_pairs:
            raise ValueError(f"No ground-truth pairs in context for pg_label={spec.label}")

        gt_set = {(pair.install_index, pair.remove_index) for pair in gt_pairs}
        prompt_question = build_rlm_question(spec=spec, max_pairs=max_pairs_per_group)

        print_task14_sample_context(
            sample_index=i,
            spec=spec,
            gt_pairs=gt_pairs,
            sampling=sampling,
            full_support_indices=full_support_indices,
            support_indices=support_indices,
            support_in_context=support_in_context,
            context_size=context_size,
            context_line_count=len(context_lines),
            context_coverage=context_coverage,
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(evaluated_specs),
                "task": "protecting_group_pairs",
                "pg_label": spec.label,
                "functional_group": spec.functional_group,
                "gt_pair_count": len(gt_pairs),
            },
            tags=["run_rlms", "sample", "task14_PROTECTING_GROUP_PAIRS"],
        ):
            completion = rlm.completion(
                prompt=sample_context,
                root_prompt=prompt_question,
            )
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        predicted = parse_response(response)
        precision, recall, f1 = precision_recall_f1(predicted=predicted, ground_truth=gt_set)
        is_exact_match = predicted == gt_set
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        samples_run += 1

        print_task14_sample_metrics(
            sample_index=i,
            spec=spec,
            response=response,
            predicted=predicted,
            gt_set=gt_set,
            precision=precision,
            recall=recall,
            f1=f1,
            exact_set_match=is_exact_match,
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
            build_task14_wandb_sample_log(
                sample_index=i,
                spec=spec,
                gt_pairs=gt_pairs,
                sampling=sampling,
                full_support_indices=full_support_indices,
                support_indices=support_indices,
                support_in_context=support_in_context,
                predicted=predicted,
                precision=precision,
                recall=recall,
                f1=f1,
                exact_set_match=float(is_exact_match),
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
                "running_exact_set_match_accuracy": exact_match_count / samples_run,
                "running_macro_precision": macro_precision / samples_run,
                "running_macro_recall": macro_recall / samples_run,
                "running_macro_f1": macro_f1 / samples_run,
            }
        )

    total = samples_run
    macro_precision = macro_precision / total if total else 0.0
    macro_recall = macro_recall / total if total else 0.0
    macro_f1 = macro_f1 / total if total else 0.0

    print_task14_run_summary(
        total=total,
        exact_match_count=exact_match_count,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
    )

    update_task14_run_summary(
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
        max_pairs_per_group=args.max_pairs_per_group,
    )
