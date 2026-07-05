import argparse
import random
import shutil
import subprocess
import uuid

import wandb
from rlm import RLM
from rlm.codeact_helpers import build_context_pipeline, load_lines
from rlm.tracing import init_tracing, using_tracing_attributes

from task17b_smirks_sequential_graph import (
    build_rlm_question,
    parse_chain_response,
    precision_recall_f1,
)
from task17b_ground_truth import (
    FIXED_QUESTIONS,
    TASK17B_FORCED_CHAIN_COUNT,
    TASK17B_GROUND_TRUTH_DEFINITION,
    TASK17B_TOTAL_REACTIONS,
    build_task17b_wandb_sample_log,
    chains_for_context_sampling,
    full_dataset_chain_count,
    full_support_indices_for_question,
    ground_truth_chains_in_context,
    min_selected_ground_truth_for_spec,
    random_pool_excluded_indices,
    print_task17b_run_summary,
    print_task17b_sample_context,
    print_task17b_sample_metrics,
    print_task17b_startup_banner,
    update_task17b_run_summary,
)

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"
MAX_CHAINS_PER_QUESTION = 0
ENVIRONMENT = "docker"
DOCKER_IMAGE = "rlm-sandbox"
DOCKER_MEMORY_LIMIT = "20g"


def docker_is_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def resolve_environment(requested: str) -> str:
    if requested != "docker":
        return requested
    if docker_is_available():
        return "docker"
    print(
        "WARNING: Docker is unavailable in this session "
        "(permission denied or daemon not running). "
        "Falling back to environment=local."
    )
    return "local"


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task17b",
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
        description="Run RLM task 17b — SMIRKS sequential 2- or 3-reaction chains."
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
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
        "--max-chains-per-question",
        type=int,
        default=MAX_CHAINS_PER_QUESTION,
        help=(
            "Maximum ground-truth chains per question; use 0 for all "
            f"chains in context (default: {MAX_CHAINS_PER_QUESTION})."
        ),
    )
    parser.add_argument(
        "--environment",
        choices=["local", "docker"],
        default=ENVIRONMENT,
        help=(
            "RLM code-execution environment (default: docker). "
            "Use local when Docker is unavailable."
        ),
    )
    return parser.parse_args()


def build_rlm_init_kwargs(*, model_name: str, environment: str) -> dict:
    kwargs = {
        "backend": BACKEND,
        "backend_kwargs": {"model_name": model_name},
        "verbose": False,
        "max_depth": 2,
        "environment": environment,
    }
    if environment == "docker":
        kwargs["environment_kwargs"] = {
            "image": DOCKER_IMAGE,
            "memory_limit": DOCKER_MEMORY_LIMIT,
            "bootstrap_packages": False,
        }
    return kwargs


def parse_predictions(response: str, chain_length: int) -> list[tuple[int, ...]]:
    return [
        chain
        for chain in parse_chain_response(response, chain_length=chain_length)
        if len(chain) == chain_length
    ]


def main(
    model_name: str,
    context_size: int,
    max_chains_per_question: int,
    environment: str,
) -> None:
    if max_chains_per_question < 0:
        raise ValueError("--max-chains-per-question must be non-negative.")

    environment = resolve_environment(environment)
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    evaluated_specs = [
        spec for spec in FIXED_QUESTIONS if full_dataset_chain_count(spec) > 0
    ]
    if not evaluated_specs:
        raise ValueError("No task17b questions have non-empty ground truth.")

    print_task17b_startup_banner(max_chains_per_question=max_chains_per_question)

    rlm_init_kwargs = build_rlm_init_kwargs(model_name=model_name, environment=environment)
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    run = wandb.init(
        project="RLMs-Task17b",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "num_questions": len(evaluated_specs),
            "question_ids": [spec.question_id for spec in evaluated_specs],
            "chain_lengths": [spec.chain_length for spec in evaluated_specs],
            "forced_chain_count": TASK17B_FORCED_CHAIN_COUNT,
            "max_chains_per_question": max_chains_per_question,
            "environment": environment,
            "docker_image": DOCKER_IMAGE if environment == "docker" else None,
            "docker_memory_limit": DOCKER_MEMORY_LIMIT if environment == "docker" else None,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "SMIRKS sequential 2- or 3-reaction chain discovery.",
            "ground_truth_definition": TASK17B_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK17B_TOTAL_REACTIONS,
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
        sampling = chains_for_context_sampling(spec, context_size)
        support_indices = set(sampling.support_indices)
        sampling_excluded = random_pool_excluded_indices(
            spec, support_indices, context_size=context_size
        )
        min_selected = min_selected_ground_truth_for_spec(spec)

        context_pipeline = build_context_pipeline(
            name=CONTEXT_PIPELINE_NAME,
            lines=lines,
            rng=random.Random(SEED + i),
            min_selected_ground_truth=min_selected,
        )
        sample_context = context_pipeline.build_context(
            context_size=context_size,
            correct_indices=support_indices,
            query=f"task17b_{spec.question_id}",
            excluded_indices=sampling_excluded,
        )
        context_lines = [line for line in sample_context.splitlines() if line.strip()]
        context_indices = {
            int(line.split(" ", 1)[0])
            for line in context_lines
            if " " in line and line.split(" ", 1)[0].isdigit()
        }
        support_in_context = len(support_indices & context_indices)
        context_coverage = len(context_lines) / len(lines) if lines else 0.0

        gt_chains = ground_truth_chains_in_context(
            context_lines,
            spec,
            max_chains_per_question=max_chains_per_question,
        )
        if not gt_chains:
            raise ValueError(
                f"No ground-truth chains in context for question_id={spec.question_id}"
            )

        gt_set = {tuple(chain) for chain in gt_chains}
        prompt_question = build_rlm_question(spec)

        print_task17b_sample_context(
            sample_index=i,
            spec=spec,
            gt_chains=gt_chains,
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
                "task": "smirks_sequential_chains_17b",
                "question_id": spec.question_id,
                "label": spec.label,
                "chain_length": spec.chain_length,
                "gt_chain_count": len(gt_chains),
            },
            tags=["run_rlms", "sample", "task17b_SMIRKS_SEQUENTIAL_CHAINS"],
        ):
            completion = rlm.completion(
                prompt=sample_context,
                root_prompt=prompt_question,
            )
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        predicted_list = parse_predictions(response, spec.chain_length)
        predicted = {tuple(chain) for chain in predicted_list}
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

        print_task17b_sample_metrics(
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
            build_task17b_wandb_sample_log(
                sample_index=i,
                spec=spec,
                gt_chains=gt_chains,
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

    print_task17b_run_summary(
        total=total,
        exact_match_count=exact_match_count,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
    )

    update_task17b_run_summary(
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
        max_chains_per_question=args.max_chains_per_question,
        environment=args.environment,
    )
