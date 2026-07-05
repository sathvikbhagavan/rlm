import argparse
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from io import TextIOWrapper
from pathlib import Path

import wandb
from rich.console import Console
from rlm import RLM
from rlm.codeact_helpers import load_lines
from rlm.tracing import init_tracing, using_tracing_attributes

from task16_truncated_synthesis_graph import (
    FULL_CHAIN_LENGTH,
    MAX_HEAVY_ATOMS,
    MIN_HEAVY_ATOMS,
    PREFIX_LENGTH,
    build_rlm_question,
    parse_chains,
    parse_records_from_lines,
    score_chain_predictions,
)
from task16_truncated_synthesis_ground_truth import (
    FIXED_QUESTIONS,
    TASK16_FORCED_PREFIX_COUNT,
    TASK16_GROUND_TRUTH_DEFINITION,
    TASK16_MIN_SELECTED_GROUND_TRUTH,
    TASK16_TOTAL_REACTIONS,
    build_task16_eval_context,
    build_task16_wandb_sample_log,
    full_support_indices_for_question,
    hardcoded_full_chains_for_question,
    print_task16_run_summary,
    print_task16_sample_context,
    print_task16_sample_metrics,
    print_task16_startup_banner,
    target_spec_for_question,
    update_task16_run_summary,
)

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
SEED = 42
CONTEXT_SIZE = 100
CONTEXT_PIPELINE_NAME = "random"

ENVIRONMENT = "docker"
DOCKER_IMAGE = "rlm-sandbox"
DOCKER_MEMORY_LIMIT = "30g"
VERBOSE = True
SCRIPT_DIR = Path(__file__).resolve().parent
VERBOSE_LOG_DIR = SCRIPT_DIR / "logs" / "task16_verbose"


class VerboseLogFile:
    """Send RLM rich verbose output to a file instead of stdout (avoids wandb log bloat)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: TextIOWrapper | None = None

    def attach(self, rlm: RLM) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        rlm.verbose.enabled = True
        rlm.verbose.console = Console(file=self._handle, width=120)

    def write_section(self, text: str) -> None:
        if self._handle is None:
            return
        self._handle.write(text)
        if not text.endswith("\n"):
            self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


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


def build_rlm_init_kwargs(*, model_name: str, environment: str, verbose: bool) -> dict:
    kwargs = {
        "backend": BACKEND,
        "backend_kwargs": {"model_name": model_name},
        "verbose": verbose,
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


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task16",
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
        description="Run RLM task 16 — truncated synthesis route prefixes."
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
        "--environment",
        choices=["local", "docker"],
        default=ENVIRONMENT,
        help=(
            "RLM code-execution environment (default: docker). "
            "Use local when Docker is unavailable (e.g. Cursor integrated terminal)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=VERBOSE,
        help=(
            "Enable RLM verbose output (default: on). Written to --verbose-log file, "
            "not stdout, so wandb is not flooded."
        ),
    )
    parser.add_argument(
        "--verbose-log",
        type=Path,
        default=None,
        help=(
            "Path for RLM verbose log file (default: "
            "logs/task16_verbose/verbose_<timestamp>_<run_id>.log)."
        ),
    )
    return parser.parse_args()


def main(
    model_name: str,
    context_size: int,
    environment: str,
    *,
    verbose: bool,
    verbose_log_path: Path | None,
) -> None:
    environment = resolve_environment(environment)
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    print("Parsing full dataset records (one-time)...")
    full_records = parse_records_from_lines(lines)
    print(f"Parsed {len(full_records)} reactions.")
    print_task16_startup_banner()

    rlm_init_kwargs = build_rlm_init_kwargs(
        model_name=model_name,
        environment=environment,
        verbose=verbose,
    )
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    verbose_log: VerboseLogFile | None = None
    if verbose:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = verbose_log_path or (
            VERBOSE_LOG_DIR / f"verbose_{timestamp}_{run_session_id[-8:]}.log"
        )
        verbose_log = VerboseLogFile(log_path.resolve())
        verbose_log.attach(rlm)
        print(f"RLM verbose output -> {verbose_log.path}  (tail -f to follow)")

    run = wandb.init(
        project="RLMs-Task16",
        settings=wandb.Settings(console="off"),
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "context_pipeline_name": CONTEXT_PIPELINE_NAME,
            "num_questions": len(FIXED_QUESTIONS),
            "target_questions": [q.question_id for q in FIXED_QUESTIONS],
            "prefix_length": PREFIX_LENGTH,
            "full_chain_length": FULL_CHAIN_LENGTH,
            "forced_prefix_count": TASK16_FORCED_PREFIX_COUNT,
            "min_selected_ground_truth": TASK16_MIN_SELECTED_GROUND_TRUTH,
            "min_heavy_atoms": MIN_HEAVY_ATOMS,
            "max_heavy_atoms": MAX_HEAVY_ATOMS,
            "rlm_init_kwargs": rlm_init_kwargs,
            "verbose_log_path": str(verbose_log.path) if verbose_log else None,
            "environment": environment,
            "docker_image": DOCKER_IMAGE if environment == "docker" else None,
            "docker_memory_limit": DOCKER_MEMORY_LIMIT if environment == "docker" else None,
            "task_description": "Truncated synthesis prefixes with withheld final reaction.",
            "ground_truth_definition": TASK16_GROUND_TRUTH_DEFINITION,
            "ground_truth_total_reactions": TASK16_TOTAL_REACTIONS,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    try:
        _run_task16_samples(
            rlm=rlm,
            run=run,
            run_session_id=run_session_id,
            lines=lines,
            full_records=full_records,
            context_size=context_size,
            environment=environment,
            verbose_log=verbose_log,
        )
    finally:
        rlm.close()
        if verbose_log is not None:
            verbose_log.close()

    wandb.finish()


def _run_task16_samples(
    *,
    rlm: RLM,
    run,
    run_session_id: str,
    lines: list[str],
    full_records,
    context_size: int,
    environment: str,
    verbose_log: VerboseLogFile | None,
) -> None:
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
        spec = target_spec_for_question(question)
        full_chains = hardcoded_full_chains_for_question(question.question_id)
        full_support_indices = full_support_indices_for_question(question)
        built = build_task16_eval_context(
            question=question,
            lines=lines,
            context_size=context_size,
            sample_index=i,
            seed=SEED,
            pipeline_name=CONTEXT_PIPELINE_NAME,
            full_records=full_records,
        )
        sampling = built.sampling
        support_indices = set(sampling.support_indices)
        excluded_terminals = sampling.excluded_terminal_indices
        context_lines = built.context_lines
        gt = built.gt
        filters = built.filters
        support_in_context = built.support_in_context
        context_coverage = built.context_coverage

        gt_chains = sorted(gt.accepted_reaction_indices or (gt.reaction_indices,))

        records = built.records

        prompt_question = build_rlm_question(
            spec=spec,
            docker_memory_limit=DOCKER_MEMORY_LIMIT if environment == "docker" else None,
        )

        print_task16_sample_context(
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

        if verbose_log is not None:
            verbose_log.write_section(
                f"\n{'=' * 80}\n"
                f"Sample {i + 1}/{len(FIXED_QUESTIONS)}: {question.question_id}\n"
                f"{'=' * 80}\n"
            )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(FIXED_QUESTIONS),
                "task": "truncated_synthesis",
                "question_id": question.question_id,
                "prefix_length": PREFIX_LENGTH,
                "gt_prefix_count": len(gt_chains),
            },
            tags=["run_rlms", "sample", "task16_TRUNCATED_SYNTHESIS"],
        ):
            completion = rlm.completion(
                prompt="\n".join(context_lines),
                root_prompt=prompt_question,
            )
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed_chains = parse_chains(response)
        scores = score_chain_predictions(
            pred_chains=parsed_chains,
            gt=gt,
            records=records,
            full_chains=full_chains,
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

        print_task16_sample_metrics(
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
            build_task16_wandb_sample_log(
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
                completion_prompt_char_count=len("\n".join(context_lines)),
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

    print_task16_run_summary(
        total=total,
        exact_match_count=exact_match_count,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
    )

    update_task16_run_summary(
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


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        context_size=args.context_size,
        environment=args.environment,
        verbose=args.verbose,
        verbose_log_path=args.verbose_log,
    )
