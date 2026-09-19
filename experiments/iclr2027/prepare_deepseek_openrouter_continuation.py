from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from rxnhaystack.manifest import PlannedRun, load_manifest

try:
    from .generate_baseline_campaign import QUESTION_COUNTS, quote
except ImportError:  # Direct script execution.
    from generate_baseline_campaign import QUESTION_COUNTS, quote


CAMPAIGN = "iclr2027-deepseek-paid-openrouter-continuation-v1"
OPENROUTER_MODEL = "deepseek/deepseek-v4-flash-0731"


def unfinished_source_ids(ledger: Path) -> set[str]:
    connection = sqlite3.connect(f"file:{ledger.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT run_id FROM runs
            WHERE run_id LIKE 'full-deepseek-v4-flash-%-rlm-%'
              AND status != 'succeeded'
            """
        ).fetchall()
    finally:
        connection.close()
    return {str(row[0]) for row in rows}


def render(source_runs: list[PlannedRun]) -> str:
    lines = [
        "# Generated from the authoritative v34 ledger by prepare_deepseek_openrouter_continuation.py.",
        "# Every entry links to one unfinished SwissAI source cell; successes are never duplicated.",
        "schema_version = 1",
        "",
        "[campaign]",
        f"name = {quote(CAMPAIGN)}",
        'project_root = "../.."',
        f'artifact_dir = "artifacts/{CAMPAIGN}"',
        "budget_chf = 40.0",
        "usd_to_chf = 0.80",
        "require_dataset = true",
        "require_clean_git = true",
        "require_metrics = true",
        "max_parallel_memory_mib = 49152",
        'required_secrets = ["OPENROUTER_API_KEY", "WANDB_API_KEY"]',
        "",
    ]
    for source in sorted(source_runs, key=lambda run: run.run_id):
        tier, task = source.task.split("/")
        task_id = task.removeprefix("task")
        question_count = QUESTION_COUNTS[tier][task_id]
        # CHF 0.02/trajectory is above the measured CHF ~0.0125 trajectory mean.
        estimated_cost = question_count * 0.02
        environment = {
            key: value
            for key, value in source.env.items()
            if not key.startswith("RXNHAYSTACK_SWISSAI_")
        }
        environment.update(
            {
                "RXNHAYSTACK_PROVIDER": "openrouter",
                "RXNHAYSTACK_RLM_OUTPUT_LIMIT": "4096",
                "RXNHAYSTACK_RLM_REASONING_EFFORT": "low",
                "RXNHAYSTACK_SOURCE_RUN_ID": source.run_id,
            }
        )
        environment_text = ", ".join(
            f"{key} = {quote(value)}" for key, value in sorted(environment.items())
        )
        corpus = quote("full") if source.corpus_size == "full" else str(source.corpus_size)
        command = ", ".join(quote(part) for part in source.command)
        lines.extend(
            [
                "[[runs]]",
                f"id = {quote(f'paid-openrouter-{source.run_id}')}",
                f"task = {quote(source.task)}",
                f"condition = {quote(f'paid-openrouter-{source.condition}')}",
                f"method = {quote(source.method)}",
                f"model = {quote(OPENROUTER_MODEL)}",
                f"corpus_size = {corpus}",
                f"seed = {source.seed}",
                "repetitions = 1",
                f"estimated_cost_chf = {estimated_cost:.6f}",
                f"memory_reservation_mib = {source.memory_reservation_mib}",
                f"memory_limit_mib = {source.memory_limit_mib}",
                f"question_parallelism = {source.question_parallelism}",
                f"env = {{ {environment_text} }}",
                f"command = [{command}]",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare unfinished DeepSeek cells for OpenRouter.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = load_manifest(args.source_manifest)
    unfinished = unfinished_source_ids(args.source_ledger)
    by_id = {run.run_id: run for run in source_manifest.runs}
    unknown = unfinished - by_id.keys()
    if unknown:
        raise SystemExit(f"Ledger contains unknown run IDs: {sorted(unknown)}")
    source_runs = [by_id[run_id] for run_id in unfinished]
    if not source_runs:
        raise SystemExit("No unfinished DeepSeek RLM cells remain")
    args.output.write_text(render(source_runs), encoding="utf-8")
    mapping = {
        f"paid-openrouter-{run.run_id}": run.run_id
        for run in sorted(source_runs, key=lambda item: item.run_id)
    }
    args.mapping.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Prepared {len(source_runs)} unfinished cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
