from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from rxnhaystack.dataset import DatasetError
from rxnhaystack.launcher import run_selected, select_runs
from rxnhaystack.ledger import LedgerError, RunLedger
from rxnhaystack.manifest import ManifestError, load_manifest
from rxnhaystack.runtime import load_secret_specs, perform_preflight, resolve_required_secrets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rxnhaystack", description="Reproducible RxnHaystack experiment campaigns."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a campaign manifest.")
    validate.add_argument("manifest", type=Path)

    plan = subparsers.add_parser("plan", help="Show expanded runs and perform preflight checks.")
    add_manifest_and_data_arguments(plan)
    plan.add_argument("--select", action="append", default=[], metavar="GLOB")

    run = subparsers.add_parser("run", help="Launch pending runs and record their artifacts.")
    add_manifest_and_data_arguments(run)
    run.add_argument("--select", action="append", default=[], metavar="GLOB")
    run.add_argument("--max-parallel", type=int, default=1)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument(
        "--recover-running",
        action="store_true",
        help="Mark interrupted running attempts failed and create new attempts.",
    )
    run.add_argument(
        "--secret-file",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Inject a mode-0600 secret without recording its path or value.",
    )

    status = subparsers.add_parser("status", help="Summarize a campaign ledger.")
    status.add_argument("manifest", type=Path)
    return parser


def add_manifest_and_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--cleaned-path", type=Path)


def command_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    print(
        f"Valid: {manifest.campaign.name}; {len(manifest.runs)} expanded runs; "
        f"estimated CHF {manifest.estimated_cost_chf:.2f}/{manifest.campaign.budget_chf:.2f}"
    )
    return 0


def command_plan(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    selected = select_runs(manifest.runs, args.select)
    preflight = perform_preflight(
        manifest,
        data_dir=args.data_dir,
        raw_path=args.raw_path,
        cleaned_path=args.cleaned_path,
    )
    print(
        f"Campaign {manifest.campaign.name}: {len(selected)}/{len(manifest.runs)} runs selected; "
        f"estimated CHF {sum(run.estimated_cost_chf for run in selected):.2f}; "
        f"git {preflight.git.commit[:12]}; "
        f"parallel RAM budget={manifest.campaign.max_parallel_memory_mib or 'unlimited'} MiB"
    )
    if preflight.dataset is not None:
        print(
            "Dataset raw: "
            f"rows={preflight.dataset.raw_lines}; sha256={preflight.dataset.raw_sha256}"
        )
        print(
            "Dataset cleaned: "
            f"rows={preflight.dataset.cleaned_lines}; "
            f"sha256={preflight.dataset.cleaned_sha256}"
        )
    for run in selected:
        print(
            f"{run.run_id}\t{run.task}\t{run.condition}\t{run.method}\t"
            f"{run.model}\tcorpus={run.corpus_size}\tCHF={run.estimated_cost_chf:.2f}"
            f"\tRAM reserve={run.memory_reservation_mib or 'unset'} MiB"
            f"\tRAM limit={run.memory_limit_mib or 'unset'} MiB"
            f"\tquestions={run.question_parallelism}x"
        )
    return 0


def command_run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    selected = select_runs(manifest.runs, args.select)
    secrets = resolve_required_secrets(manifest, load_secret_specs(args.secret_file))
    preflight = perform_preflight(
        manifest,
        data_dir=args.data_dir,
        raw_path=args.raw_path,
        cleaned_path=args.cleaned_path,
    )
    results = run_selected(
        manifest,
        preflight=preflight,
        selected=selected,
        secrets=secrets,
        max_parallel=args.max_parallel,
        retry_failed=args.retry_failed,
        recover_running=args.recover_running,
    )
    for result in results:
        if result.attempt is not None:
            detail = f"attempt={result.attempt}"
        elif result.status == "skipped":
            detail = "already recorded"
        else:
            detail = "no attempt started"
        print(f"{result.run_id}: {result.status} ({detail})")
        if result.error is not None:
            print(f"  {result.error}")
    return 1 if any(result.status not in {"succeeded", "skipped"} for result in results) else 0


def command_status(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    ledger_path = manifest.campaign.artifact_dir / "ledger.sqlite3"
    if not ledger_path.is_file():
        print(f"No ledger exists for {manifest.campaign.name}: {ledger_path}")
        return 0
    records = RunLedger(ledger_path).list_runs(campaign=manifest.campaign.name)
    counts = Counter(record.status for record in records)
    rendered = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
    print(f"Campaign {manifest.campaign.name}: {rendered or 'no runs'}")
    for record in records:
        print(f"{record.run_id}\t{record.status}\tattempts={record.attempts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "validate": command_validate,
        "plan": command_plan,
        "run": command_run,
        "status": command_status,
    }
    try:
        return commands[args.command](args)
    except (DatasetError, LedgerError, ManifestError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
