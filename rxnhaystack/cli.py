from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import Counter
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rxnhaystack.control_room import (
    ControlRoomError,
    build_snapshot,
    load_snapshot_directory,
    merge_snapshots,
    publish_snapshot,
    snapshot_state_digest,
    sync_snapshots,
    write_dashboard,
    write_markdown,
    write_snapshot,
)
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
    run.add_argument(
        "--max-run-seconds",
        type=float,
        help=(
            "Terminate a model subprocess that exceeds this wall time, preserving "
            "its failed attempt and allowing other selected jobs to continue."
        ),
    )
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

    control_room = subparsers.add_parser(
        "control-room", help="Publish and view shared, sanitized experiment status."
    )
    control_commands = control_room.add_subparsers(dest="control_command", required=True)
    update = control_commands.add_parser(
        "update", help="Read one local ledger and publish its current status."
    )
    update.add_argument("manifest", type=Path)
    update.add_argument("--source-id", required=True)
    update.add_argument("--machine", required=True)
    update.add_argument("--owner", required=True)
    update.add_argument("--scheduler-job-id")
    update.add_argument("--session-name")
    update.add_argument("--entity", default="liac")
    update.add_argument("--project", default="rxnhaystack-control-room")
    update.add_argument("--output-dir", type=Path)
    update.add_argument("--local-only", action="store_true")
    update.add_argument("--watch-seconds", type=float)
    update.add_argument("--heartbeat-seconds", type=float, default=1800)
    update.add_argument(
        "--secret-file",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Read WANDB_API_KEY from a private mode-0600 file.",
    )

    view = control_commands.add_parser(
        "view", help="Download shared updates and build the read-only dashboard."
    )
    view.add_argument("--entity", default="liac")
    view.add_argument("--project", default="rxnhaystack-control-room")
    view.add_argument("--cache-dir", type=Path, default=Path("artifacts/control-room/shared"))
    view.add_argument("--html", type=Path, default=Path("artifacts/control-room/index.html"))
    view.add_argument("--markdown", type=Path, default=Path("artifacts/control-room/status.md"))
    view.add_argument("--stale-after-hours", type=float, default=2)
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--refresh-seconds", type=float, default=300)
    view.add_argument("--no-sync", action="store_true")
    view.add_argument("--no-serve", action="store_true")
    view.add_argument(
        "--secret-file",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Read WANDB_API_KEY from a private mode-0600 file.",
    )
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
        max_run_seconds=args.max_run_seconds,
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


def command_control_room_update(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else manifest.campaign.project_root / "artifacts" / "control-room" / "local"
    )
    output_path = output_dir.expanduser().resolve() / f"{args.source_id}.json"
    api_key = None if args.local_only else resolve_wandb_key(args.secret_file)
    if args.watch_seconds is not None and args.watch_seconds <= 0:
        raise ManifestError("--watch-seconds must be greater than zero")
    if args.heartbeat_seconds <= 0:
        raise ManifestError("--heartbeat-seconds must be greater than zero")

    previous_state: str | None = None
    last_published = 0.0
    while True:
        snapshot = build_snapshot(
            manifest,
            source_id=args.source_id,
            machine=args.machine,
            owner=args.owner,
            scheduler_job_id=args.scheduler_job_id or os.environ.get("SLURM_JOB_ID"),
            session_name=args.session_name,
        )
        write_snapshot(output_path, snapshot)
        state = snapshot_state_digest(snapshot)
        now = time.monotonic()
        should_publish = not args.local_only and (
            state != previous_state or now - last_published >= args.heartbeat_seconds
        )
        if should_publish:
            url = publish_snapshot(
                output_path,
                api_key=api_key or "",
                entity=args.entity,
                project=args.project,
            )
            print(
                f"Published {args.source_id}: {len(snapshot['observations'])}/"
                f"{len(snapshot['experiment']['expected_runs'])} observed; {url or 'W&B complete'}"
            )
            last_published = now
        elif args.local_only:
            print(f"Wrote local status snapshot: {output_path}")
        previous_state = state
        if args.watch_seconds is None:
            return 0
        try:
            time.sleep(args.watch_seconds)
        except KeyboardInterrupt:
            print("\nControl-room updater stopped; the experiment was not touched")
            return 0


def command_control_room_view(args: argparse.Namespace) -> int:
    cache_dir = args.cache_dir.expanduser().resolve()
    if args.stale_after_hours <= 0:
        raise ManifestError("--stale-after-hours must be greater than zero")
    if not 1 <= args.port <= 65535:
        raise ManifestError("--port must be between 1 and 65535")
    if args.refresh_seconds <= 0:
        raise ManifestError("--refresh-seconds must be greater than zero")
    api_key = None if args.no_sync else resolve_wandb_key(args.secret_file)
    html_path = args.html.expanduser().resolve()
    markdown_path = args.markdown.expanduser().resolve()
    refresh_control_room(
        args,
        api_key=api_key,
        cache_dir=cache_dir,
        html_path=html_path,
        markdown_path=markdown_path,
    )
    print(f"Dashboard: {html_path}")
    print(f"Markdown: {markdown_path}")
    if args.no_serve:
        return 0
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("warning: the dashboard is being exposed beyond this machine", file=sys.stderr)
    handler = partial(SimpleHTTPRequestHandler, directory=str(html_path.parent))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    stop_refresh = threading.Event()
    refresh_thread = None
    if not args.no_sync:
        refresh_thread = threading.Thread(
            target=refresh_control_room_until_stopped,
            kwargs={
                "args": args,
                "api_key": api_key or "",
                "cache_dir": cache_dir,
                "html_path": html_path,
                "markdown_path": markdown_path,
                "stop": stop_refresh,
            },
            name="rxnhaystack-control-room-refresh",
            daemon=True,
        )
        refresh_thread.start()
    print(f"Open http://{args.host}:{args.port}/{html_path.name} (Ctrl-C stops the server)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped")
    finally:
        stop_refresh.set()
        if refresh_thread is not None:
            refresh_thread.join(timeout=2)
        server.server_close()
    return 0


def refresh_control_room(
    args: argparse.Namespace,
    *,
    api_key: str | None,
    cache_dir: Path,
    html_path: Path,
    markdown_path: Path,
) -> None:
    if not args.no_sync:
        paths = sync_snapshots(
            api_key=api_key or "",
            entity=args.entity,
            project=args.project,
            output_dir=cache_dir,
        )
        print(f"Downloaded {len(paths)} current machine snapshots")
    snapshots = load_snapshot_directory(cache_dir)
    merged = merge_snapshots(snapshots, stale_after_seconds=args.stale_after_hours * 3600)
    write_dashboard(html_path, merged)
    write_markdown(markdown_path, merged)


def refresh_control_room_until_stopped(
    *,
    args: argparse.Namespace,
    api_key: str,
    cache_dir: Path,
    html_path: Path,
    markdown_path: Path,
    stop: threading.Event,
) -> None:
    while not stop.wait(args.refresh_seconds):
        try:
            refresh_control_room(
                args,
                api_key=api_key,
                cache_dir=cache_dir,
                html_path=html_path,
                markdown_path=markdown_path,
            )
        except Exception as error:
            print(f"warning: dashboard refresh failed: {error}", file=sys.stderr)


def resolve_wandb_key(specifications: list[str]) -> str:
    supplied = load_secret_specs(specifications)
    unexpected = sorted(set(supplied) - {"WANDB_API_KEY"})
    if unexpected:
        raise ManifestError(
            "Control-room commands accept only WANDB_API_KEY; unexpected secret name(s): "
            + ", ".join(unexpected)
        )
    key = supplied.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY")
    if not key:
        raise ManifestError(
            "Missing WANDB_API_KEY. Export it or use --secret-file WANDB_API_KEY=~/.wandb_api_key"
        )
    return key


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "validate": command_validate,
        "plan": command_plan,
        "run": command_run,
        "status": command_status,
        "control-room": lambda parsed: (
            command_control_room_update(parsed)
            if parsed.control_command == "update"
            else command_control_room_view(parsed)
        ),
    }
    try:
        return commands[args.command](args)
    except (ControlRoomError, DatasetError, LedgerError, ManifestError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
