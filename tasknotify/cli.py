from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
import smtplib
import socket
import sqlite3
import ssl
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("~/.config/task-notify/config.toml")
DEFAULT_STATE_DIR = Path("~/.local/state/task-notify")
DEFAULT_CEILING_PATTERNS = (
    "stopped by the balance guard",
    "cost ceiling",
    "budget exhausted",
    "budget exceeded",
    "key limit exceeded",
)


class NotifyError(RuntimeError):
    """A notification could not be configured, observed, or delivered."""


@dataclass(frozen=True)
class MailConfig:
    recipient: str
    sender: str
    host: str
    port: int
    security: str
    username: str | None
    password_file: Path | None


@dataclass(frozen=True)
class LedgerSnapshot:
    total: int
    pending: int
    running: int
    succeeded: int
    failed: int
    cost_chf: float
    calls: int
    total_tokens: int
    failures: tuple[str, ...]

    @property
    def terminal(self) -> bool:
        return self.pending == 0 and self.running == 0


@dataclass(frozen=True)
class Notification:
    event_id: str
    outcome: str
    title: str
    description: str
    project: str
    execution_machine: str
    chat_title: str
    chat_machine: str
    chat_id: str | None
    started_at: str
    observed_at: str
    elapsed_seconds: float
    ledger: str | None = None
    selectors: tuple[str, ...] = ()
    snapshot: LedgerSnapshot | None = None
    command: str | None = None
    exit_code: int | None = None
    log_file: str | None = None
    scheduler_job_id: str | None = None
    scheduler_state: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def read_mail_config(path: str | Path) -> MailConfig:
    config_path = expand_path(path)
    if not config_path.is_file():
        raise NotifyError(f"Mail configuration not found: {config_path}")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    transport = raw.get("smtp", {})
    required = {
        "recipient": raw.get("recipient"),
        "sender": raw.get("sender"),
        "host": transport.get("host"),
        "port": transport.get("port"),
        "security": transport.get("security"),
    }
    missing = [name for name, value in required.items() if value in (None, "")]
    if missing:
        raise NotifyError(f"Mail configuration is missing: {', '.join(missing)}")
    security = str(required["security"]).lower()
    if security not in {"ssl", "starttls", "none"}:
        raise NotifyError("smtp.security must be ssl, starttls, or none")
    password_file = transport.get("password_file")
    return MailConfig(
        recipient=str(required["recipient"]),
        sender=str(required["sender"]),
        host=str(required["host"]),
        port=int(required["port"]),
        security=security,
        username=str(transport["username"]) if transport.get("username") else None,
        password_file=expand_path(password_file) if password_file else None,
    )


def read_password(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.is_file():
        raise NotifyError(f"SMTP password file not found: {path}")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise NotifyError(f"SMTP password file must not be accessible by group/others: {path}")
    password = path.read_text(encoding="utf-8").strip()
    if not password:
        raise NotifyError(f"SMTP password file is empty: {path}")
    return password


def notification_subject(notification: Notification) -> str:
    labels = {
        "done": "DONE",
        "failed": "FAILED",
        "cost_ceiling": "COST CEILING",
    }
    return (
        f"[{labels[notification.outcome]}] {notification.title} on {notification.execution_machine}"
    )


def notification_body(notification: Notification) -> str:
    lines = [
        f"Outcome: {notification.outcome.replace('_', ' ').upper()}",
        f"Task: {notification.title}",
        f"Description: {notification.description}",
        f"Project: {notification.project}",
        f"Execution machine: {notification.execution_machine}",
        f"Launched by Codex chat: {notification.chat_title}",
        f"Codex chat machine: {notification.chat_machine}",
    ]
    if notification.chat_id:
        lines.append(f"Codex chat/thread ID: {notification.chat_id}")
    lines.extend(
        [
            f"Started: {notification.started_at}",
            f"Observed: {notification.observed_at}",
            f"Elapsed: {notification.elapsed_seconds / 3600:.2f} hours",
        ]
    )
    if notification.command:
        lines.append(f"Command: {notification.command}")
    if notification.exit_code is not None:
        lines.append(f"Exit code: {notification.exit_code}")
    if notification.scheduler_job_id:
        lines.append(f"Scheduler job: {notification.scheduler_job_id}")
    if notification.scheduler_state:
        lines.append(f"Scheduler state: {notification.scheduler_state}")
    if notification.ledger:
        lines.append(f"Ledger: {notification.ledger}")
    if notification.selectors:
        lines.append(f"Selected work: {', '.join(notification.selectors)}")
    if notification.snapshot:
        snapshot = notification.snapshot
        lines.extend(
            [
                "",
                "Ledger summary:",
                f"  total: {snapshot.total}",
                f"  succeeded: {snapshot.succeeded}",
                f"  failed: {snapshot.failed}",
                f"  running: {snapshot.running}",
                f"  pending: {snapshot.pending}",
                f"  recorded cost: CHF {snapshot.cost_chf:.4f}",
                f"  calls: {snapshot.calls}",
                f"  tokens: {snapshot.total_tokens}",
            ]
        )
        if snapshot.failures:
            lines.append("Failure examples:")
            lines.extend(f"  - {failure}" for failure in snapshot.failures)
    if notification.log_file:
        lines.append(f"Log: {notification.log_file}")
    lines.extend(["", "This message was generated by task-notify."])
    return "\n".join(lines)


def send_email(notification: Notification, config_path: str | Path) -> None:
    config = read_mail_config(config_path)
    password = read_password(config.password_file)
    message = EmailMessage()
    message["Subject"] = notification_subject(notification)
    message["From"] = config.sender
    message["To"] = config.recipient
    message.set_content(notification_body(notification))

    if config.security == "ssl":
        with smtplib.SMTP_SSL(
            config.host, config.port, timeout=30, context=ssl.create_default_context()
        ) as client:
            if config.username:
                client.login(config.username, password or "")
            client.send_message(message)
        return
    with smtplib.SMTP(config.host, config.port, timeout=30) as client:
        if config.security == "starttls":
            client.starttls(context=ssl.create_default_context())
        if config.username:
            client.login(config.username, password or "")
        client.send_message(message)


def read_ledger(path: str | Path, selectors: tuple[str, ...]) -> LedgerSnapshot:
    ledger_path = expand_path(path)
    if not ledger_path.is_file():
        raise NotifyError(f"Ledger not found: {ledger_path}")
    try:
        connection = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT run_id, status, metrics_json, error FROM runs ORDER BY run_id"
        ).fetchall()
    except sqlite3.Error as error:
        raise NotifyError(f"Could not read ledger {ledger_path}: {error}") from error
    finally:
        if "connection" in locals():
            connection.close()
    selected = [
        row
        for row in rows
        if not selectors or any(fnmatch.fnmatchcase(str(row["run_id"]), item) for item in selectors)
    ]
    if not selected:
        raise NotifyError(f"No ledger runs matched: {', '.join(selectors) or '*'}")
    counts = {status: 0 for status in ("pending", "running", "succeeded", "failed")}
    cost = 0.0
    calls = 0
    tokens = 0
    failures: list[str] = []
    for row in selected:
        counts[str(row["status"])] += 1
        metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
        cost += float(metrics.get("cost_chf", 0) or 0)
        calls += int(metrics.get("calls", 0) or 0)
        tokens += int(metrics.get("total_tokens", 0) or 0)
        if row["status"] == "failed" and len(failures) < 5:
            detail = str(row["error"] or "no error recorded").replace("\n", " ")
            failures.append(f"{row['run_id']}: {detail[:300]}")
    return LedgerSnapshot(
        total=len(selected),
        pending=counts["pending"],
        running=counts["running"],
        succeeded=counts["succeeded"],
        failed=counts["failed"],
        cost_chf=cost,
        calls=calls,
        total_tokens=tokens,
        failures=tuple(failures),
    )


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def slurm_job_state(job_id: str) -> tuple[bool, str]:
    """Return whether a Slurm allocation is active and its normalized state."""

    try:
        queued = subprocess.run(
            ["squeue", "--noheader", "--jobs", job_id, "--format", "%T"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NotifyError(f"Could not query Slurm job {job_id} with squeue: {error}") from error
    if queued.returncode != 0:
        raise NotifyError(f"squeue failed for job {job_id}: {queued.stderr.strip()}")
    queued_states = [line.strip().upper() for line in queued.stdout.splitlines() if line.strip()]
    if queued_states:
        return True, queued_states[0]

    try:
        accounted = subprocess.run(
            [
                "sacct",
                "--noheader",
                "--allocations",
                "--jobs",
                job_id,
                "--format",
                "State",
                "--parsable2",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NotifyError(f"Could not query Slurm job {job_id} with sacct: {error}") from error
    if accounted.returncode != 0:
        raise NotifyError(f"sacct failed for job {job_id}: {accounted.stderr.strip()}")
    states = [
        line.split("|", 1)[0].strip().upper().split()[0]
        for line in accounted.stdout.splitlines()
        if line.strip().strip("|")
    ]
    return False, states[0] if states else "UNKNOWN"


def log_has_ceiling(path: str | None, patterns: tuple[str, ...]) -> bool:
    if path is None:
        return False
    log_path = expand_path(path)
    if not log_path.is_file():
        return False
    tail_size = min(log_path.stat().st_size, 256_000)
    with log_path.open("rb") as handle:
        handle.seek(-tail_size, os.SEEK_END)
        text = handle.read().decode("utf-8", errors="replace").lower()
    return any(pattern.lower() in text for pattern in patterns)


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"sent_events": [], "pending": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NotifyError(f"Invalid watcher state {path}: {error}") from error
    return value if isinstance(value, dict) else {"sent_events": [], "pending": {}}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def notification_from_dict(value: dict[str, Any]) -> Notification:
    snapshot = value.get("snapshot")
    return Notification(
        **{
            **value,
            "selectors": tuple(value.get("selectors", ())),
            "snapshot": LedgerSnapshot(**snapshot) if snapshot else None,
        }
    )


def common_notification(args: argparse.Namespace, *, outcome: str, event_id: str) -> Notification:
    started = datetime.fromisoformat(args.started_at)
    now = datetime.now(UTC)
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return Notification(
        event_id=event_id,
        outcome=outcome,
        title=args.title,
        description=args.description,
        project=args.project,
        execution_machine=args.machine,
        chat_title=args.chat_title,
        chat_machine=args.chat_machine,
        chat_id=args.chat_id,
        started_at=started.isoformat(),
        observed_at=now.isoformat(),
        elapsed_seconds=max((now - started.astimezone(UTC)).total_seconds(), 0),
        scheduler_job_id=getattr(args, "slurm_job_id", None),
        scheduler_state=getattr(args, "scheduler_state", None),
    )


def deliver_pending(state: dict[str, Any], config_path: str | Path) -> None:
    sent = set(state.setdefault("sent_events", []))
    pending = state.setdefault("pending", {})
    for event_id, raw in list(pending.items()):
        if event_id in sent:
            pending.pop(event_id, None)
            continue
        send_email(notification_from_dict(raw), config_path)
        sent.add(event_id)
        pending.pop(event_id, None)
    state["sent_events"] = sorted(sent)


def watch_ledger(args: argparse.Namespace) -> int:
    state_path = expand_path(args.state_file)
    state = load_state(state_path)
    started_monotonic = time.monotonic()
    terminal_event: str | None = None
    while True:
        snapshot = read_ledger(args.ledger, tuple(args.select))
        scheduler_state: str | None = None
        if args.slurm_job_id:
            launcher_alive, scheduler_state = slurm_job_state(args.slurm_job_id)
        else:
            launcher_alive = process_alive(args.launcher_pid)
        args.scheduler_state = scheduler_state
        ceiling = (
            args.cost_ceiling_chf is not None and snapshot.cost_chf >= args.cost_ceiling_chf
        ) or log_has_ceiling(args.log_file, tuple(args.ceiling_pattern))

        outcomes: list[str] = []
        if ceiling:
            outcomes.append("cost_ceiling")
        if snapshot.terminal:
            outcomes.append("failed" if snapshot.failed else "done")
            terminal_event = outcomes[-1]
        elif not launcher_alive:
            outcomes.append("cost_ceiling" if ceiling else "failed")
            terminal_event = outcomes[-1]

        sent = set(state.setdefault("sent_events", []))
        pending = state.setdefault("pending", {})
        for outcome in outcomes:
            event_id = f"{args.watch_id}:{outcome}"
            if event_id in sent or event_id in pending:
                continue
            base = common_notification(args, outcome=outcome, event_id=event_id)
            notification = Notification(
                **{
                    **asdict(base),
                    "ledger": str(expand_path(args.ledger)),
                    "selectors": tuple(args.select),
                    "snapshot": snapshot,
                    "log_file": str(expand_path(args.log_file)) if args.log_file else None,
                }
            )
            pending[event_id] = asdict(notification)
        save_state(state_path, state)

        try:
            deliver_pending(state, args.config)
        except (NotifyError, OSError, smtplib.SMTPException) as error:
            print(f"task-notify: delivery pending: {error}", file=sys.stderr, flush=True)
        save_state(state_path, state)
        if terminal_event and not state.get("pending"):
            return 1 if terminal_event == "failed" else 0
        if args.timeout_seconds and time.monotonic() - started_monotonic >= args.timeout_seconds:
            raise NotifyError("Watcher timeout reached before a terminal event was delivered")
        time.sleep(args.poll_seconds)


def run_command(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise NotifyError("run requires a command after --")
    started = datetime.now(UTC)
    completed = subprocess.run(command, check=False)
    outcome = "done" if completed.returncode == 0 else "failed"
    notification = common_notification(
        argparse.Namespace(**{**vars(args), "started_at": started.isoformat()}),
        outcome=outcome,
        event_id=f"{args.watch_id}:{outcome}:{int(started.timestamp())}",
    )
    notification = Notification(
        **{
            **asdict(notification),
            "command": shlex.join(command),
            "exit_code": completed.returncode,
        }
    )
    send_email(notification, args.config)
    return completed.returncode


def send_test(args: argparse.Namespace) -> int:
    now = utc_now()
    notification = Notification(
        event_id="test",
        outcome="done",
        title="task-notify delivery test",
        description="A delivery test from the reusable task notification utility.",
        project="task-notify",
        execution_machine=args.machine,
        chat_title=args.chat_title,
        chat_machine=args.chat_machine,
        chat_id=args.chat_id,
        started_at=now,
        observed_at=now,
        elapsed_seconds=0,
    )
    send_email(notification, args.config)
    return 0


def add_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--machine", default=socket.gethostname())
    parser.add_argument("--chat-title", required=True)
    parser.add_argument("--chat-machine", default=socket.gethostname())
    parser.add_argument("--chat-id", default=os.environ.get("CODEX_THREAD_ID"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="task-notify",
        description="Email durable completion, failure, and cost-ceiling notifications.",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    run_parser = subparsers.add_parser("run", help="wrap any command and email its exit status")
    add_context_arguments(run_parser)
    run_parser.add_argument("--watch-id", required=True)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(handler=run_command)

    watch_parser = subparsers.add_parser(
        "watch-ledger", help="watch a SQLite runs ledger until completion or early stop"
    )
    add_context_arguments(watch_parser)
    watch_parser.add_argument("--watch-id", required=True)
    watch_parser.add_argument("--ledger", required=True)
    watch_parser.add_argument("--select", action="append", default=[])
    launcher = watch_parser.add_mutually_exclusive_group()
    launcher.add_argument("--launcher-pid", type=int)
    launcher.add_argument(
        "--slurm-job-id",
        help="Slurm allocation to observe with squeue and sacct instead of a local PID.",
    )
    watch_parser.add_argument("--log-file")
    watch_parser.add_argument("--cost-ceiling-chf", type=float)
    watch_parser.add_argument(
        "--ceiling-pattern", action="append", default=list(DEFAULT_CEILING_PATTERNS)
    )
    watch_parser.add_argument("--started-at", default=utc_now())
    watch_parser.add_argument("--poll-seconds", type=float, default=60)
    watch_parser.add_argument("--timeout-seconds", type=float, default=0)
    watch_parser.add_argument(
        "--state-file",
        default=None,
        help="Durable JSON state; defaults under ~/.local/state/task-notify.",
    )
    watch_parser.set_defaults(handler=watch_ledger)

    test_parser = subparsers.add_parser("test-email", help="send one delivery test")
    test_parser.add_argument("--machine", default=socket.gethostname())
    test_parser.add_argument("--chat-title", required=True)
    test_parser.add_argument("--chat-machine", default=socket.gethostname())
    test_parser.add_argument("--chat-id", default=os.environ.get("CODEX_THREAD_ID"))
    test_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    test_parser.set_defaults(handler=send_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command_name == "watch-ledger" and args.state_file is None:
        args.state_file = str(DEFAULT_STATE_DIR / f"{args.watch_id}.json")
    try:
        return int(args.handler(args))
    except NotifyError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
