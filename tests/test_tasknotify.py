from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tasknotify import cli


def write_ledger(path: Path, rows: list[tuple[str, str, dict | None, str | None]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                metrics_json TEXT,
                error TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO runs VALUES (?, ?, ?, ?)",
            [
                (run_id, status, json.dumps(metrics) if metrics else None, error)
                for run_id, status, metrics, error in rows
            ],
        )


def notification() -> cli.Notification:
    return cli.Notification(
        event_id="watch:done",
        outcome="done",
        title="Claude Docker RLM",
        description="Remaining Tier-4 cells",
        project="RxnHaystack",
        execution_machine="liacpc14",
        chat_title="RxnHaystack master chat",
        chat_machine="liacpc14",
        chat_id="thread-123",
        started_at="2026-09-18T00:00:00+00:00",
        observed_at="2026-09-18T01:00:00+00:00",
        elapsed_seconds=3600,
        ledger="/tmp/ledger.sqlite3",
        selectors=("claude-*",),
        snapshot=cli.LedgerSnapshot(2, 0, 0, 2, 0, 1.25, 8, 100, ()),
    )


def test_read_ledger_filters_and_summarizes_metrics(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.sqlite3"
    write_ledger(
        ledger,
        [
            ("claude-a", "succeeded", {"cost_chf": 1.2, "calls": 4, "total_tokens": 90}, None),
            ("claude-b", "failed", {"cost_chf": 0.05, "calls": 1, "total_tokens": 10}, "boom"),
            ("gpt-a", "pending", None, None),
        ],
    )

    snapshot = cli.read_ledger(ledger, ("claude-*",))

    assert snapshot == cli.LedgerSnapshot(
        total=2,
        pending=0,
        running=0,
        succeeded=1,
        failed=1,
        cost_chf=1.25,
        calls=5,
        total_tokens=100,
        failures=("claude-b: boom",),
    )
    assert snapshot.terminal


def test_mail_uses_private_password_file_and_contains_launch_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    password = tmp_path / "smtp-password"
    password.write_text("private-app-password\n", encoding="utf-8")
    password.chmod(0o600)
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
recipient = "recipient@example.com"
sender = "sender@example.com"
[smtp]
host = "smtp.example.com"
port = 465
security = "ssl"
username = "sender@example.com"
password_file = "{password}"
""",
        encoding="utf-8",
    )
    client = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = client
    smtp = MagicMock(return_value=context)
    monkeypatch.setattr(cli.smtplib, "SMTP_SSL", smtp)

    cli.send_email(notification(), config)

    client.login.assert_called_once_with("sender@example.com", "private-app-password")
    message = client.send_message.call_args.args[0]
    assert message["To"] == "recipient@example.com"
    assert message["Subject"] == "[DONE] Claude Docker RLM on liacpc14"
    body = message.get_content()
    assert "Launched by Codex chat: RxnHaystack master chat" in body
    assert "Codex chat machine: liacpc14" in body
    assert "recorded cost: CHF 1.2500" in body


def test_password_file_rejects_group_or_world_access(tmp_path: Path) -> None:
    password = tmp_path / "smtp-password"
    password.write_text("secret", encoding="utf-8")
    password.chmod(0o644)

    with pytest.raises(cli.NotifyError, match="group/others"):
        cli.read_password(password)


def test_log_ceiling_detection_is_explicit(tmp_path: Path) -> None:
    log = tmp_path / "queue.log"
    log.write_text(
        "OpenRouter remaining USD 2.9; protected reserve USD 3\n"
        "Claude Docker queue stopped by the balance guard\n",
        encoding="utf-8",
    )

    assert cli.log_has_ceiling(log, cli.DEFAULT_CEILING_PATTERNS)
    assert not cli.log_has_ceiling(log, ("unrelated phrase",))


def test_terminal_failed_ledger_is_durable_and_sent_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.sqlite3"
    write_ledger(ledger, [("run-a", "failed", None, "provider failed")])
    delivered: list[cli.Notification] = []
    monkeypatch.setattr(cli, "send_email", lambda item, _config: delivered.append(item))
    args = argparse.Namespace(
        ledger=str(ledger),
        select=["run-*"],
        launcher_pid=None,
        slurm_job_id=None,
        cost_ceiling_chf=None,
        log_file=None,
        ceiling_pattern=list(cli.DEFAULT_CEILING_PATTERNS),
        state_file=str(tmp_path / "state.json"),
        started_at="2026-09-18T00:00:00+00:00",
        title="Test work",
        description="One test run",
        project="Tests",
        machine="test-host",
        chat_title="Test chat",
        chat_machine="test-host",
        chat_id="thread-test",
        watch_id="test-watch",
        config=str(tmp_path / "unused.toml"),
        timeout_seconds=1,
        poll_seconds=0.001,
    )

    assert cli.watch_ledger(args) == 1
    assert [item.outcome for item in delivered] == ["failed"]
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["sent_events"] == ["test-watch:failed"]
    assert state["pending"] == {}
    assert (tmp_path / "state.json").stat().st_mode & 0o777 == 0o600


def test_slurm_job_state_checks_queue_then_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    running = MagicMock(returncode=0, stdout="RUNNING\n", stderr="")
    run = MagicMock(return_value=running)
    monkeypatch.setattr(cli.subprocess, "run", run)

    assert cli.slurm_job_state("123") == (True, "RUNNING")
    assert run.call_args.args[0][0] == "squeue"

    absent = MagicMock(returncode=0, stdout="", stderr="")
    completed = MagicMock(returncode=0, stdout="COMPLETED|\n", stderr="")
    run.side_effect = [absent, completed]

    assert cli.slurm_job_state("123") == (False, "COMPLETED")
    assert run.call_args.args[0][0] == "sacct"
