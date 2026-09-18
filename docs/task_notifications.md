# Email notifications for long-running tasks

`task-notify` is a small, project-independent utility for receiving an email
when a command or a SQLite-backed experiment finishes, fails, stops early, or
reaches a recorded cost boundary. Every message states what ran, the execution
machine, the project, and the title and machine of the Codex chat that launched
it.

It uses ordinary SMTP, so there is no per-message service cost. Credentials are
read from a private file and never passed on the command line.

## One-time mail setup

Create `~/.config/task-notify/config.toml`:

```toml
recipient = "you@example.com"
sender = "you@example.com"

[smtp]
host = "smtp.gmail.com"
port = 465
security = "ssl"
username = "you@example.com"
password_file = "~/.smtp_app_password"
```

For Gmail, `~/.smtp_app_password` must contain a Google app password, not the
normal account password. Protect it with:

```bash
chmod 600 ~/.smtp_app_password
```

Verify delivery:

```bash
task-notify test-email \
  --chat-title "My project master chat" \
  --chat-machine "liacpc14"
```

## Wrap any command

```bash
task-notify run \
  --watch-id analysis-20260918 \
  --title "Final analysis" \
  --description "Regenerate all paper tables and figures" \
  --project "My project" \
  --chat-title "Paper master chat" \
  --chat-machine "liacpc14" \
  -- make figures
```

The wrapped command retains its original exit code. A successful exit sends a
`DONE` message; a nonzero exit sends `FAILED`.

## Watch an RxnHaystack-style SQLite ledger

```bash
task-notify watch-ledger \
  --watch-id model-docker-jobs \
  --title "Model Docker RLM jobs" \
  --description "Tier-4 Tasks 16, 17, and 17b" \
  --project "RxnHaystack" \
  --chat-title "RxnHaystack ICLR 2027 master chat" \
  --chat-machine "liacpc14" \
  --ledger /absolute/path/to/ledger.sqlite3 \
  --select 'model-*-task16-*' \
  --select 'model-*-task17-*' \
  --launcher-pid 12345 \
  --log-file /absolute/path/to/queue.log \
  --cost-ceiling-chf 30
```

Selectors use shell-style matching but are evaluated internally; they are not
expanded by the shell. The watcher totals calls, tokens and recorded CHF from
the selected rows. It recognizes a completed scope, a scope with failures, a
dead launcher with unfinished work, a numerical cost limit, and common budget
or balance-stop messages in the queue log.

Watcher state is stored with mode 600 under `~/.local/state/task-notify/`.
Terminal notifications remain pending and are retried if SMTP is temporarily
unavailable, so the result is not lost merely because mail delivery failed at
the moment the experiment ended.
