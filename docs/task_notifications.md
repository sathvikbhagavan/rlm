# Email notifications for long-running tasks

`task-notify` is a small, project-independent utility for receiving an email
when a command or a SQLite-backed experiment finishes, fails, stops early, or
reaches a recorded cost boundary. Every message states what ran, the execution
machine, the project, and the title and machine of the Codex chat that launched
it.

It uses ordinary SMTP, so there is no per-message service cost. Credentials are
read from a private file and never passed on the command line.

## Where it is installed

The tracked source in this repository is `tasknotify/`. Installing this project
with `uv tool install --editable /path/to/rlm` creates the user command at
`~/.local/bin/task-notify`; uv keeps its environment under
`~/.local/share/uv/tools/rlms/`.

Machine-specific files are deliberately outside Git:

- mail settings: `~/.config/task-notify/config.toml`;
- SMTP app password: `~/.smtp_app_password`, mode 600;
- sent/pending event state: `~/.local/state/task-notify/`, mode 600.

The tracked source can be updated with the repository. The app password must
never be committed, pasted into a job command, or included in a Codex/Claude
prompt.

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

## Attach to a job that is already running

Attaching a watcher does not restart, signal, or otherwise modify the job. The
person or coding assistant attaching it should first identify:

1. the real launcher PID, not merely the `tmux` shell;
2. the absolute SQLite ledger path;
3. non-overlapping run-ID selectors defining exactly the assigned work;
4. the queue log, if it records budget or balance stops;
5. the task's original start time;
6. a stable human title for the Codex/Claude chat that launched it.

Example discovery commands:

```bash
pgrep -af 'rxnhaystack run|run_.*campaign|sbatch'
sqlite3 /absolute/path/ledger.sqlite3 \
  'select status, count(*) from runs group by status;'
```

Then start a sidecar watcher, preferably in `tmux` on a workstation or in a
small scheduler allocation on a cluster:

```bash
tmux new-session -d -s notify-my-job "task-notify watch-ledger \
  --watch-id unique-project-job-id \
  --title 'Human-readable job name' \
  --description 'Model, method, tasks, scales, and repetitions being run' \
  --project 'Project name' \
  --chat-title 'Human-readable Codex or Claude chat title' \
  --chat-machine 'machine-where-that-chat-runs' \
  --ledger /absolute/path/ledger.sqlite3 \
  --select 'exact-non-overlapping-run-glob-*' \
  --launcher-pid ACTUAL_LAUNCHER_PID \
  --log-file /absolute/path/queue.log \
  --cost-ceiling-chf 30 \
  --started-at '2026-09-18T10:40:51+02:00'"
```

After attachment, verify all three of these:

```bash
tmux list-sessions | grep notify-my-job
stat ~/.local/state/task-notify/unique-project-job-id.json
task-notify test-email --chat-title 'Delivery check' --chat-machine "$(hostname)"
```

Omit `--cost-ceiling-chf` for free jobs. Omit `--launcher-pid` only when several
launchers jointly own the selected scope; in that case the watcher waits until
every selected ledger row is terminal. Without a ledger or another reliable
result record, an observer can detect that a process disappeared but cannot
scientifically distinguish success from failure. For such jobs, start the next
run with `task-notify run` or ask the assistant to define explicit success and
failure records first.

For a Slurm allocation, replace `--launcher-pid` with the scheduler ID:

```bash
--slurm-job-id 66534424
```

The watcher checks `squeue` while the allocation is active and `sacct` after it
leaves the queue. The final email includes both the job ID and scheduler state.

## Short prompt for a coding assistant

Use this when asking Codex or Claude to attach notifications:

```text
Attach task-notify to this already-running job without restarting or signalling
it. Inspect the actual launcher PID, ledger, exact non-overlapping run IDs, queue
log, start time, and recorded cost boundary. Use a stable human chat title and
record both the chat title and chat machine in the notification. Verify the
selected ledger count before launch, start the watcher in tmux or a small Slurm
job, confirm its mode-600 state file exists, and send one test email. Do not put
the SMTP app password in a command, log, Git, or your response. Report the
watcher name, selected count, state path, and delivery-test result.
```

## Install on another workstation or cluster

On each machine, pull a commit containing `tasknotify/`, then install it for the
current user:

```bash
git pull --ff-only origin main
uv tool install --editable "$(pwd)"
task-notify --help
```

Create the same private SMTP configuration in that machine's home directory
and protect both files:

```bash
chmod 700 ~/.config/task-notify ~/.local/state/task-notify
chmod 600 ~/.config/task-notify/config.toml ~/.smtp_app_password
```

Run `task-notify test-email` from the same kind of node that will run the
watcher. Some clusters allow HTTPS but block outbound SMTP. If the test fails
on a compute node, keep the experiment there but run the ledger watcher on a
login/transfer node that can read the shared ledger and reach Gmail SMTP. Check
the cluster's login-node policy before leaving a long-lived `tmux` process; a
small scheduler sidecar is preferable where required.
