# Email notifications for RxnHaystack jobs

RxnHaystack uses the standalone, project-independent `task-notify` utility.
Its source, complete setup guide, tests, and attachment examples live at
<https://github.com/amansouri3476/tasknotify>; they are not part of this
research repository. The same utility can monitor CoQ, Catelier, or unrelated
work.

Install it separately:

```bash
git clone git@github.com:amansouri3476/tasknotify.git ~/tasknotify
uv tool install --editable ~/tasknotify
task-notify --help
```

The command can wrap arbitrary work or watch selected rows in the RxnHaystack
SQLite ledger. A typical attachment is:

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
  --launcher-pid 12345 \
  --log-file /absolute/path/to/queue.log \
  --progress-file /absolute/path/to/resource-trace.jsonl \
  --cost-ceiling-chf 30
```

The default is one email per hour. It reports `PROGRESS` when the selected
ledger, active run, or progress file changes; `STATUS` when the work remains
active but unchanged for one interval; and `STALL` after two hours with no
observable change. A stall alert is not proof of a hung request. Configure the
schedule with:

```bash
--status-interval-seconds 3600 --stall-after-seconds 7200
```

Use `--slurm-job-id JOB_ID` instead of `--launcher-pid` on Jed or Kuma. The
private SMTP settings and durable state remain outside every project at
`~/.config/task-notify/` and `~/.local/state/task-notify/`.

See the standalone README before attaching a new watcher. Never copy an SMTP
app password into this repository, a shell command, a log, or a chat.
