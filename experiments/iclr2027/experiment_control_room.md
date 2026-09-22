# RxnHaystack Dashboard

The dashboard combines sanitized, read-only summaries of the authoritative
SQLite experiment ledgers on `liacpc14`, `liacpc15`, Jed, and Kuma. The ledgers
remain the scientific source of truth. W&B only carries status snapshots; it
never receives prompts, responses, secrets, commands, or artifact paths.

## 1. Open it from Amin's Mac

Paste this single line into the laptop terminal:

```bash
DASHBOARD_HOST=amin@128.178.38.26; DASHBOARD_SOCKET="$HOME/.ssh/rxnhaystack-dashboard.sock"; ssh "$DASHBOARD_HOST" 'test -d "$HOME/rlm_dashboard/.git" || git clone git@github.com:sathvikbhagavan/rlm.git "$HOME/rlm_dashboard"; cd "$HOME/rlm_dashboard" && git pull --ff-only origin main && "$HOME/.local/bin/uv" sync --frozen && bash experiments/iclr2027/start_dashboard_viewer.sh' && { ssh -S "$DASHBOARD_SOCKET" -O exit "$DASHBOARD_HOST" >/dev/null 2>&1 || true; rm -f "$DASHBOARD_SOCKET"; for dashboard_pid in $(lsof -nP -tiTCP:8876 -sTCP:LISTEN 2>/dev/null); do kill "$dashboard_pid"; done; dashboard_wait=0; while lsof -nP -tiTCP:8876 -sTCP:LISTEN >/dev/null 2>&1 && [ "$dashboard_wait" -lt 50 ]; do sleep 0.1; dashboard_wait=$((dashboard_wait + 1)); done; ! lsof -nP -tiTCP:8876 -sTCP:LISTEN >/dev/null 2>&1; } && ssh -fN -M -S "$DASHBOARD_SOCKET" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 8876:127.0.0.1:8765 "$DASHBOARD_HOST" && open "http://127.0.0.1:8876/index.html?$(date +%s)"
```

It updates the separate dashboard checkout on `liacpc14`, restarts the viewer,
closes the named dashboard tunnel, clears a legacy listener left by an older
version of this command, opens one new managed tunnel, and opens the page with
macOS `open`. It never touches a benchmark process. If the page later reports
a lost connection, paste the same line again.

## 2. Make each execution machine report

Run the applicable one-line command once on each machine now. The command:

1. creates or updates a separate `~/rlm_dashboard` checkout;
2. finds every authoritative ledger assigned to that machine;
3. validates it without changing it;
4. publishes immediately; and
5. leaves a reporter running in `tmux`, checking every five minutes and sending
   a heartbeat at least every 30 minutes.

On `liacpc14`:

```bash
test -d "$HOME/rlm_dashboard/.git" || git clone git@github.com:sathvikbhagavan/rlm.git "$HOME/rlm_dashboard"; cd "$HOME/rlm_dashboard" && git pull --ff-only origin main && "$HOME/.local/bin/uv" sync --frozen && RXNHAYSTACK_DASHBOARD_ENTITY=liac bash experiments/iclr2027/start_dashboard_reporting.sh Amin liacpc14 "$HOME" --restart
```

On Jed:

```bash
test -d "$HOME/rlm_dashboard/.git" || git clone git@github.com:sathvikbhagavan/rlm.git "$HOME/rlm_dashboard"; cd "$HOME/rlm_dashboard" && git pull --ff-only origin main && "$HOME/.local/bin/uv" sync --frozen && RXNHAYSTACK_DASHBOARD_ENTITY=liac bash experiments/iclr2027/start_dashboard_reporting.sh Amin jed "$HOME" --restart
```

On Kuma:

```bash
test -d "$HOME/rlm_dashboard/.git" || git clone git@github.com:sathvikbhagavan/rlm.git "$HOME/rlm_dashboard"; cd "$HOME/rlm_dashboard" && git pull --ff-only origin main && "$HOME/.local/bin/uv" sync --frozen && RXNHAYSTACK_DASHBOARD_ENTITY=liac bash experiments/iclr2027/start_dashboard_reporting.sh Amin kuma "$HOME" --restart
```

On Sathvik's `liacpc15`:

```bash
test -d "$HOME/rlm_dashboard/.git" || git clone git@github.com:sathvikbhagavan/rlm.git "$HOME/rlm_dashboard"; cd "$HOME/rlm_dashboard" && git pull --ff-only origin main && "$HOME/.local/bin/uv" sync --frozen && RXNHAYSTACK_DASHBOARD_ENTITY=sathvikbhagavan-epfl bash experiments/iclr2027/start_dashboard_reporting.sh Sathvik liacpc15 "$HOME" --restart
```

Each person uses their own `~/.wandb_api_key`, with file mode 600. Sathvik can
publish to his own W&B entity; the viewer merges that public status project
with the lab project. His underlying experiment projects may remain private.

## 3. Know whether the display is trustworthy

Every model/method/machine row shows:

- `Attempted`: jobs with at least one ledger attempt, out of jobs
  assigned to that row;
- `Terminal coverage`: a normalized bar whose green portion is successful jobs
  and red portion is terminal failures; unfilled space is running or pending;
- `Result last changed`: the newest start or finish time among the row's runs;
- `Reporter last checked`: when that machine last inspected its ledger, which
  can advance even when no experimental result changes;
- `Execution`: `running`, `paused`, `not started`, `complete`, or `complete
  with failures`; this describes the work rather than the reporter;
- `fresh`: the reporter checked within two hours;
- `stale`: unfinished work exists, but its reporter has not checked within two
  hours;
- `unreported`: no reporter from the assigned machine has reached the combined
  dashboard;
- `final`: all jobs in the row are success or failure, so an ongoing heartbeat
  is no longer needed.

Each individual run also shows its last result update and last reporter check.
For a stale or unreported row/run, press **copy refresh command**, paste it into
the named machine, wait for publication to finish, and refresh the browser.
That button supplies the exact command and W&B entity for the owner.

The dashboard cannot claim that an unreported ledger is empty. It displays the
job as unknown/not started until its assigned machine reports. Finished results
remain valid if their old reporter later stops.

## 4. When must a reporter be restarted?

Normally, setup is one-time. Restart the machine's command only after:

- the machine rebooted;
- its reporter `tmux` session was stopped;
- a new authoritative ledger was created or moved; or
- the dashboard marks unfinished work stale or unreported.

Re-running the command is safe. `--restart` replaces only dashboard reporter
sessions; it does not stop or alter experiments.

## 5. What is included

The dashboard has exactly six top-level scientific sections, in this order:

1. the full six-model benchmark;
2. the matched-cardinality study;
3. the oracle-predicate study;
4. the deterministic oracle executor;
5. the prospective Task-16 decomposition; and
6. the DeepSeek CodeAct x1000 study.

Machine shards, retries, provider changes, pilots and repair experiments never
appear as additional sections. Their individual attempts are folded into the
corresponding canonical runs inside one of these six studies. Unknown future
experiment names are hidden until they are deliberately approved as a new
scientific study.

Direct-OpenAI GPT completion/recovery records and paid-OpenRouter DeepSeek
continuations are folded into the corresponding original run IDs in the full
benchmark view. They are not displayed as competing experiments. The
individual-run table retains each alternate completion run ID so the transport
change remains auditable.

The viewer reads these W&B status projects:

- `liac/rxnhaystack-dashboard`;
- `sathvikbhagavan-epfl/rxnhaystack-dashboard`; and
- the old `liac/rxnhaystack-control-room` project during migration.

Duplicate snapshots of the same immutable attempt are deduplicated. Low-level
reporter diagnostics are collapsed by default; the main status table is the
normal operational view.
