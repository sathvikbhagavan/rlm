# Shared experiment control room

This is the common live view for experiments running on `liacpc14`,
`liacpc15`, Jed, and Kuma. It replaces status numbers copied by hand between
chats. Each machine reads its own SQLite result ledger, removes sensitive
information, and publishes a compressed status record to the shared `liac`
W&B team. The dashboard merges those records by exact run ID.

The dashboard is read-only. It cannot launch, retry, or cancel model work.
Local SQLite ledgers remain the authoritative scientific records.

## What it reports

- expected, successful, active, stale, failed, and pending jobs;
- a model-by-method matrix for LLM, CodeAct, and RLM;
- owner, machine, Git commit, Slurm job or tmux session, and last update;
- API calls, tokens, recorded cost, unknown-cost attempts, duration, and peak
  memory when the local metrics contain them;
- sanitized failure classes such as API timeout, rate limit, context overflow,
  policy refusal, memory limit, or interruption;
- duplicate execution when independent machines appear to have attempted the
  same immutable run ID.

It never publishes prompts, responses, raw errors, commands, environment
variables, credentials, artifact paths, resource-trace paths, or dataset paths.
Status files are mode `0600`. W&B uploads are gzip-compressed; the current
6,300-job definition compresses to roughly half a megabyte per update.

## One source means one ledger

A source ID identifies one physical SQLite ledger and must remain stable. Do
not reuse it for another clone or ledger. If phases use separate clones and
ledgers, give each phase its own source ID. This lets the merger distinguish a
copied ledger from accidental duplicate execution.

Recommended names for the current work are:

| Work | Source ID |
| --- | --- |
| Amin's local v34 ledger | `liacpc14-v34` |
| Sathvik's Qwen full ledger | `liacpc15-qwen-v34` |
| Sathvik's Gemini full ledger | `liacpc15-gemini-v34` |
| Jed GLM RLM ledger | `jed-glm-rlm-v34` |
| Kuma GPT LLM ledger | `kuma-gpt-llm-v34` |
| Kuma GPT CodeAct ledger | `kuma-gpt-codeact-v34` |
| Kuma GPT RLM ledger | `kuma-gpt-rlm-v34` |
| Kuma Claude LLM ledger | `kuma-claude-llm-v34` |
| Kuma Claude CodeAct ledger | `kuma-claude-codeact-v34` |
| Kuma Claude RLM ledger | `kuma-claude-rlm-v34` |
| Qwen matched-cardinality ledger | `liacpc15-qwen-matched-v7` |
| GPT matched-cardinality ledger | `kuma-gpt-matched-v7` |

If several phases truly share one ledger, publish it once under one source ID.

## Publish one update

Pull the current repository version on the machine, then run from its clone:

```bash
uv run --frozen rxnhaystack control-room update \
  experiments/iclr2027/full-campaign.toml \
  --source-id jed-glm-rlm-v34 \
  --machine jed \
  --owner Sathvik \
  --scheduler-job-id 66534424 \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

Change the source ID, machine, owner, and optional job/session label for the
ledger being reported. Every person uses their own W&B key. The key file must
be private (`chmod 600 ~/.wandb_api_key`). Its value and path are not included
in the status record.

For a tmux-run phase, use a session label instead:

```bash
uv run --frozen rxnhaystack control-room update \
  experiments/iclr2027/full-campaign.toml \
  --source-id liacpc14-v34 \
  --machine liacpc14 \
  --owner Amin \
  --session-name rxn-deepseek-capped-v34 \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

For matched cardinality, use
`experiments/iclr2027/matched-cardinality-campaign.toml`.

### Keep a long experiment current

An updater can remain in its own tmux session:

```bash
uv run --frozen rxnhaystack control-room update \
  experiments/iclr2027/full-campaign.toml \
  --source-id liacpc14-v34 \
  --machine liacpc14 \
  --owner Amin \
  --session-name rxn-deepseek-capped-v34 \
  --watch-seconds 300 \
  --heartbeat-seconds 1800 \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

It checks locally every five minutes. It publishes only when ledger state
changes or when the 30-minute heartbeat is due. `Ctrl-C` stops only the status
updater; it does not touch the experiment. Use `--local-only` to test record
generation without contacting W&B.

## Open the combined dashboard

On `liacpc14`, or any machine with the repository and a W&B key:

```bash
uv run --frozen rxnhaystack control-room view \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

This downloads the latest record from every source, writes:

- `artifacts/control-room/index.html` — interactive dashboard;
- `artifacts/control-room/status.md` — concise generated Markdown status;
- `artifacts/control-room/shared/*.json` — verified local cache;

and serves the dashboard at `http://127.0.0.1:8765/index.html`. Press `Ctrl-C`
to stop the web server. While it is open, it downloads fresh source records and
regenerates the page every five minutes; the browser reloads the page once per
minute. Override the server interval with `--refresh-seconds` if needed.

When viewing a remote machine from a laptop, forward the local-only port:

```bash
ssh -L 8765:127.0.0.1:8765 amin@liacpc14
```

Then open `http://127.0.0.1:8765/index.html` on the laptop. Keeping the server
bound to `127.0.0.1` avoids exposing it to the network.

To generate the files without starting a web server:

```bash
uv run --frozen rxnhaystack control-room view \
  --no-serve \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

## How merging works

Every experiment file defines the expected run IDs. Machines publish only runs
that actually started, so their thousands of untouched pending rows are never
added together. The dashboard takes the union of expected IDs and merges
attempts using their run ID, specification hash, attempt number, and start time.

- A successful attempt is counted once even if its ledger was copied.
- A fresh active attempt is shown as running.
- An active attempt reported only by a source whose heartbeat is older than two
  hours is shown as stale.
- A failed result stays failed unless another source has a valid success.
- Independent attempt histories for one run ID raise a duplicate warning.
- Costs and calls are summed once per unique attempt. Missing cost remains
  unknown rather than becoming zero.
- Different experiment-file hashes are displayed separately, never combined.

## What each Codex should report back

After updating its clone, each Codex should:

1. run its focused tests and confirm the exact Git commit;
2. identify every local ledger and assign one stable source ID to each;
3. publish one update with the correct owner, machine, and Slurm/tmux label;
4. report the W&B URL, source ID, expected/observed counts, snapshot hash, and
   whether the worktree was clean;
5. leave the updater watching only while its associated experiment is active.

It must not change or recover ledger state merely to publish status. A status
failure must not stop, restart, or alter the model experiment.

## Current limitations

- Scheduler state is supplied as a label; the dashboard does not query remote
  Slurm clusters.
- Cost comes from ledger metrics, not the provider's live account balance.
  Attempts without usage metadata remain unknown.
- Median duration is descriptive, not a reliable completion forecast, because
  task durations differ greatly.
- The dashboard deliberately has no controls that mutate experiments.
