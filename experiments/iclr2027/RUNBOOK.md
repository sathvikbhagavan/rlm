# Running the ICLR 2027 experiment campaign

This is the practical guide for the person actually launching RxnHaystack runs.
It explains what we need to run, in what order, and how several co-authors can
divide the work without duplicating expensive API calls.

## What we are running

There are three manifests, but only two represent new work:

| Manifest | Purpose | Expanded runs | Planning cost |
| --- | --- | ---: | ---: |
| `full-campaign.toml` | The complete 100-question benchmark with six models | 6,300 | CHF 939.25 |
| `matched-cardinality-campaign.toml` | The corpus-size/cardinality control on the 65 questions with ordinary reaction-row positives | 1,450 | CHF 70.98 |
| `baseline-campaign.toml` | Archival GPT-5-mini reproduction only | 1,050 | CHF 67.32 |

The full campaign already contains every GPT-5-mini baseline cell. Do **not**
also run `baseline-campaign.toml` unless we specifically need an independent
archival reproduction. The current combined planning estimate for the full and
matched campaigns is CHF 1,010.23, including their contingencies.

The full campaign compares these models:

- SwissAI: DeepSeek-V4-Flash-0731, GLM-5.2, and Qwen3.5-397B-A17B;
- OpenRouter: Gemini-3.7-Flash, Claude-Sonnet-5, and GPT-5-mini.

Each model is evaluated with LLM and CodeAct at 100 and 500 reaction rows, and
RLM at 100, 500, and the full corpus. Every cell has five independent inference
repetitions.

The matched-cardinality campaign uses Qwen3.5-397B-A17B and GPT-5-mini. It holds
K=1 while scaling the corpus through 100, 500, 5,000, 50,000, and full. At
N=5,000 it additionally evaluates K=5 and K=20 on Tier 2--3. Tier 1 remains in
the K=1 sweep because some questions have only one positive. Tier 4 is omitted:
its answers are routes and chains, and we deliberately do not invent a special
meaning of cardinality for them.

## Before anyone spends API credit

Every runner should start from the same clean commit on `main`. From the
repository root:

```bash
git checkout main
git pull --ff-only
uv sync --frozen
uv run --frozen pytest -q
```

Prepare or verify the exact dataset:

```bash
uv run --frozen \
  --with-requirements dataset/requirements.txt \
  python -m dataset.prepare --verify-only
```

If the files are absent, omit `--verify-only`; the preparation command performs
an atomic download, cleaning, checksum check, and row-count check. See
`dataset/README.md` for the expected hashes and the data-location overrides.

Each person uses their own credentials. Create these files outside the
repository and make them private:

```bash
chmod 600 ~/.openrouter_api_key \
  ~/.swissai_research_api_key \
  ~/.wandb_api_key
```

The full manifests currently preflight all three names before starting any
selected subset. A collaborator therefore needs all three private files even if
their assigned selector uses only one inference provider. Values and file paths
are never written to the ledger or artifacts.

Finally, validate the generated scientific inputs and perform the full
provenance preflight:

```bash
uv run --frozen python experiments/iclr2027/generate_full_campaign.py --check
uv run --frozen python experiments/iclr2027/generate_matched_cardinality_campaign.py --check
uv run --frozen rxnhaystack validate experiments/iclr2027/full-campaign.toml
uv run --frozen rxnhaystack validate experiments/iclr2027/matched-cardinality-campaign.toml
uv run --frozen rxnhaystack plan experiments/iclr2027/full-campaign.toml \
  --select 'full-gpt-5-mini-tier1-task1-llm-x100-r01'
```

`plan` must report the expected Git commit and dataset checksums. A tracked
working-tree edit stops an archival run. Generated artifacts are Git-ignored and
do not make the tree dirty.

## Recommended sequence

### 1. Run the free infrastructure smoke tests

```bash
uv run --frozen rxnhaystack run experiments/iclr2027/smoke.toml
uv run --frozen rxnhaystack run experiments/iclr2027/concurrency-smoke.toml
uv run --frozen rxnhaystack run experiments/iclr2027/metrics-adapter-smoke.toml
```

These exercise resume behavior, concurrency, metrics, artifacts, and memory
telemetry without calling a paid model.

### 2. Profile representative real cells

Do this at `--max-parallel 1`. Profile at least one cheap LLM cell, one CodeAct
cell, and the longest Tier-4 full-corpus RLM cell for both an open and a paid
model. For example:

```bash
uv run --frozen rxnhaystack run experiments/iclr2027/full-campaign.toml \
  --select 'full-qwen3.5-397b-tier1-task1-llm-x100-r01' \
  --select 'full-qwen3.5-397b-tier4-task16-codeact-x500-r01' \
  --select 'full-qwen3.5-397b-tier4-task16-rlm-xfull-r01' \
  --max-parallel 1 \
  --secret-file OPENROUTER_API_KEY=~/.openrouter_api_key \
  --secret-file SWISSAI_RESEARCH_API_KEY=~/.swissai_research_api_key \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

Repeat the worst-case probes with the paid models before releasing thousands of
runs. Inspect `metadata.json`, `metrics.json`, `resource-trace.jsonl`, and W&B.
Confirm output quality, API cost, wall time, peak process-tree RSS, and absence
of repeated 429/5xx responses.

### 3. Run the three open models

These are effectively free for us and should go first. They test the complete
pipeline and begin producing paper results while paid profiling is reviewed.
The model-level selectors are:

```text
full-deepseek-v4-flash-*
full-glm-5.2-*
full-qwen3.5-397b-*
```

Start with `--max-parallel 2` or `4` per machine. Increase only after observing
memory and SwissAI rate-limit behavior.

### 4. Run the three closed models

After reviewing their profile cells, release:

```text
full-gemini-3.7-flash-*
full-claude-sonnet-5-*
full-gpt-5-mini-*
```

Claude accounts for most of the projected spend, so watch its first several
Tier-4 RLM cells before letting the entire selector continue. Successful IDs
are immutable and skipped on rerun, so stopping and resuming is safe.

### 5. Run the matched-cardinality study

This study can run at the same time as the full campaign **on a different
machine**. Do not launch both campaign schedulers freely on one host: their
48-GiB memory budgets are independent and cannot see each other's reservations.

```text
matched-qwen3.5-397b-*
matched-gpt-5-mini-*
```

Its RLM workers intentionally use question-level parallelism one. The launcher
still runs independent worker cells in parallel, subject to the memory budget.

## A complete launch command

Replace the selector with the co-author's assigned non-overlapping glob:

```bash
uv run --frozen rxnhaystack run experiments/iclr2027/full-campaign.toml \
  --select 'full-qwen3.5-397b-*' \
  --max-parallel 4 \
  --secret-file OPENROUTER_API_KEY=~/.openrouter_api_key \
  --secret-file SWISSAI_RESEARCH_API_KEY=~/.swissai_research_api_key \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

Parallelism exists at two levels. `--max-parallel` is the number of isolated
worker processes. Within a worker, LLM questions use up to four concurrent API
requests, CodeAct up to two isolated agents, and RLM one question at a time.
The scheduler also enforces a 48-GiB aggregate reservation: typical reservations
are 2--4 GiB for LLM, 4--6 GiB for CodeAct, and 8--16 GiB for RLM. A separate
watchdog terminates any worker that crosses its hard limit.

Five repetitions remain separate runs. Do not replace them with one request
using `n=5`: CodeAct and RLM are adaptive trajectories whose later calls depend
on earlier responses.

## Splitting work among co-authors

The safest split is one model per person or machine. Record assignments in the
master chat before launching; no two people should receive overlapping globs.
For example:

| Owner | Assignment |
| --- | --- |
| A | `full-deepseek-v4-flash-*` |
| B | `full-glm-5.2-*` |
| C | `full-qwen3.5-397b-*` and `matched-qwen3.5-397b-*` |
| D | `full-gemini-3.7-flash-*` |
| E | `full-claude-sonnet-5-*` |
| F | `full-gpt-5-mini-*` and `matched-gpt-5-mini-*` |

If there are fewer machines, split further by tier or method. Run IDs follow:

```text
full-{model}-{tier}-task{task}-{method}-x{context}-r{repetition}
```

Therefore selectors such as `full-claude-sonnet-5-tier4-*` or
`full-gpt-5-mini-*-rlm-*` are valid. Multiple `--select` arguments form a union.
Always use a written assignment list; shell globs are easy to overlap.

Each machine has its own SQLite ledger. Do not copy one collaborator's
`ledger.sqlite3` over another's. W&B combines the scientific records through the
globally unique `rxnhaystack_run_id`. For archival handoff, each collaborator
should preserve their whole campaign artifact directory and identify the exact
Git commit, selector, hostname, and owner. Keep independently produced artifact
trees under owner-specific directories when collecting them centrally.

If a single consolidated local ledger is important, run all selectors through
one coordinator checkout instead of distributing them across independent
machines.

## Monitoring, interruption, and retries

In another terminal:

```bash
uv run --frozen rxnhaystack status experiments/iclr2027/full-campaign.toml
```

An ordinary restart skips successes. Retry recorded failures explicitly:

```bash
uv run --frozen rxnhaystack run experiments/iclr2027/full-campaign.toml \
  --select 'ASSIGNED-GLOB' --retry-failed --max-parallel 2 \
  --secret-file OPENROUTER_API_KEY=~/.openrouter_api_key \
  --secret-file SWISSAI_RESEARCH_API_KEY=~/.swissai_research_api_key \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

If a machine died while attempts were marked `running`, add
`--recover-running`. The abandoned attempts remain in the ledger as failures;
the new attempts are not allowed to erase their history.

Pause a selector if costs or error rates depart materially from profiling. Do
not repeatedly retry authentication failures, systematic 400 responses, or
memory-limit failures without fixing their cause.

## Definition of done

For every assigned selector:

- `status` has no pending, running, or unexplained failed runs;
- every successful attempt has metrics, metadata, stdout/stderr, and a resource trace;
- W&B contains the expected unique run IDs and model/provider labels;
- actual cost is reconciled against the ledger and provider dashboard;
- the artifact directory and exact Git commit are archived;
- a second person checks cell counts before plots or aggregate tables are regenerated.
