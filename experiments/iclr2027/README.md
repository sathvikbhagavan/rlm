# ICLR 2027 experiment operations

The campaign launcher treats an experiment matrix as immutable scientific input.
Changing a command, model, seed, condition, or other run field requires a new run
ID; the ledger rejects reuse of an existing ID with a changed specification.

## Preflight and smoke test

```bash
uv run rxnhaystack validate experiments/iclr2027/smoke.toml
uv run rxnhaystack plan experiments/iclr2027/smoke.toml
uv run rxnhaystack run experiments/iclr2027/smoke.toml
uv run rxnhaystack status experiments/iclr2027/smoke.toml
```

The smoke campaign costs nothing. It verifies Git provenance, both dataset
checksums, command availability, launcher-to-worker configuration, the mandatory
metrics contract, artifact metadata, and resume behavior.

## Manifest contract

Each `[[runs]]` table specifies:

- a stable lowercase `id`;
- `task`, `condition`, `method`, and exact `model`;
- `corpus_size` and `positive_cardinality` as independent variables;
- `seed`, `repetitions`, and estimated CHF cost per repetition;
- a scheduler `memory_reservation_mib` and watchdog `memory_limit_mib`;
- the argument-vector `command`, without shell interpolation;
- optional non-secret scalar `env` values.

The campaign specifies its CHF ceiling and a frozen `usd_to_chf` conversion.
Update that conversion from a cited source when the paid matrix is frozen.
Version-controlled manifests must never contain credentials.

Paid campaign manifests declare credential names, never values:

```toml
[campaign]
required_secrets = ["OPENROUTER_API_KEY", "SWISSAI_RESEARCH_API_KEY", "WANDB_API_KEY"]
```

`rxnhaystack run` validates these before claiming or starting any run. A value
may come from the current environment or a `--secret-file`; missing values stop
the campaign with the exact missing names and an example command.

When `campaign.max_parallel_memory_mib` is set, every run must declare a
reservation. The scheduler starts a run only when its reservation fits alongside
the currently active runs. The launcher independently samples the worker's whole
process tree. For a Docker RLM it also reads every attempt-owned container's
Linux memory counter, adds that to host process-tree RAM, and applies
`memory_limit_mib` to the combined value. Host, Docker, and combined peaks and
whether the limit fired are written to both metadata and ledger metrics. The
launcher also writes `resource-trace.jsonl`: timestamped host/Docker/combined
samples and process boundaries. Migrated RLM workers add prompt-free iteration
and subcall start/finish events to that same monotonic timeline, allowing memory
peaks to be attributed to RLM activity without logging model inputs or outputs.
An interrupted RLM attempt removes only Docker containers bearing its unique
label.

## Secrets

Secret files must be mode `0600`. They are injected into child processes but
only their environment-variable names are recorded:

```bash
uv run rxnhaystack run experiments/iclr2027/CAMPAIGN.toml \
  --secret-file OPENROUTER_API_KEY=~/.openrouter_api_key \
  --secret-file SWISSAI_RESEARCH_API_KEY=~/.swissai_research_api_key \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

Each collaborator creates their own files outside the repository and protects
them before use:

```bash
chmod 600 ~/.openrouter_api_key ~/.swissai_research_api_key ~/.wandb_api_key
```

Do not put a key directly in a manifest, command line, shell history, Git, or a
shared artifact directory. The same manifest works with every collaborator's
own credentials.

## Selection, parallelism, and recovery

```bash
uv run rxnhaystack run experiments/iclr2027/CAMPAIGN.toml \
  --select 'oracle-*' --max-parallel 4
```

`--max-parallel` caps active worker processes; `max_parallel_memory_mib` is the
second, memory-weighted cap. The launcher's threads only supervise isolated
worker processes. They do not share an RLM, CodeAct executor, W&B run, or API
client between benchmark replicates.

Within a migrated worker, `question_parallelism` controls the bounded number of
questions in flight:

```toml
question_parallelism = 4
```

One-shot LLM and CodeAct workers use async concurrency. Every CodeAct question
owns its own agent, context, executor, and LLM client. RLM workers must likewise
construct an RLM per concurrent question; a mutable RLM instance is never
shared between worker threads.

Before a large campaign, profile representative cells with `--max-parallel 1`.
Use the recorded `peak_combined_memory_mib` to set per-method reservations with
headroom (start with 1.3x the measured peak) and hard limits (start with 1.5x).
Profile the longest Tier-4 RLM task separately; do not extrapolate it from a
one-shot LLM worker.

Five statistical repetitions remain five immutable run IDs. For the current
OpenRouter `openai/gpt-5-mini` endpoint, multi-choice `n` is not supported. Even
on providers that implement `n`, it is not equivalent to five independent
CodeAct or RLM trajectories because later tool and recursive calls depend on
each earlier response. Provider-side prompt caching may reduce repeated-prefix
cost without reusing an answer; OpenRouter response caching must stay disabled
for stochastic evaluation because it returns an earlier response verbatim.

Successful run IDs are skipped automatically. Failed runs require
`--retry-failed`. A process interrupted while its ledger state is `running`
requires the deliberately explicit `--recover-running`; the abandoned attempt
is retained as failed before the new attempt begins.

Artifacts live under the campaign's ignored `artifact_dir`. Every attempt has
`metadata.json`, `metrics.json`, `resource-trace.jsonl`, `stdout.log`, and
`stderr.log`. The SQLite
ledger retains all attempts and their costs. Workers must write metrics through
`rxnhaystack.metrics.write_run_metrics`; required fields are calls, input/output/
total tokens, latency, tool time, and CHF cost. Task-specific results belong in
the `results` object.

## Submitted-paper baseline campaign

`baseline-campaign.toml` is generated from
`generate_baseline_campaign.py`. It covers the paper's 100 fixed questions on
`openai/gpt-5-mini`: LLM and CodeAct at 100 and 500 reaction lines, and RLM at
100, 500, and the full corpus, with five independent model repetitions. This is
1,050 resumable run cells. The estimate is CHF 67.32: historical submitted-run
spend converted at 0.80 CHF/USD with a 25% contingency. The previous CHF 1,200
figure was an artificial flat reservation and must not be interpreted as
expected spend.

The frozen plot inventory has 30 runner groups. Four additional Tier-3 runner
prototypes (`11`, `12`, `16`, and `19`) remain in the repository but are not
part of the paper's 100-question benchmark and are therefore not scheduled.

Regenerate and validate it with:

```bash
uv run --frozen python experiments/iclr2027/generate_baseline_campaign.py
uv run --frozen python experiments/iclr2027/generate_baseline_campaign.py --check
uv run --frozen rxnhaystack validate experiments/iclr2027/baseline-campaign.toml
```

The archival campaign deliberately requires a clean tracked worktree. After
reviewing and committing the implementation, profile one repetition of a cheap,
medium, and worst-case cell before opening the full scheduler. For example:

```bash
uv run --frozen rxnhaystack run experiments/iclr2027/baseline-campaign.toml \
  --select 'baseline-gpt5mini-tier1-task1-llm-x100-r01' \
  --select 'baseline-gpt5mini-tier4-task16-codeact-x500-r01' \
  --select 'baseline-gpt5mini-tier4-task16-rlm-xfull-r01' \
  --max-parallel 2 \
  --secret-file OPENROUTER_API_KEY=~/.openrouter_api_key \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

Use the measured peaks and costs to revise reservations if necessary, generate
a new campaign version if any immutable setting changes, then run pending cells.

## Six-model campaign

`full-campaign.toml` is generated by `generate_full_campaign.py`. It runs the
same 1,050-cell design for three SwissAI-hosted open models and three paid
OpenRouter models, for 6,300 cells total. `RXNHAYSTACK_PROVIDER` selects the
transport without changing task prompts. LLM and CodeAct use the shared
LlamaIndex chat interface; RLM uses its native OpenAI-compatible client.

The provisional CHF 1,161.22 planning estimate uses the submitted GPT-5-mini
token footprint, current per-token list prices for the three closed models,
0.80 CHF/USD, and a 25% general margin. A live Claude CodeAct x500 trial showed
longer conversations than GPT-5-mini, so Claude's CodeAct allowance has a
further measured safety factor. SwissAI-hosted models are budgeted at CHF 0.
Freeze the estimate only after the Docker-based RLM calibration.

Anthropic CodeAct and RLM conversations enable OpenRouter prompt caching and use
the immutable job name to keep turns on the same provider endpoint. This does
not change benchmark prompts or outputs. CodeAct cache reads and writes are
recorded in the timestamped resource trace. The live 10-question Claude trial
cost CHF 4.79 with caching versus CHF 12.13 without it; the recorded allowance
for that cell is CHF 6.51.

```bash
uv run --frozen python experiments/iclr2027/generate_full_campaign.py --check
uv run --frozen rxnhaystack validate experiments/iclr2027/full-campaign.toml
uv run --frozen rxnhaystack run experiments/iclr2027/full-campaign.toml \
  --max-parallel 8 \
  --secret-file OPENROUTER_API_KEY=~/.openrouter_api_key \
  --secret-file SWISSAI_RESEARCH_API_KEY=~/.swissai_research_api_key \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

All three secret names are checked before any worker starts. This campaign-wide
check intentionally requires both provider credentials even when `--select`
temporarily narrows execution to one provider.

## Matched-cardinality experiment

`matched-cardinality-campaign.toml`, generated by
`generate_matched_cardinality_campaign.py`, makes the earlier manifest field
scientifically effective. The random sampler now includes exactly the requested
number of ground-truth reactions and excludes every other positive. Impossible
requests fail rather than silently changing K. With `corpus_size = "full"`,
"full" means all negative reactions plus exactly K sampled positives, so its
actual row count is `dataset_size - total_positives + K`.

The RLM experiment uses Qwen3.5-397B-A17B (SwissAI, open) and GPT-5-mini
(OpenRouter, closed). It covers all 65 submitted Tier-1, Tier-2, and Tier-3
questions—the complete benchmark subset whose answers are ordinary sets of
individual reaction rows. Tier 4 is excluded because its 35 questions concern
routes, chains, or supporting reaction structures rather than independent row
positives; no special cardinality definition is introduced.

- hold K=1 while N is 100, 500, 5,000, 50,000, or full for all 65 questions;
- hold N=5,000 while K is 1, 5, or 20 for the 55 Tier-2 and Tier-3 questions;
- retain Tier 1 only in the K=1 sweep because some products have one match;
- repeat every cell five times.

This is 1,450 worker cells and 4,350 question trajectories, with a CHF 70.98
planning estimate including 50% contingency. Combined with the six-model
matrix, the current estimate is CHF 1,232.20.
