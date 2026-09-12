# What changed in RxnHaystack

This guide brings a co-author up to speed on the work completed before the ICLR
2027 benchmark runs. It focuses on what changed, why it matters scientifically,
and what should be checked before we spend API credit.

For the exact sequence of commands and reporting steps, read
[`running_benchmark.md`](running_benchmark.md) after this document.

## Short version

The original research scripts could produce the workshop results, but they were
difficult to reproduce safely on another machine. Many scripts assumed one
person's filesystem paths, jobs were usually launched manually, failures were
easy to lose, and memory/cost limits were not coordinated across simultaneous
runs.

The revised code keeps the benchmark questions and prompts, while adding:

- one verified way to reconstruct the exact USPTO dataset;
- portable model, dataset, seed, and context settings;
- support for both OpenRouter and SwissAI;
- safe simultaneous question and job execution;
- complete records of calls, tokens, time, cost, failures, and memory use;
- restartable benchmark execution with unique job names;
- checked experiment files for the six-model benchmark and the
  matched-cardinality study;
- tests covering the new behavior.

The submitted plot data was also rechecked: it contains all 100 benchmark
questions in 30 task-script groups, including the four Tier-4 Task-15 questions.
The earlier 96-question concern applied to an older incomplete result set, not
the final plotting archive.

## 1. The dataset can now be reproduced exactly

The benchmark uses the 2023 USPTO reaction-SMILES release. The new preparation
command downloads the raw file safely, checks it, cleans it with a recorded
RDKit version, and verifies the final result.

Expected files:

| File | Rows | SHA-256 fingerprint |
| --- | ---: | --- |
| Raw USPTO reactions | 137,261 | `8ba53c8aa5a513bdef651fb06f3ed1392ac3056c5f472087bdc52e820440b791` |
| Cleaned reactions used by RxnHaystack | 122,456 | `9f9b2e71676e3e8f132b495b3fc62e2fdec01bc5b43f7be3dc09ef279c351b14` |

The cleaning procedure parses reactions with RDKit, removes atom-map numbers,
uses canonical isomeric SMILES, sorts molecules within each reaction field, and
removes duplicate canonical reactions. The dataset procedure uses RDKit
2022.9.5 because that is the version needed to reproduce the recorded cleaned
file; the benchmark program itself uses the newer locked project environment.

Read [`dataset/README.md`](../../dataset/README.md) and
[`dataset/prepare.py`](../../dataset/prepare.py).

## 2. The scripts no longer belong to one person's filesystem

The old task scripts contained paths such as `/home/bhagavan/...`. All benchmark
runners now accept their dataset, model, random seed, context size, and output
locations from the common runner. They still have sensible defaults for direct
development use, but a recorded benchmark job receives every important setting
explicitly.

This means Amin and Sathvik can check out the same commit on different machines,
keep datasets in their own home directories, use their own credentials, and
still run scientifically identical jobs.

The central settings and preparation checks live in
[`rxnhaystack/runtime.py`](../../rxnhaystack/runtime.py).

## 3. Every job has a stable identity and history

The TOML experiment files describe the task, method, model, corpus size, random
seed, repetition, estimated cost, memory allowance, and command. The runner
turns each repetition into a unique job name such as:

```text
full-gpt-5-mini-tier1-task1-llm-x100-r01
```

The exact settings are fingerprinted. Reusing the same job name with different
scientific settings is rejected rather than silently mixing results.

A small local SQLite file records pending, running, successful, failed,
interrupted, and retried attempts. Successful jobs are skipped when the same
command is restarted. Failures and abandoned attempts remain visible instead of
being overwritten.

The relevant code is in [`rxnhaystack/manifest.py`](../../rxnhaystack/manifest.py),
[`rxnhaystack/ledger.py`](../../rxnhaystack/ledger.py), and
[`rxnhaystack/launcher.py`](../../rxnhaystack/launcher.py).

## 4. API credentials are private and belong to each user

Credentials are not stored in Git, TOML experiment files, result records, or
commands as literal values. Each user keeps private mode-600 files for:

```text
OPENROUTER_API_KEY
SWISSAI_RESEARCH_API_KEY
WANDB_API_KEY
```

The runner checks that required values exist before starting work and passes
them only to the child process. Recorded results contain credential names, never
values or source paths.

## 5. OpenRouter and SwissAI use the same benchmark prompts

The three closed models use OpenRouter:

```text
google/gemini-3.7-flash
anthropic/claude-sonnet-5
openai/gpt-5-mini
```

The three open models use the SwissAI OpenAI-compatible endpoint:

```text
RCP-AIaaS/deepseek-ai/DeepSeek-V4-Flash-0731
CSCS-Inference/zai-org/GLM-5.2
RCP-AIaaS/Qwen/Qwen3.5-397B-A17B
```

LLM and CodeAct use a shared LlamaIndex chat interface. RLM uses its native
OpenAI-compatible client. Provider selection changes the connection and
credential, not the chemistry question. OpenRouter-only request options are not
sent to SwissAI when the endpoint does not advertise them.

All three SwissAI model names were confirmed against the live service and
successfully answered test requests. A live Qwen RLM check completed both root
and recursive model calls.

See [`rxnhaystack/providers.py`](../../rxnhaystack/providers.py).

## 6. Independent questions can run simultaneously

The original benchmark commonly processed one question after another. The new
code allows bounded simultaneous work:

- LLM: up to four questions in flight inside one worker;
- CodeAct: up to two questions, each with its own agent, tools, and model client;
- RLM: one question at a time because long recursive tasks need more memory and
  share complex state;
- outer runner: several independent worker processes, controlled by
  `--max-parallel`.

CodeAct agents never share a mutable tool environment. Results are restored to
the original question order before aggregate metrics are calculated.

Every CodeAct tool now receives that question's retrieved reaction rows as the
preloaded Python list `lines`. Tier 1 and one Tier-2 script already followed
this design; three other Tier-2 scripts accepted the rows but did not insert
them into the tool, and Tier 3--4
left the tool namespace empty. This encouraged models to copy tens of thousands
of tokens back into a Python literal. The system instruction now tells every
model to operate on `lines` directly. A structural test checks all 34 CodeAct
scripts.

We kept five repetitions as five independent jobs. A single `n=5` model request
would not be equivalent for CodeAct or RLM because later calls depend on earlier
responses and tool outputs.

See [`rxnhaystack/concurrency.py`](../../rxnhaystack/concurrency.py) and the
migrated Tier 1--4 runner scripts.

## 7. Memory use now limits simultaneous work

Every job declares an estimated RAM allowance and a hard upper limit. The outer
runner starts another worker only when the sum of active allowances fits within
the machine-wide 48-GiB benchmark limit.

While a worker runs, the code repeatedly measures the resident memory of the
worker and all processes it started. This “process-tree RAM” includes Python,
RLM helpers, and any child process, rather than looking at only the first
process. A worker crossing its hard limit is stopped and recorded as failed.

Pressing Ctrl-C once now signals every active worker, terminates its isolated
process tree, and records the interrupted attempt. If a machine stops too
abruptly to record that transition, the next launch can recover the still-marked
running attempt without deleting its history.

Each attempt writes `resource-trace.jsonl`, a time-ordered memory record. RLM
iteration and recursive-call events share the same timeline without storing
private prompt text, so a memory spike can be associated with the work occurring
at that moment.

See [`rxnhaystack/resources.py`](../../rxnhaystack/resources.py).

SwissAI's large models use an explicit 300-second request deadline instead of
inheriting the chat client's 60-second default. Hidden client-level retries are
disabled so a single slow request cannot silently multiply that deadline;
CodeAct and whole-job retries remain visible in the result history.

SwissAI chat calls use the endpoint's native `enable_thinking=false`
chat-template setting. Live trials showed that Qwen otherwise spent 30,000-token
allowances in a separately returned hidden chain of thought before emitting any
final content. LlamaIndex cannot pass that hidden channel to CodeAct, so CodeAct
keeps its visible multi-turn reasoning/tool loop but not the unusable server-side
thinking channel. This transport difference must be reported with the results.

## 8. Time, tokens, calls, and cost are recorded consistently

The previous task scripts already logged useful information to W&B, but fields
were not uniform enough for one reliable completion record. The new adapter
collects the existing per-question logs and writes a standard `metrics.json`
containing:

- successful model-call count;
- input, output, and total tokens;
- complete wall time;
- model/tool timing where available;
- USD and converted CHF cost;
- task-specific results;
- peak process-tree RAM and whether the limit was reached.

SwissAI currently does not bill us per token, so those jobs record zero provider
cost rather than an absent cost field. W&B settings now also identify the exact
job, repetition, provider, question parallelism, context size, and positive
cardinality.

See [`rxnhaystack/campaign_metrics.py`](../../rxnhaystack/campaign_metrics.py),
[`rxnhaystack/metrics.py`](../../rxnhaystack/metrics.py), and
[`rxnhaystack/worker.py`](../../rxnhaystack/worker.py).

## 9. The full six-model experiment is written down and checked

The main experiment contains the final 100 questions in 30 task-script groups.
For each of six models it runs seven method/context settings and five
repetitions:

```text
30 task scripts × 7 settings × 5 repetitions = 1,050 jobs per model
6 models × 1,050 jobs = 6,300 jobs

100 questions × 7 settings × 5 repetitions = 3,500 trajectories per model
6 models × 3,500 trajectories = 21,000 question-level trajectories
```

The old CHF 1,200 GPT-only figure was an artificial per-job reservation, not an
expected bill. The revised planning estimate uses historical token counts and
current closed-model prices:

| Model | Estimate including safety margin |
| --- | ---: |
| Three SwissAI models | CHF 0 |
| Gemini-3.7-Flash | CHF 229.89 |
| Claude-Sonnet-5 | CHF 613.04 |
| GPT-5-mini | CHF 96.33 |
| **Full benchmark** | **CHF 939.25** |

Actual cost can differ because models may take different numbers of recursive
turns. This is why the running guide requires small real-model trials before the
full launch.

See [`generate_full_campaign.py`](generate_full_campaign.py) and
[`full-campaign.toml`](full-campaign.toml).

## 10. Positive cardinality is now a real experimental control

The earlier TOML field for positive cardinality reached the worker environment
but did not alter context sampling. It now does.

For ordinary reaction-set questions, the sampler includes exactly K
ground-truth reaction rows and excludes every other positive reaction. K=0 is
supported, and an impossible K fails clearly rather than being silently changed.
For a “full” matched context, the context contains every negative row plus
exactly K sampled positives.

We apply this only where K has the simple meaning “number of positive reaction
rows”:

- all 65 submitted Tier 1--3 questions;
- Qwen3.5-397B-A17B and GPT-5-mini;
- K=1 across corpus sizes 100, 500, 5,000, 50,000, and full;
- K=5 and K=20 additionally at N=5,000 for Tier 2--3;
- five repetitions.

Tier 4 is excluded because its answers are routes, chains, and supporting
reaction structures. We deliberately did not create a complicated special
definition for it.

The calculation is:

```text
per model: [(1 Tier-1 script × 5 settings) + (20 Tier-2/3 scripts × 7 settings)] × 5 repetitions = 725 jobs
two models: 2 × 725 = 1,450 jobs

per model: [(10 Tier-1 questions × 5 settings) + (55 Tier-2/3 questions × 7 settings)] × 5 repetitions = 2,175 trajectories
two models: 2 × 2,175 = 4,350 question-level trajectories
```

See [`rlm/codeact_helpers.py`](../../rlm/codeact_helpers.py),
[`generate_matched_cardinality_campaign.py`](generate_matched_cardinality_campaign.py),
and [`matched-cardinality-campaign.toml`](matched-cardinality-campaign.toml).

## 11. The runner checks work before spending money

The command-line interface provides four actions:

- `validate`: check the internal consistency of an experiment file;
- `plan`: check this machine's Git state, dataset fingerprints, and commands,
  then show selected jobs without executing them;
- `run`: execute pending selected jobs and record every attempt;
- `status`: summarize the local completion record.

The `plan` output now prints both dataset row counts and SHA-256 fingerprints,
making it possible for two users to compare their inputs directly.

## 12. Tests added for the new behavior

The tests cover:

- dataset cleaning and exact reconstruction;
- experiment-file parsing, job identity, budgets, and secrets;
- success, failure, interruption, retry, and resumption records;
- simultaneous execution and preservation of answer order;
- memory measurement and enforced limits;
- provider selection for OpenRouter and SwissAI;
- metrics conversion and required fields;
- exact positive-cardinality sampling, including impossible cases;
- completeness and cost of the generated six-model and matched-cardinality
  experiments;
- importability and migration of all 102 task runners;
- dataset fingerprints printed by `plan`.

At the time of this guide, the full local suite passes with 369 tests and 10
optional skips. The skips cover optional environments not required for the
benchmark.

## 13. Human chemistry review support

The merged project also contains a standalone `human_eval/` application for
specialist review of Tier-4 prospective questions. It includes instructions,
study design, assignment handling, restoration/export tools, tests, and a
bundled cleaned dataset for the standalone reviewer package.

This application is separate from the automated six-model benchmark. Running
the automated experiments does not complete expert route validation; the human
review still needs assigned chemistry reviewers and its own recorded outputs.

See [`human_eval/README.md`](../../human_eval/README.md).

## What has not been hidden or simplified away

- The three API/W&B credential values are currently required before any selected
  part of the full experiment starts, even when only one provider is selected.
- Separate machines keep separate local SQLite completion records. W&B and the
  unique job names provide the common view; local database files must not
  overwrite one another.
- Cost estimates depend on historical GPT-5-mini token use. Model-specific
  trajectory lengths must be checked with the small trials.
- The optional GPT-5-mini baseline file duplicates GPT-5-mini work already in
  the six-model experiment and should not normally be run.
- Tier-4 matched cardinality is intentionally absent.
- The unrelated `sb/catalysis` upstream-development branch was not merged into
  the paper code.

## What Sathvik should review before launch

Please check the following scientific choices rather than reviewing every line
of runner code first:

1. Confirm the raw/cleaned dataset row counts and fingerprints.
2. Confirm that the 30 submitted task groups and 100 questions are the intended
   benchmark, including Tier-4 Task 15.
3. Confirm the six exact model names and the two providers.
4. Confirm the seven main method/context settings and five repetitions.
5. Confirm the 65-question matched-cardinality scope and K/N settings.
6. Confirm that the cost ceiling and proposed division of work are acceptable.
7. Run the preparation checks and small real-model trials in
   [`running_benchmark.md`](running_benchmark.md).
8. Compare trial outputs, memory, timing, and costs with Amin before either user
   releases their full assigned set of jobs.

## Relevant commits

These commits contain the main pieces discussed above:

```text
b423506  Add standalone RxnHaystack human validation pilot
ac826ea  Update human evaluation for revised Tier 4 questions
3c47e3e  Build reproducible multi-provider benchmark experiments
b662deb  Add the first collaborative execution guide
```

The current documentation revision supersedes the first execution guide. Use
`git log --oneline` to see its final commit after this revision is approved and
pushed.
