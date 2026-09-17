# What changed in RxnHaystack

This guide brings a co-author up to speed on the work completed before the ICLR
2027 benchmark runs. It focuses on what changed, why it matters scientifically,
and what should be checked before we spend API credit.

For the exact sequence of commands and reporting steps, read
[`running_benchmark.md`](running_benchmark.md) after this document.
The overnight real-model findings and complete safety-limit table are in
[`testing_and_safety_report.md`](testing_and_safety_report.md).

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
anthropic/claude-haiku-4.5
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

Anthropic's multi-turn CodeAct and RLM requests explicitly enable OpenRouter
prompt caching. The job name supplies a stable provider-routing session, while
the chemistry prompts and scoring remain unchanged. Each CodeAct turn records
cache reads, cache writes, nominal tokens, and actual provider cost in the
timestamped resource trace. A live Claude Tier-4 CodeAct x500 check reused
79.46% of nominal input tokens and reduced cost from CHF 12.13 without caching
to CHF 4.79 with caching; the experiment reserves CHF 6.51 for that cell.

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

Because the context can be tens of thousands of tokens long, the same short
instruction is repeated after the question at the end of the initial user
message. A live 53k-token Qwen probe changed from an unclosed 4,096-token copy of
the context to a complete 390-token action using `lines`.

The shared controller also accepts each task's own answer shape instead of
assuming every answer is one comma-separated line. This matters for Tier 4,
where a correct answer may contain several routes or reaction pairs. After the
eighth permitted reasoning/tool turn, the controller allows at most two
answer-only attempts and explicitly forbids further code. If the first answer
attempt still contains code, it is not executed and the model receives one final
correction. Live Tier-4 Qwen checks exercised both a clean multi-line `ANSWER:`
response and the correction path; tests cover the boundary and answer marker.

Each CodeAct model turn also has a provider-independent 2,048-output-token
limit. A final Qwen trial found one task-specific first turn repeatedly reaching
the 300-second request deadline under the former 30,000-token allowance while
other questions continued normally. The lower explicit limit preserves ordinary
short actions while bounding a pathological turn's latency and paid cost. A
4,096-token follow-up trial still exceeded three minutes; the known-good action
on the same long prompt required only 390 tokens.

Launcher-managed CodeAct tools now run in a persistent child process for each
question rather than a Python thread in the benchmark worker. Normal variables
persist across turns. A generated block exceeding 60 seconds, a native crash,
or a 4,096-MiB address-space limit terminates the complete child process group;
the controller reports the failure and restores a clean process with the same
preloaded `lines`. This replaces a thread timeout that could return an error but
could not stop the underlying computation, leaving a worker unable to finish.
The shared factory applies this to all 34 CodeAct scripts, and agent cleanup
stops every child even when the workflow itself fails or is interrupted.

RLM now sends an explicit limit on every root and recursive request. SwissAI
uses 2,048 tokens with its hidden thinking channel disabled. The three
OpenRouter models use a 4,096-token total ceiling and the provider's recorded
`low` reasoning effort. This is necessary because a GPT-5-mini calibration spent
five consecutive 2,048-token responses entirely on hidden reasoning and returned
zero visible words or actions. OpenRouter's live model metadata reports
reasoning as enabled by default at medium effort for GPT/Gemini and high effort
for Claude. Low effort leaves most of the bounded response for visible RLM code.
The 30-iteration and two-level recursion limits are unchanged. Each question
also has a 30-minute total trajectory cutoff, checked between turns. At that
boundary the RLM asks once for a final answer based on its accumulated work,
records the cutoff with normal metrics, and advances instead of failing and
rerunning the whole multi-question job. A block already in progress retains its
separate five-minute limit, so finalization can occur up to roughly five minutes
after the cutoff. A
GPT-5-mini full-corpus Task-16 diagnostic motivated this limit: seven questions
finished within 25 minutes each, while the next exceeded 40 minutes by repeating
five-minute searches.

Docker-executed model code also has a 300-second wall-time limit per block. It
terminates the complete in-container process group, including multiprocessing
children, reports the timeout to the RLM, and leaves the container usable for a
recovery turn. A live full-corpus Task-16 trajectory otherwise held about 23
GiB of Docker memory and one CPU beyond five minutes. A second trial reached the
30,720-MiB outer combined limit before Docker's former 30-GiB cap could return
an OOM error to the controller. Task 16 now caps its container at 24 GiB, leaving
the host controller headroom to survive and report a runaway tool. Full-corpus
RLM jobs reserve 28,672 MiB and cannot run two at once within the 48-GiB
machine-wide allowance.

Recorded Tier-4 Tasks 16, 17, and 17b now require Docker isolation and fail
before model calls rather than silently falling back when Docker is inaccessible.
This boundary is necessary because model-generated native RDKit code can crash
an in-process Python worker. Standalone development may still use the local
fallback, and the remaining RLM tasks use local execution directly. Those local
workers have an 8,192-MiB address-space limit, with a temporary 4,096-MiB limit
around model-generated Python so the controller retains error-reporting
headroom. Both values are recorded in the experiment files and resource trace.
The outer 30-GiB combined-memory monitor remains the final safeguard.

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
worker and all its host child processes. For Docker RLM work it additionally
reads the container's Linux memory-control counter, which includes container
processes and container-charged cache. The hard job limit applies to host and
Docker memory added together, rather than overlooking a container because it is
owned by the Docker daemon.

Pressing Ctrl-C once now signals every active worker, terminates its isolated
process tree, removes only the Docker containers bearing that attempt's unique
label, and records the interrupted attempt. If a machine stops too abruptly to
record that transition, the next launch can recover the still-marked running
attempt without deleting its history.

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
- peak host process-tree RAM, peak Docker memory, their combined peak, and
  whether the limit was reached.

SwissAI currently does not bill us per token, so those jobs record zero provider
cost rather than an absent cost field. W&B settings now also identify the exact
job, repetition, provider, question parallelism, context size, and positive
cardinality.

The runner recalculates the spending ceiling before each queued job starts,
using actual costs for completed attempts and estimates for unfinished work. If
the revised total exceeds the ceiling, no further job is launched; the untouched
jobs remain pending and already-running work finishes cleanly. This prevents a
systematic model-specific underestimate from silently propagating through the
whole queue.

The adapter reads both ordinary dictionary summaries and the current W&B
`Summary` object. A full 10-question live run exposed that the latter does not
provide the dictionary `.items()` method; a compatibility test now reproduces
that interface so successful scientific work cannot fail during final metric
serialization.

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
| Claude-Haiku-4.5 | CHF 417.50 |
| GPT-5-mini | CHF 96.33 |
| **Full benchmark** | **CHF 743.72** |

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

The exact current test count is reported by the required full-suite check in
[`running_benchmark.md`](running_benchmark.md). Optional skips cover environments
that are not required for the benchmark.

## 13. Human chemistry review support

The merged project also contains a standalone `human_eval/` application for
specialist review of Tier-4 prospective questions. It includes instructions,
study design, assignment handling, restoration/export tools, tests, and a
bundled cleaned dataset for the standalone reviewer package.

This application is separate from the automated six-model benchmark. Running
the automated experiments does not complete expert route validation; the human
review still needs assigned chemistry reviewers and its own recorded outputs.

See [`human_eval/README.md`](../../human_eval/README.md).

## 14. SwissAI quota enforcement

The first broad GLM LLM launch showed that SwissAI applies a shared limit of 15
request starts per minute to one research credential. Four launcher workers can
contain up to sixteen simultaneous LLM questions, so memory-safe concurrency
alone was not sufficient. Twenty-three v18 jobs received HTTP 429 before any
answer was recorded, and the launcher was stopped.

The final full experiment is consequently v19. SwissAI request starts using the
same credential are now spaced 4.25 seconds apart through a local lock shared by
all worker processes. A provider-rejected request waits for `Retry-After` and is
retried at most twice. This policy is used by the one-shot, CodeAct, and native
RLM clients, is generated into the experiment files, and has deterministic
tests. The v18 results remain diagnostics and are not mixed with v19.

## 15. Final Claude choice and CodeAct response allowance

The first v19 GLM LLM phase produced 214 complete jobs, but 63 later requests
reached the five-minute provider deadline, three received provider 5xx errors,
and four active jobs were interrupted when the launcher was stopped. These are
diagnostic records; no v19 work is mixed into the final result directory.

Claude Sonnet 5 was replaced by the pinned `anthropic/claude-haiku-4.5` model.
Haiku supports reasoning and tool use at half Sonnet's list-token prices, which
reduces the full six-model estimate from CHF 1,161.22 to CHF 743.72. The combined
full and matched-cardinality estimate is now CHF 814.70.

The CodeAct response allowance ultimately increased from 2,048 to 30,000
tokens. The 8,192-token intermediate check was enough for GPT-5-mini but still
truncated Haiku actions on Task 16. Claude also rendered Python actions as
`execute_python` or `execute_code` XML and sometimes appended a proposed answer
to the same turn. The controller now accepts those two exact wrappers, executes
the action before scoring a normal-turn answer, and explicitly prohibits
copying the already-preloaded reaction rows into generated code.

The final v25 Haiku Task-16 x500 job completed all ten questions, made 44 model
calls and 34 tool executions, hit the 30,000-token bound zero times, cost CHF
1.239, and obtained 1/10 exact match with macro-F1 0.195. This is the evidence
for retaining the much cheaper Haiku model and the larger bounded allowance.

## 16. SwissAI one-shot concurrency

The first final-style Qwen Tier-1 check ran four questions concurrently. Nine
of ten returned correctly, but one exceeded the five-minute provider deadline,
so the job failed rather than writing partial metrics. A health request returned
in 1.5 seconds, and all ten benchmark questions subsequently completed in 36
seconds with concurrency one. A broader GLM attempt then returned 5/6 answers
on three hard Task-2 jobs before one request in each batch timed out. Full
benchmark v27 therefore uses one-question parallelism for every SwissAI LLM
model. This changes throughput, not prompts, sampling, or inference limits.

The first serialized GLM Task-2 x500 request still timed out at both five and
ten minutes when its request reserved 30,000 output tokens. With an otherwise
identical 4,096-token allowance it returned in 33 seconds and used 42 output
tokens. Full benchmark v28 applies that bound to SwissAI LLM requests only;
OpenRouter LLM, CodeAct, and RLM retain their separately recorded limits.

## 17. GLM CodeAct scheduling bound

The final v28 GLM LLM phase completed all 300 jobs and 1,000 answers. The first
GLM Task-16 CodeAct x500 check subsequently completed one nine-turn trajectory
but then encountered repeated 300-second provider timeouts while reserving
30,000 output tokens per turn. That completed trajectory used 5,272 output
tokens in total across all nine turns. Full benchmark v29 therefore records an
8,192-token per-turn CodeAct limit for GLM only. Other models retain the
30,000-token CodeAct limit. The v28 LLM artifacts remain unchanged and are the
authoritative GLM LLM results. An 8,192-token v29 comparison with two concurrent
questions also returned neither initial response before the five-minute
deadline and was stopped. A v30 comparison serialized requests at 4,096 tokens,
but consecutive legitimate CodeAct turns hit the output ceiling. Full benchmark
v31 therefore combines serialization with 8,192 output tokens per GLM CodeAct
turn. V31 is used for GLM CodeAct and RLM if its release cell completes.

## 18. CodeAct retry and workflow deadline alignment

The v31 GLM release cell exposed a mismatch between two existing safety limits:
three 300-second provider attempts plus 2- and 4-second backoffs require at
least 906 seconds, while most CodeAct question workflows stopped at 900 seconds
(and Task 5 stopped at 600). The outer deadline therefore cancelled the final
retry before it could receive its full allowance. Full benchmark v32 explicitly
records `RXNHAYSTACK_CODEACT_WORKFLOW_TIMEOUT_SECONDS=1800` for every CodeAct
job. All 34 CodeAct scripts read and report this value; their historical
standalone defaults remain unchanged. Per-request, tool, turn, and memory limits
are unchanged.

## 19. GLM CodeAct output allowance after the broad run

The first 66 completed jobs in the v32 serialized GLM CodeAct run returned
1,106 model turns. Of those, 93 turns (8.4%) ended at the 8,192-token output
limit, spread across 16 jobs. That is frequent enough to alter the scientific
result rather than merely prevent pathological output. Full benchmark v33
therefore gives GLM the same 30,000-token per-turn allowance as every other
CodeAct model while keeping GLM questions serialized. The 300-second request,
1,800-second question-workflow, eight-turn reasoning, two-answer-attempt,
60-second generated-tool, and 4,096-MiB generated-tool limits remain in place.

## 20. GLM CodeAct midpoint release check

The v33 Task-16 x500 serial release cell did not return its first model answer:
all three 300-second attempts timed out with a 30,000-token reservation. This
was an API timeout, not an HTTP 429 rate-limit response. Full benchmark v34
records a practical 16,384-token GLM CodeAct midpoint, doubling the truncating
8,192 setting while remaining well below the unschedulable 30,000 setting. It
retains serial questions and every existing workflow, retry, tool, turn, and
memory boundary. V34 is a release check until that demanding cell completes; it
is not authorization to launch the broad GLM phase.

## 21. Safe SwissAI quota division across clusters

The SwissAI limiter originally coordinated worker processes through a locked
timestamp file in node-local `/tmp`. That remains the correct mechanism for
one node, but two clusters using the same credential cannot see each other's
lock and could each attempt the full 15-request-per-minute account allowance.

The execution environment now accepts
`RXNHAYSTACK_SWISSAI_HOST_REQUESTS_PER_MINUTE_CAP`. The effective request rate
is the lower of this machine-level cap and the rate recorded in the experiment
description, so the override can only make execution more conservative. The
launcher records the cap in `metadata.json`; it is deliberately not part of the
scientific run identity because it changes waiting time rather than prompts,
model settings, or answers. Invalid and non-positive caps stop before a model
request. Two hosts can, for example, be assigned six requests per minute each,
leaving three requests per minute of headroom under the shared limit of 15.

This mechanism divides quota; it is not a distributed lock. Separate nodes on
the same cluster must also receive explicit shares unless their workers run in
one allocation and share the same local limiter state.

## What has not been hidden or simplified away

- The three API/W&B credential values are currently required before any selected
  part of the full experiment starts, even when only one provider is selected.
- Separate machines keep separate local SQLite completion records. W&B and the
  shared control room provide the common view; local database files must not
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
9eab451  Use Claude Haiku and raise CodeAct output limit
24f583f  Accept Claude CodeAct tool syntax
711f9f6  Execute CodeAct tools before proposed answers
e4b2959  Prevent CodeAct context duplication
5b0358c  Raise CodeAct turn limit for hard tasks
06afdd6  Support Claude execute-code actions
```

The current documentation revision supersedes the first execution guide. Use
`git log --oneline` to see its final commit after this revision is approved and
pushed.

## 22. The launcher now terminates a completely hung model subprocess

A DeepSeek RLM job stopped producing model events while four HTTPS sockets
remained in `CLOSE-WAIT`. Its 1,800-second RLM cutoff could not help because
that cutoff is evaluated between iterations and the provider call never
returned. Resource sampling continued for more than eight hours, confirming
that this was a live but stalled process rather than a memory termination.

The runner now accepts `--max-run-seconds`. This is enforced by the parent
launcher, outside the model SDK and task script. At the boundary it terminates
the complete process group, records `wall_time_limit_exceeded` in the resource
trace and metadata, marks the attempt failed, and permits other selected jobs
to continue. The resumed full benchmark uses 21,600 seconds (six hours): the
largest jobs contain ten questions at a 30-minute per-question ceiling, leaving
one additional hour for finalization and cleanup. Focused tests exercise both
the watchdog termination and the persisted failed-attempt record.

## 23. All machines now share a read-only experiment control room

Manually copied status counts became unreliable once work was divided among
`liacpc14`, `liacpc15`, Jed, and Kuma. The control room reads each machine's
SQLite ledger and publishes a sanitized, compressed snapshot to one W&B team
project. It merges by immutable run and attempt identity, so copied ledgers are
not double-counted and independent duplicate execution is highlighted.

The interactive page shows the model/method completion matrix, active and stale
work, reporting machines, failure classes, calls, tokens, recorded cost, and
peak memory. A concise Markdown view is generated from the same data. Neither
view can mutate experiments. Prompts, responses, raw errors, credentials,
commands, and local paths are explicitly excluded from uploaded records.

See [`experiment_control_room.md`](experiment_control_room.md) for the exact
per-machine update and viewing commands.
