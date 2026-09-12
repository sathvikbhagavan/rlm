# Running the ICLR 2027 benchmark

This is a walkthrough for launching RxnHaystack. It explains what we
need to run, the order, how different we can divide the work, what to check after each stage, and what to send back to the coding assistant before continuing.

We first run a few small tests with real models, check their cost, memory use, outputs,
and logs, and only then choose how much parallelism each machine can safely use.

## What we need to run

There are three experiment-description files, but only two contain new work:

| Experiment file | Purpose | Separately tracked jobs | Estimated cost with safety margin |
| --- | --- | ---: | ---: |
| `full-campaign.toml` | Complete 100-question benchmark with six models | 6,300 | CHF 939.25 |
| `matched-cardinality-campaign.toml` | Corpus-size/cardinality control on the 65 questions with ordinary reaction-row positives | 1,450 | CHF 70.98 |
| `baseline-campaign.toml` | Optional independent GPT-5-mini reproduction | 1,050 | CHF 67.32 |

The TOML filenames are already used by the runner. Below, we call
them experiment files or sets of benchmark runs.

### What is a “separately tracked job”?

The experiment files describe groups of settings compactly. For example, one
entry can request five repetitions. Before execution, the runner turns those
five repetitions into five uniquely named jobs ending in `r01` through `r05`.
Each job has its own status, cost, output files, and retry history. This is what
the command-line tool currently calls an “expanded run.”

A job is not always one benchmark question. It launches one task script, and
that script may answer between one and ten related questions. The counts work
out as follows.

For the full benchmark, each model has seven method/context settings:

```text
2 LLM settings + 2 CodeAct settings + 3 RLM settings = 7 settings
```

The 100 benchmark questions are grouped into 30 task scripts. For one model:

```text
tracked jobs: 30 task scripts × 7 settings × 5 repetitions = 1,050 jobs
question trajectories: 100 questions × 7 settings × 5 repetitions = 3,500 trajectories
```

Across six models:

```text
6 models × 1,050 jobs per model = 6,300 tracked jobs
6 models × 3,500 trajectories per model = 21,000 question-level trajectories
```

For the matched-cardinality experiment, Tier 1 contributes one task script and
10 questions, each evaluated at five settings. Tier 2--3 contribute 20 task
scripts and 55 questions, each evaluated at seven settings: five corpus sizes
with K=1, plus K=5 and K=20 at N=5,000. For one model:

```text
tracked jobs before repetition: (1 Tier-1 script × 5 settings) + (20 Tier-2/3 scripts × 7 settings) = 145
tracked jobs: 145 × 5 repetitions = 725 jobs

question trajectories before repetition: (10 Tier-1 questions × 5 settings) + (55 Tier-2/3 questions × 7 settings) = 435
question trajectories: 435 × 5 repetitions = 2,175 trajectories
```

Across the two matched-cardinality models:

```text
2 models × 725 jobs per model = 1,450 tracked jobs
2 models × 2,175 trajectories per model = 4,350 question-level RLM trajectories
```

The full experiment already includes every GPT-5-mini setting from the optional
baseline reproduction. We should not run `baseline-campaign.toml` as well unless
we explicitly want a second, independent GPT-5-mini reproduction.

The estimated total for the full and matched-cardinality experiments is CHF
1,010.23. This includes a 25% safety margin for the six-model benchmark and a
50% safety margin for the matched-cardinality experiment. The margins protect
against models producing longer trajectories than GPT-5-mini did historically;
they are not expected charges.

### Models and methods

The full benchmark compares:

- SwissAI: DeepSeek-V4-Flash-0731, GLM-5.2, and Qwen3.5-397B-A17B;
- OpenRouter: Gemini-3.7-Flash, Claude-Sonnet-5, and GPT-5-mini.

As calculated above, each full-benchmark model accounts for:

```text
30 task scripts × 7 method/context settings × 5 repetitions = 1,050 tracked jobs
100 questions × 7 method/context settings × 5 repetitions = 3,500 question-level trajectories
```

The planned costs, including the safety margin, are CHF 0 for the three SwissAI
models, CHF 229.89 for Gemini, CHF 613.04 for Claude, and CHF 96.33 for
GPT-5-mini. Each matched-cardinality model accounts for 725 jobs and 2,175
question-level trajectories, as calculated above; only its GPT-5-mini half is
expected to incur API charges.

For each model we run:

- LLM at 100 and 500 reaction rows;
- CodeAct at 100 and 500 reaction rows;
- RLM at 100 rows, 500 rows, and the full corpus;
- five independent repetitions of every setting.

The matched-cardinality experiment uses Qwen3.5-397B-A17B and GPT-5-mini. It
holds K=1 while increasing the corpus through 100, 500, 5,000, 50,000, and full.
At N=5,000 it additionally evaluates K=5 and K=20 on Tier 2--3. Tier 1 remains
in the K=1 comparison because some questions have only one positive. Tier 4 is
omitted because its answers are routes and chains, avoiding the need to define a
different notion of cardinality for those questions.

## Stage 0: Assign what model each of us runs

Write down which person owns each model or group of jobs. A simple two-person split could be:

| Person | Example responsibility |
| --- | --- |
| Amin | Selected closed models and the GPT-5-mini matched-cardinality runs |
| Sathvik | Selected open models and the Qwen matched-cardinality runs |

We should make the final split after learning each machine's available RAM and the API limits seen in the small real-model tests.

Before continuing, send the coding assistant:

```text
Person/machine name:
Available RAM:
Models assigned:
```

The assistant should return an assignment table containing exact run-name
patterns.

## Stage 1: prepare each machine

Every user should begin from the same clean commit on `main` and run commands
from the repository root. If the checkout has edits, stop and reconcile them first.

Install exactly the recorded Python environment:

```bash
uv sync --frozen
uv run --frozen pytest -q
```

`--frozen` tells `uv` to use the recorded dependency versions without silently
changing `uv.lock`. The tests should finish with the same pass/skip counts on
each machine. Report any failure before making API calls.

### Prepare the isolated Tier-4 RLM environment

Tier-4 Tasks 16, 17, and 17b execute model-written Python and native RDKit code.
Their recorded benchmark jobs require Docker isolation; they deliberately stop
before making model calls if this user cannot reach Docker. First check access:

```bash
docker info
```

If this reports permission denied, ask the machine administrator to grant this
user Docker access. On a conventional Linux installation the administrator may
use `sudo usermod -aG docker "$USER"`; the user must then sign out and back in.
Membership in the Docker group is a privileged machine-level permission, so it
should follow the host's normal policy.

Build and test the pinned chemistry sandbox from the repository root:

```bash
docker build -t rlm-sandbox -f tier4/Dockerfile.sandbox tier4
docker run --rm rlm-sandbox python -c \
  'import dill, numpy, rdkit, requests; print(numpy.__version__, rdkit.__version__)'
```

The build must complete successfully, and the test must print NumPy `1.26.4`
and RDKit `2025.09.6`. Report the `docker info` result and those two versions.

### Prepare or verify the dataset

```bash
uv run --frozen \
  --with-requirements dataset/requirements.txt \
  python -m dataset.prepare --verify-only
```

This checks that both the raw and cleaned USPTO files exist and exactly match the
expected row counts and SHA-256 fingerprints. A SHA-256 fingerprint is a compact
identifier for the exact file contents: a one-character difference produces a
different value.

If the files are absent, rerun the command without `--verify-only`. It downloads
the raw file to a temporary location, verifies it, cleans it with the recorded
RDKit version, then atomically moves completed files into place. See
`dataset/README.md` for expected values and non-default data locations.

### Prepare credentials

Each person uses their own credentials. Store them outside the repository and
restrict their permissions:

```bash
chmod 600 ~/.openrouter_api_key \
  ~/.swissai_research_api_key \
  ~/.wandb_api_key
```

At present, a full-experiment launch requires all three credential values even
when a user selects only one provider. The launch command supplies them with
the three `--secret-file` arguments shown below. This makes missing credentials
visible before execution, but means every participating machine needs all three
files.

### Check the experiment descriptions

Run these commands:

```bash
uv run --frozen python experiments/iclr2027/generate_full_campaign.py --check
uv run --frozen python experiments/iclr2027/generate_matched_cardinality_campaign.py --check
uv run --frozen rxnhaystack validate experiments/iclr2027/full-campaign.toml
uv run --frozen rxnhaystack validate experiments/iclr2027/matched-cardinality-campaign.toml
uv run --frozen rxnhaystack plan experiments/iclr2027/full-campaign.toml \
  --select 'full-gpt-5-mini-tier1-task1-llm-x100-r01'
```

Here is what each command does:

1. The first two reconstruct the expected experiment text in memory and compare
   it with the checked-in TOML file. They do not call a model or rewrite a file.
   Silence means the generated file is current; an error means the generator and
   experiment file disagree.
2. The next two read every setting and check names, repetitions, estimated cost,
   memory limits, commands, and uniqueness of all resulting job IDs. They do not
   inspect the dataset or call an API.
3. The final `plan` command performs the checks that depend on this machine. It
   verifies the Git state, exact dataset contents, and required executables, then
   prints the one selected example job without running it or spending money.

The example after `--select` is an exact job name. It asks the tool to show only
the first GPT-5-mini Tier-1 LLM job, which keeps the output short enough to
inspect by eye.

The final command should print:

- the Git commit identifier shared by all users;
- raw dataset: 137,261 rows and SHA-256
  `8ba53c8aa5a513bdef651fb06f3ed1392ac3056c5f472087bdc52e820440b791`;
- cleaned dataset: 122,456 rows and SHA-256
  `9f9b2e71676e3e8f132b495b3fc62e2fdec01bc5b43f7be3dc09ef279c351b14`;
- exactly one selected job, including its model, method, corpus size, estimated
  price, memory allowance, and question parallelism.

Why must the Git checkout be clean? The result record stores the commit
identifier so we know exactly which code produced an answer. If a tracked source
file has an uncommitted edit, the commit alone no longer identifies the code, so
the paid benchmark refuses to begin. Result directories under `artifacts/` are
intentionally excluded from Git and do not count as source edits.

Before continuing, send the coding assistant the complete output of the five
commands, plus:

```text
Machine/user:
Git commit shown by plan:
Test result:
Raw dataset rows and SHA-256:
Cleaned dataset rows and SHA-256:
Credential files present with mode 600: yes/no
```

Proceed only when the assistant—or a second person checking manually—confirms
that every machine reports the same commit and dataset fingerprints.

## Stage 2: run the free infrastructure checks

```bash
uv run --frozen rxnhaystack run experiments/iclr2027/smoke.toml
uv run --frozen rxnhaystack run experiments/iclr2027/concurrency-smoke.toml
uv run --frozen rxnhaystack run experiments/iclr2027/metrics-adapter-smoke.toml
```

These use deterministic local programs rather than paid models. Together they
test dataset access, recording of results, resumption, simultaneous work,
metrics, output files, and memory measurement.

Afterward, run:

```bash
uv run --frozen rxnhaystack status experiments/iclr2027/smoke.toml
uv run --frozen rxnhaystack status experiments/iclr2027/concurrency-smoke.toml
uv run --frozen rxnhaystack status experiments/iclr2027/metrics-adapter-smoke.toml
```

Report the three status outputs. Do not proceed if any local check failed or if
expected `metadata.json`, `metrics.json`, and `resource-trace.jsonl` files are
missing under `artifacts/`.

## Stage 3: try a few representative real-model jobs

In this guide, “profile” means running a small, deliberately chosen set of real
jobs before the full benchmark. The purpose is not to draw scientific
conclusions. It is to measure how the actual provider, model, method, and machine
behave so we can choose safe parallelism and detect broken outputs while only a
few API calls are at risk.

We need to cover:

1. a small LLM job, to check the simplest prompt and token accounting;
2. a CodeAct job, to check isolated agents and tool timing;
3. a difficult full-corpus Tier-4 RLM job, to check long trajectories and peak
   memory;
4. at least one SwissAI model and every paid model before releasing that paid
   model's full set of jobs.

In CodeAct, the retrieved rows are preloaded in each isolated Python tool as a
list named `lines`. During trial review, confirm the model uses `lines` rather
than copying the full `<context>` block into generated Python code. The prompt
repeats this instruction immediately after the question so it remains visible
even when the context is very long.

CodeAct permits at most eight reasoning/tool turns, followed—only when needed—by
up to two answer-only attempts. At that boundary the model is explicitly told not
to run more code; if it nevertheless returns code, that code is not executed and
the final correction is requested once more. Tier-4 answers may contain several
lines; the controller recognizes
the `ANSWER:` marker and leaves validation of the following task-specific format
to the task's own parser.

Each CodeAct model turn is capped at 2,048 output tokens. The older scripts
requested as many as 30,000, which allowed a malformed action to occupy a
provider for five minutes and could create an outsized paid bill. A 4,096-token
trial still took over three minutes on Qwen, whereas the successful representative
action used 390 tokens. The final cap is
written into every CodeAct job in the experiment file and applied equally to
SwissAI and OpenRouter models.

RLM requests use the same 2,048-token per-call output bound. Live full-corpus
turns normally used 129--860 output tokens; without a request bound, one turn
continued for more than two minutes. The RLM-level limits of 30 iterations and
two recursion levels remain unchanged.

Tier-4 Tasks 16, 17, and 17b require the Docker environment for recorded
benchmark jobs. There is no automatic local fallback during these archival
runs: a missing daemon, image, or user permission is reported before model calls
begin. This matters because native-library failures in model-generated code can
terminate an in-process Python worker. The container instead turns a failed tool
process into an error the RLM controller can record and respond to.

Standalone development runs may still fall back to the local Python environment,
and the other RLM tasks use that environment directly. Launcher-managed local
RLM workers have an 8,192-MiB address-space limit; model-generated Python is
temporarily limited to 4,096 MiB so the controller retains error-reporting
headroom. The separate 30,720-MiB process-tree limit remains the final safeguard.
Both local limits are recorded in `resource-trace.jsonl` when they apply.

Start with one open model, one job at a time:

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

Then repeat the same three method/context types with Gemini, Claude, and
GPT-5-mini. The exact paid job IDs are:

```text
full-gemini-3.7-flash-tier1-task1-llm-x100-r01
full-gemini-3.7-flash-tier4-task16-codeact-x500-r01
full-gemini-3.7-flash-tier4-task16-rlm-xfull-r01

full-claude-sonnet-5-tier1-task1-llm-x100-r01
full-claude-sonnet-5-tier4-task16-codeact-x500-r01
full-claude-sonnet-5-tier4-task16-rlm-xfull-r01

full-gpt-5-mini-tier1-task1-llm-x100-r01
full-gpt-5-mini-tier4-task16-codeact-x500-r01
full-gpt-5-mini-tier4-task16-rlm-xfull-r01
```

Use these after `--select` in the same form as the Qwen examples, and keep
`--max-parallel 1`. A previously successful job is skipped, so these trials
become part of the final dataset rather than being thrown away.

For every trial job, inspect:

- `metrics.json`: calls, input/output tokens, actual cost, total time, and tool
  time;
- `metadata.json`: exact setting, completion state, and peak process-tree RAM;
- `resource-trace.jsonl`: whether RAM grew steadily or spiked during a particular
  RLM iteration or subcall, and whether the expected local RLM memory limit was
  applied;
- `stdout.log` and `stderr.log`: parse failures, authentication errors, 400/429/
  5xx responses, retries, or chemistry-tool errors;
- W&B: the run exists once, has the expected model/provider labels, and contains
  per-question outputs and final metrics.

Before scaling up, all trial jobs should satisfy these checks:

- completion state is successful;
- answers are present and parse into the requested format;
- calls and token counts are nonzero;
- SwissAI jobs record zero provider cost rather than a missing metric;
- paid cost is reasonably close to the estimate; investigate a difference above
  2× before continuing;
- peak RAM stays comfortably below the hard limit—use 80% as the point at which
  we stop and reassess;
- there are no repeated authentication, rate-limit, or provider errors.

Send this report for each trial group:

```text
Git commit:
Machine and total RAM:
Exact command and selected job IDs:
Succeeded / failed:
Model and provider:
Calls:
Input / output tokens:
Actual CHF cost:
Wall time:
Peak process-tree RAM:
Highest percentage of the job's hard RAM limit:
RLM environment (Docker or local) and local-limit trace event, if applicable:
Any 400, 401, 429, 5xx, timeout, parsing, or chemistry errors:
W&B link or run name:
Anything surprising in the answers or logs:
```

The coding assistant should compare the report with the stored estimates and
recommend either one conservative `--max-parallel` value for the whole assigned
model or separate values for LLM, CodeAct, and RLM work. Separate values require
launching those methods as separate selections, as explained below. Do not
automatically raise parallelism merely because the machine has unused CPU;
provider limits and memory are the relevant constraints.

## Stage 4: understand job selection and divide the benchmark

`--select` filters jobs by their unique names. The `*` character is a wildcard:
it means “any sequence of characters.” For example:

```text
full-qwen3.5-397b-*
```

means every full-benchmark job whose name begins with
`full-qwen3.5-397b-`. A “non-overlapping” assignment means no job name can match
patterns assigned to two different people. Otherwise both may pay for the same
job because separate machines have separate completion records.

The safest division is one model per person or machine:

```text
full-deepseek-v4-flash-*
full-glm-5.2-*
full-qwen3.5-397b-*
full-gemini-3.7-flash-*
full-claude-sonnet-5-*
full-gpt-5-mini-*
matched-qwen3.5-397b-*
matched-gpt-5-mini-*
```

If a model must be divided further, job names follow this structure:

```text
full-{model}-{tier}-task{task}-{method}-x{context}-r{repetition}
```

Thus `full-claude-sonnet-5-tier4-*` selects Claude Tier 4, while
`full-gpt-5-mini-*-rlm-*` selects GPT-5-mini RLM work across tiers. Multiple
`--select` arguments are combined.

To use different parallelism for each method, run three non-overlapping
commands, for example:

```text
full-qwen3.5-397b-*-llm-*
full-qwen3.5-397b-*-codeact-*
full-qwen3.5-397b-*-rlm-*
```

The LLM command can usually use the highest value, CodeAct a lower value, and
RLM the lowest. Trial measurements—not these general expectations—determine the
actual values.

Before launch, the coding assistant should produce and check a table like:

| Owner | Machine | Exact run-name pattern(s) | Number of jobs | Estimated CHF | Starting parallelism |
| --- | --- | --- | ---: | ---: | ---: |
| Amin | … | … | … | … | … |
| Sathvik | … | … | … | … | … |

The job counts across owners must sum to 6,300 for the full benchmark and 1,450
for the matched-cardinality experiment, with no duplicate job IDs. Ask the
assistant to compute this from the experiment files rather than counting by
hand.

## Stage 5: launch the assigned work

Replace `RUN-NAME-PATTERN` below with the exact pattern from the checked
assignment table. In other words, do not type those capitalized placeholder
words literally.

```bash
uv run --frozen rxnhaystack run experiments/iclr2027/full-campaign.toml \
  --select 'RUN-NAME-PATTERN' \
  --max-parallel 4 \
  --secret-file OPENROUTER_API_KEY=~/.openrouter_api_key \
  --secret-file SWISSAI_RESEARCH_API_KEY=~/.swissai_research_api_key \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

Use the parallelism recommended after the trial jobs, not necessarily `4`.
`--max-parallel` controls the number of separate worker processes. Inside each
worker, LLM can have up to four questions in flight, CodeAct up to two isolated
agents, and RLM one question at a time.

The runner also limits the sum of estimated RAM for active workers to 48 GiB.
Typical allowances are 2--4 GiB for LLM, 4--6 GiB for CodeAct, and 8--16 GiB for
RLM. Each worker has an additional hard RAM limit. Do not run the full benchmark
and matched-cardinality experiment simultaneously on the same host unless their
combined memory use has been explicitly budgeted: the two runner processes do
not share a memory counter.

We keep five repetitions as five jobs. Do not replace them with one request
using `n=5`; CodeAct and RLM make later calls based on earlier responses, so five
answers from one initial request are not five independent trajectories.

Recommended order:

1. run the three open models first;
2. review their first complete tier and all paid-model trials;
3. release GPT-5-mini and Gemini;
4. release Claude while watching its early Tier-4 RLM cost closely;
5. run matched Qwen and matched GPT on another machine or after the main work.

## Stage 6: monitor, stop safely, and resume

In another terminal on each machine:

```bash
uv run --frozen rxnhaystack status experiments/iclr2027/full-campaign.toml
```

Stopping the launcher does not erase successful jobs. Running the same command
again skips them. To retry jobs already recorded as failures:

```bash
uv run --frozen rxnhaystack run experiments/iclr2027/full-campaign.toml \
  --select 'RUN-NAME-PATTERN' \
  --retry-failed \
  --max-parallel 2 \
  --secret-file OPENROUTER_API_KEY=~/.openrouter_api_key \
  --secret-file SWISSAI_RESEARCH_API_KEY=~/.swissai_research_api_key \
  --secret-file WANDB_API_KEY=~/.wandb_api_key
```

If a machine stopped while jobs were recorded as running, add
`--recover-running`. The incomplete attempt is retained and a new attempt is
created.

Press Ctrl-C once to stop a launcher. It signals each active worker, terminates
that worker's isolated process tree, records the attempt as interrupted, and
leaves successful jobs untouched. Wait for the command to return to the shell;
do not press Ctrl-C repeatedly. If the machine or Python process was killed too
abruptly to finish that bookkeeping, the next launch reports those jobs as
running and `--recover-running` performs the recovery described above.

SwissAI requests use an explicit 300-second request deadline because its large
models can take longer than the client library's 60-second default. There are no
hidden client retries multiplying that deadline. For diagnosis only, the value
can be overridden for a launch by setting
`RXNHAYSTACK_SWISSAI_REQUEST_TIMEOUT_SECONDS`; do not change it between
benchmark jobs without recording and justifying the deviation.

SwissAI chat calls use the endpoint's native `enable_thinking=false` option.
Live trials showed that its separately returned hidden thinking channel could
consume a 30,000-token allowance without producing final content, and
LlamaIndex cannot pass that channel to CodeAct. CodeAct retains its visible
multi-turn reasoning and tool loop. This is a transport setting, not a change to
the chemistry prompt, and it must be disclosed as a cross-provider difference.

Stop and report before retrying when:

- authentication fails;
- the same 400 response appears repeatedly;
- repeated 429 responses show the provider limit is being exceeded;
- a worker reaches its RAM limit;
- cost per job exceeds 2× the estimate;
- outputs are empty, unparsable, or obviously answer a different question.

Send a progress report at the end of each working session:

```text
Owner and run-name pattern:
Git commit:
Successful / failed / running / pending jobs:
CHF spent according to result files:
CHF spent according to provider dashboard:
Peak RAM observed:
Repeated error types and affected job IDs:
Retries already attempted:
W&B runs present and updating: yes/no
Decision requested from the assistant:
```

## Stage 7: collect and verify completed results

Each machine has its own SQLite completion record. Do not overwrite one user's
`ledger.sqlite3` with another's. W&B combines scientific records using the
unique `rxnhaystack_run_id`. Preserve each machine's complete result directory
under an owner-specific name when collecting files centrally.

For every assigned pattern, confirm:

- no pending, running, or unexplained failed jobs remain;
- every success has `metadata.json`, `metrics.json`, `resource-trace.jsonl`,
  `stdout.log`, and `stderr.log`;
- W&B contains each expected run ID exactly once;
- model, provider, method, context size, repetition, and positive cardinality
  agree between the experiment file, local result, and W&B;
- total local cost agrees reasonably with the provider dashboard;
- the exact Git commit and dataset fingerprints are recorded with the archive.

Send the assistant all status outputs, the assignment table, result locations,
and a W&B export or access link. Ask it to produce a completeness report listing
expected, present, duplicate, failed, and missing job IDs before any result is
aggregated.

## Stage 8: turn completed runs into the paper result

Once completeness is confirmed:

1. freeze a read-only copy of raw result files and W&B exports;
2. calculate per-question and aggregate metrics with confidence intervals;
3. verify that aggregation uses exactly 100 main-benchmark questions and the 65
   eligible matched-cardinality questions;
4. generate final tables and plots from scripts, not manual spreadsheet edits;
5. ask a second person to compare plotted counts and labels with the completeness
   report;
6. write the exact models, endpoints, sampling, repetitions, costs, timing,
   memory limits, failures, and retry policy into the paper and appendix;
7. rerun figure generation from the frozen results in a clean checkout;
8. archive code commit, experiment files, dataset fingerprints, raw results,
   figure inputs, and the final paper source together.

The final report to the coding assistant should contain:

```text
Final Git commit:
Owners and completed run-name patterns:
Expected / present / failed / duplicate / missing jobs:
Main question trajectories present:
Matched-cardinality trajectories present:
Total cost by model and provider:
Wall time and peak RAM summaries:
W&B export location:
Local result archive locations:
Known deviations from the planned experiment:
```

Only after that report is clean should we regenerate the submission figures and
write numerical claims into the manuscript.
