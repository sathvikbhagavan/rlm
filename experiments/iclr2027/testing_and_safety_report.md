# Testing and safety changes before the ICLR 2027 runs

This report explains what changed after the real-model and Docker tests, why the
testing took several hours, and which limits will protect the full benchmark.
It is intended for Amin, Sathvik, and anyone checking the experiment before API
credit is spent.

For launch commands, read [`running_benchmark.md`](running_benchmark.md). For the
broader engineering history, read [`what_changed.md`](what_changed.md).

## Current status

- The final experiment descriptions are full benchmark v32 and
  matched-cardinality v7.
- The full benchmark contains 6,300 jobs and has a planned ceiling of
  CHF 743.72.
- The matched-cardinality experiment contains 1,450 jobs and has a planned
  ceiling of CHF 70.98.
- The combined planned amount is CHF 814.70. These are conservative planning
  figures, not predictions of the final invoice.
- All locally recorded ICLR preparation and diagnostics are conservatively
  bounded at CHF 35.91 through the successful v25 Haiku check. This charges the
  four interrupted Haiku checks at their full planned ceilings even though they
  were stopped early. Adding it to both planned experiments gives CHF 850.61.
- The exact USPTO raw and cleaned files, all six model identifiers, all three
  credentials, and the Docker image have been verified on Amin's machine.
- After the Claude compatibility correction, the complete automated suite
  reports 431 passed and 10 skipped tests; the focused final checks report 21
  passed.

The v16 through v27 directories contain calibration and diagnostic work. The
v28 directory contains the authoritative completed GLM LLM phase plus the
failed 30,000-token concurrent, 8,192-token concurrent, and 4,096-token serial
CodeAct release checks. Full benchmark v32 uses the isolated 8,192-token serial
GLM CodeAct specification and aligns the CodeAct workflow and request-retry
deadlines.

## Why the final testing took so long

The slow part was observing real Tier-4 behavior, not running ordinary unit
tests. RLM allows a model to write and execute Python over the reaction corpus.
Some generated programs performed large graph searches, allocated tens of
gigabytes, created child processes, or ran until a safety limit stopped them.

The final GPT-5-mini full-corpus Task-16 diagnostic ran for about 90 minutes. It
completed seven questions and entered the eighth before being stopped. Across
the completed and partial work it made 55 completed RLM iterations, spent about
75 minutes in generated tools and 11 minutes in model calls, and reached
24,575.9 MiB of Docker memory and 25,268.5 MiB combined memory. Thirteen tool
iterations reached approximately 300 seconds. This was useful evidence: it
showed that the system could contain the work, but that a complete trajectory
also needed its own time boundary.

Earlier Qwen diagnostics exposed the memory problem. One generated computation
held roughly 23 GiB beyond five minutes. Another reached the outer 30-GiB job
limit before Docker's former 30-GiB limit could return an out-of-memory error to
the RLM. Synthetic tests were then used to prove the corrections without paying
for more model calls: a sleeping process group was terminated at its deadline,
and a 25-GiB allocation was OOM-killed while the same container remained usable.

## Changes made as a result of the tests

### Docker memory is now visible

Docker processes belong to the Docker daemon and are not descendants of the
Python benchmark worker. Measuring only the worker's process tree therefore
missed most RLM memory. Each attempt now labels its containers uniquely and
publishes their Linux cgroup paths. The launcher records host process-tree,
Docker, and combined memory every 0.2 seconds. The hard job limit applies to the
combined value.

### Interruption and cleanup are exact

When the launcher is interrupted, it terminates the active worker process group
and removes only containers with that attempt's unique label. Successful,
failed, interrupted, and retried attempts remain in the SQLite history. The
cleanup tests confirmed that unrelated containers are not selected.

### Generated Docker programs have a real deadline

The earlier Docker execution path could wait forever for `docker exec`. It now
runs model-written Python under an in-container process-group timeout. On
timeout, the generated program and its multiprocessing children are terminated,
an error is returned to the RLM, and the container remains available for the
next reasoning turn.

### Docker OOM happens before the host controller is endangered

Task 16 now limits its Docker container to 24 GiB. Its full-corpus worker
reserves 28 GiB for scheduling and has a 30-GiB combined hard limit. This
ordering lets Docker kill an inefficient generated process and report it to the
controller before the outer monitor must terminate the entire job.

### Model response limits are compatible with hidden reasoning

A GPT-5-mini RLM trial produced five responses billed at exactly 2,048 output
tokens but containing no visible action: mandatory hidden reasoning had consumed
the allowance. OpenRouter RLM calls now use a 4,096-token total response limit
and `low` reasoning effort. SwissAI RLM calls use 2,048 tokens and explicitly
disable the server's hidden-thinking channel.

### Complete RLM trajectories are bounded

The RLM retains its 30 reasoning iterations and two recursion levels. In
addition, every benchmark RLM question has a 1,800-second cutoff checked between
turns. At the cutoff, the controller requests one answer-only response from the
work accumulated so far, records that finalization, and advances to the next
question rather than failing and restarting the surrounding job.

An in-progress generated block retains its separate 300-second limit, so the
cutoff may be observed up to roughly five minutes after the 30-minute mark. The
answer-only model call can add a small amount of additional time.

### Timeout and error reporting is end-to-end

The RLM core emits whether a completion was finalized because of the trajectory
cutoff. The resource trace now preserves that flag. `metrics.json` summarizes it
as `results.rlm_timeout_finalizations`. RLM iteration trace records also preserve
`had_error` and `code_block_count`, so failed tool turns are visible without
storing prompts, generated code, or answers in the resource trace.

### Claude caching and budget checks were tightened

Claude CodeAct and RLM calls use a stable OpenRouter session and explicit
provider-side prompt caching. Cache reads, writes, and actual cost are recorded.
The experiment generator uses the observed Claude CodeAct cost plus headroom.
The experiment descriptions are also rejected if their planned total exceeds
their declared CHF ceiling. Before each new job, the launcher replaces available
estimates with recorded actual costs and rechecks the remaining work. This is a
rolling launch guard, not a provider-account limit: already-running calls can
finish, and the two experiment files have separate ledgers.

### Matched-cardinality settings were reconciled

The matched GPT experiment still contained the old 2,048-token RLM setting. It
now uses the same 4,096-token, low-reasoning, 1,800-second configuration as the
full experiment. Qwen uses the SwissAI 2,048-token/no-hidden-thinking path.

## Effective limits in the final experiment

| Area | Effective limit | What happens at the boundary |
| --- | --- | --- |
| Full planned cost | CHF 1,250 declared; CHF 743.72 currently planned | Validation rejects excess work; the launcher rechecks recorded actual plus remaining estimated cost before each job. |
| Matched planned cost | CHF 100 declared; CHF 70.98 currently planned | Same rolling check in its separate ledger. Together the declarations leave CHF 150 for all diagnostics under the CHF 1,500 project limit; CHF 35.91 is conservatively accounted for already. |
| Active-worker memory | 49,152 MiB total scheduling allowance | A worker waits until its declared reservation fits. |
| LLM parallelism | 4 questions per worker | Further questions wait. |
| CodeAct parallelism | 2 isolated questions per worker, except 1 for GLM | Further questions wait. GLM is serialized because concurrent x500 requests did not schedule reliably. |
| RLM parallelism | 1 question per worker | Questions run sequentially within the task job. |
| SwissAI request starts | 15 requests/minute per credential, spaced by 4.25 seconds across local worker processes | Further request starts wait on a credential-specific local lock. Different co-authors use different credentials. |
| SwissAI rejected request | At most 2 retries | An HTTP 429 that produced no model answer waits for the provider's `Retry-After` duration and is retried visibly. |
| LLM worker memory | 2–4 GiB reserved; 4–8 GiB hard limit | The launcher terminates a worker exceeding its combined hard limit. |
| CodeAct worker memory | 4–6 GiB reserved; 8–12 GiB hard limit | Same combined-memory enforcement. |
| RLM worker memory | 8/10/28 GiB reserved for 100/500/full context; 16/20/30 GiB hard limit | Same combined-memory enforcement. |
| CodeAct response | 30,000 output tokens per model turn, except 8,192 for GLM | The provider response is truncated at the recorded bound. GLM's 30,000-token concurrent request failed after repeated timeouts; 8,192 with two concurrent questions also failed to return either initial response; and 4,096 serially truncated consecutive legitimate turns. V31 isolates 8,192 tokens with one GLM question at a time. |
| LLM/CodeAct reasoning | `low` in Tier 1; `high` in Tiers 2–4 | This preserves the collaborator's original task settings. SwissAI disables only its separate hidden-thinking channel. |
| SwissAI LLM response | 4,096 output tokens per request | Keeps large-context requests schedulable; 500 reaction indices fit within the bound. OpenRouter LLM limits are unchanged. |
| CodeAct reasoning loop | 8 tool/reasoning turns, then at most 2 answer-only attempts | Further code is not executed; the controller asks for the final answer. |
| CodeAct generated tool | 60 seconds and 4,096 MiB | Its complete child process group is stopped and a clean namespace is restored. |
| CodeAct provider request | 300 seconds; at most 2 timeout retries | A timed-out request is retried with recorded backoff; other errors are not silently retried. |
| CodeAct question workflow | 1,800 seconds in the final experiment | This contains three 300-second request attempts plus backoff and still leaves time for successful reasoning/tool turns. Standalone scripts retain their historical 600/900-second defaults unless this recorded setting is supplied. |
| RLM provider request | 300 seconds | The request fails visibly rather than hanging forever. |
| RLM response, SwissAI | 2,048 output tokens; hidden thinking disabled | The visible response remains bounded. |
| RLM response, OpenRouter | 4,096 total output tokens; `low` reasoning | Hidden and visible output share this bound. |
| RLM reasoning loop | 30 iterations and 2 recursion levels | Iteration exhaustion triggers one answer-only response. |
| RLM question trajectory | 1,800 seconds, checked between turns | One answer-only response is requested and the cutoff is counted in `metrics.json`. |
| Docker RLM generated block | 300 seconds | The in-container process group is terminated and the RLM receives the error. |
| Task-16 Docker memory | 24 GiB | Docker OOM-kills the generated process while retaining host-controller headroom. |
| Local RLM process/tool memory | 8,192/4,096 MiB address space | Runaway local allocations raise inside the isolated tool boundary. |
| Full-context RLM concurrency on a 62-GiB machine | Effectively one 28-GiB-reserved worker under the 48-GiB scheduling allowance | A second full-context RLM waits. |

Do not increase these values during the final benchmark without versioning the
experiment description. A different limit changes the evaluated inference
procedure.

## Results of representative real-model tests

| Model and method | Wall time | Recorded CHF | Main observation |
| --- | ---: | ---: | --- |
| Qwen Task-16 CodeAct x500 | 18.1 min | 0 | 6/10 exact; macro-F1 0.736. |
| Gemini Task-16 CodeAct x500 | 7.0 min | 0.486 | 4/10 exact; macro-F1 0.458. |
| GPT-5-mini Task-16 CodeAct x500 | 10.2 min | 0.479 | Completed safely; macro-F1 0.136. |
| Claude Task-16 CodeAct x500 | 12.4 min | 5.116 | Prompt caching active; macro-F1 0.150. |
| Four representative one-shot jobs | 24–32 s for GPT/Gemini/Qwen; 8 s for Claude | 0–0.241 each | All completed with nonzero usage and parsable results. |
| GPT-5-mini Task-16 full RLM diagnostic | 90 min before deliberate interruption | about CHF 0.144 at the pinned conversion | Exposed repeated tool timeouts and motivated the trajectory cutoff. |

These are calibration results, not a balanced scientific comparison.

## Completed v18 diagnostics

The broad v16 LLM and CodeAct trials did not need to be repeated. One
single-question, 100-row RLM job was run for each of the six models using
Tier-3 Task 13, repetition 1.

The pilots were run on 13 September 2026 from commit `8ecd2a6`, two at a time.
All six succeeded on their first attempt and answered the question exactly.

| Pilot | Calls | Wall time | Actual CHF | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek-V4-Flash Task-13 RLM x100 | 4 | 39.57 s | 0 | 359.3 MiB |
| GLM-5.2 Task-13 RLM x100 | 1 | 28.07 s | 0 | 368.6 MiB |
| Qwen3.5-397B Task-13 RLM x100 | 5 | 23.04 s | 0 | 357.6 MiB |
| Gemini-3.7-Flash Task-13 RLM x100 | 2 | 7.03 s | 0.005554 | 359.9 MiB |
| Claude-Sonnet-5 Task-13 RLM x100 | 3 | 9.03 s | 0.038804 | 358.4 MiB |
| GPT-5-mini Task-13 RLM x100 | 3 | 12.04 s | 0.002442 | 358.8 MiB |
| **Total** | **18** | **under 2 min elapsed** | **0.046800** | **368.6 MiB maximum** |

They verify that every exact model can complete the RLM request path, that v18
recorded normal results and usage, and that the two provider transports produce
compatible artifacts. They remain calibration evidence; v19 uses a separate
result directory because the provider quota policy changed before any complete
benchmark phase.

Every run recorded calls, tokens, cost, score, a W&B URL, resource usage, and an
explicit zero for `results.rlm_timeout_finalizations`. No iteration error,
memory-limit event, cancellation, or timeout finalization occurred. The actual
CHF 0.046800 total was well below the planned CHF 0.218817 upper estimate.

These results verify provider and recording compatibility on a short Tier-3
case. They do **not** prove that the time, turn, response-length, or memory
limits are neutral on the hardest Tier-4 full-corpus questions. Those limits
remain explicit computational budgets and their boundary events must be
reported in the final results.

The first attempted v18 GLM LLM phase then started requests faster than
SwissAI's credential-wide quota of 15 per minute. Twenty-three jobs received
HTTP 429 and no GLM LLM job completed. The launcher was stopped immediately.
This was a transport-capacity failure, not a scored model result. In v19,
request starts sharing one SwissAI credential are spaced 4.25 seconds apart
across worker processes. A rejected request is retried at most twice after the
provider's stated delay. The rejected request has no model response and the
retry therefore does not resample or replace a completed trajectory.

The v19 GLM LLM phase subsequently established that request-start pacing alone
does not bound the number of slow requests already in flight. It preserved 214
complete jobs, while 63 jobs reached the 300-second provider deadline and three
received provider 5xx errors; four more were interrupted when the launcher was
stopped. These remain diagnostics. The final launch must use lower SwissAI
in-flight concurrency and first demonstrate that the timeout rate is acceptable.

The final v25 GLM release check used `--max-parallel 1`: two Tier-1 jobs covering
20 question trajectories completed in 43.5 and 44.6 seconds. All 20 responses
were recorded with no 429, timeout, or provider error; peak memory was 328.4 MiB
and the x100/x500 macro-F1 scores were 0.880/0.773. This supports one SwissAI LLM
job at a time on each credential.

The equivalent Qwen check showed that four-question internal concurrency was
too high for the 397B endpoint: 9/10 x100 questions returned, but one exceeded
the 300-second request deadline, which correctly failed the incomplete job. A
single tiny endpoint request returned in 1.5 seconds, and the same ten benchmark
questions then completed serially in 36 seconds with 10/10 answers and macro-F1
1.0. Full benchmark v26 therefore recorded Qwen LLM question parallelism as one.
The exact v26 x100 and x500 launcher jobs
then both passed in 42.7 and 172.5 seconds, with zero timeout/provider errors,
macro-F1 0.967/1.0, and peak memory below 331 MiB.

The subsequent broad GLM phase showed the same long-tail effect on the harder
Tier-2 Task-2 prompts: three jobs returned 5/6 answers before the sixth request
timed out. Full benchmark v27 therefore serializes LLM questions for all three
SwissAI models. OpenRouter LLM jobs retain four-question concurrency.

Serialization alone did not make GLM Task-2 x500 schedulable: the first request
timed out at both five and ten minutes while reserving 30,000 output tokens. The
identical prompt returned in 33 seconds when its allowance was 4,096 tokens and
used only 42. Full benchmark v28 therefore records a 4,096-token SwissAI LLM
allowance. CodeAct retains 30,000; RLM retains its separate 2,048 setting.

Claude Sonnet 5 was replaced by the pinned Claude Haiku 4.5 model at half the
input and output list prices. The first 8,192-token Haiku checks exposed two
provider-format variants and showed that otherwise valid code was still being
truncated. The controller now accepts fenced Python and Claude's explicit
`execute_python`/`execute_code` wrappers, always executes a normal-turn action
before considering a proposed answer, and tells every model not to duplicate
the preloaded `lines` context.

The final 30,000-token v25 check completed all ten Task-16 x500 questions in
9.7 minutes with 44 model calls, 34 isolated tool executions, zero output-limit
hits, an 860.1-MiB memory peak, and CHF 1.239 actual cost. It achieved 1/10
exact match and macro-F1 0.195. GPT-5-mini's matched 8,192-token check completed
in 6.2 minutes for CHF 0.292, with 1/10 exact match and macro-F1 0.244. These
checks establish that Haiku is usable and that 30,000 is a ceiling rather than
routine per-turn consumption.

A second Task-16 stress-test sweep is not necessary. The v25 Haiku cell is the
first final measurement. The v20 GPT-5-mini cell remains calibration because it
predates the final prompt; its v25 repetitions should be launched only by their
assigned owner.

## Final release check

The following checks were completed before and after the six small pilots. They
must be repeated after any code or experiment-description change and before any
larger selection:

1. confirm a clean checkout at the pushed `main` commit;
2. run the complete test suite;
3. regenerate in check-only mode and validate both experiment files;
4. run `plan` and compare the dataset fingerprints;
5. verify no pilot has missing calls, tokens, cost, result, or resource metrics;
6. verify `results.rlm_timeout_finalizations` is present, even when its value is
   zero;
7. stop and inspect any authentication error, repeated provider failure,
   unparsed answer, memory termination, or unexpected timeout finalization.
