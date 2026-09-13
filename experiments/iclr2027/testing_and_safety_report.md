# Testing and safety changes before the ICLR 2027 runs

This report explains what changed after the real-model and Docker tests, why the
testing took several hours, and which limits will protect the full benchmark.
It is intended for Amin, Sathvik, and anyone checking the experiment before API
credit is spent.

For launch commands, read [`running_benchmark.md`](running_benchmark.md). For the
broader engineering history, read [`what_changed.md`](what_changed.md).

## Current status

- The final experiment descriptions are full benchmark v18 and
  matched-cardinality v6.
- The full benchmark contains 6,300 jobs and has a planned ceiling of
  CHF 1,161.22.
- The matched-cardinality experiment contains 1,450 jobs and has a planned
  ceiling of CHF 70.98.
- The combined planned amount is CHF 1,232.20. These are conservative planning
  figures, not predictions of the final invoice.
- The exact USPTO raw and cleaned files, all six model identifiers, all three
  credentials, and the Docker image have been verified on Amin's machine.
- After the final telemetry correction, the complete automated suite reports
  421 passed and 10 skipped tests.

The v16 and v17 directories contain calibration and diagnostic work. They are
deliberately separate from v18 and will not be mistaken for final results.

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
their declared CHF ceiling. This is a planning safeguard; it is not a real-time
stop on the provider account.

### Matched-cardinality settings were reconciled

The matched GPT experiment still contained the old 2,048-token RLM setting. It
now uses the same 4,096-token, low-reasoning, 1,800-second configuration as the
full experiment. Qwen uses the SwissAI 2,048-token/no-hidden-thinking path.

## Effective limits in the final experiment

| Area | Effective limit | What happens at the boundary |
| --- | --- | --- |
| Full planned cost | CHF 1,500 declared; CHF 1,161.22 currently planned | Validation rejects an experiment description whose planned work exceeds the declaration. |
| Matched planned cost | CHF 100 declared; CHF 70.98 currently planned | Same planning check; this is not live provider-account enforcement. |
| Active-worker memory | 49,152 MiB total scheduling allowance | A worker waits until its declared reservation fits. |
| LLM parallelism | 4 questions per worker | Further questions wait. |
| CodeAct parallelism | 2 isolated questions per worker | Further questions wait. |
| RLM parallelism | 1 question per worker | Questions run sequentially within the task job. |
| LLM worker memory | 2–4 GiB reserved; 4–8 GiB hard limit | The launcher terminates a worker exceeding its combined hard limit. |
| CodeAct worker memory | 4–6 GiB reserved; 8–12 GiB hard limit | Same combined-memory enforcement. |
| RLM worker memory | 8/10/28 GiB reserved for 100/500/full context; 16/20/30 GiB hard limit | Same combined-memory enforcement. |
| CodeAct response | 2,048 output tokens per model turn | The provider response is truncated at the recorded bound. |
| CodeAct reasoning loop | 8 tool/reasoning turns, then at most 2 answer-only attempts | Further code is not executed; the controller asks for the final answer. |
| CodeAct generated tool | 60 seconds and 4,096 MiB | Its complete child process group is stopped and a clean namespace is restored. |
| CodeAct provider request | 300 seconds; at most 2 timeout retries | A timed-out request is retried with recorded backoff; other errors are not silently retried. |
| CodeAct question workflow | 900 seconds | The question workflow stops instead of running indefinitely. |
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

## Proposed v18 pilots

The broad v16 LLM and CodeAct trials do not need to be repeated: the subsequent
changes affect RLM limits and telemetry, not their prompts or scoring. The
smallest useful v18 release check is one single-question, 100-row RLM job for
each of the six models using Tier-3 Task 13, repetition 1.

| Pilot | Planned CHF |
| --- | ---: |
| DeepSeek-V4-Flash Task-13 RLM x100 | 0 |
| GLM-5.2 Task-13 RLM x100 | 0 |
| Qwen3.5-397B Task-13 RLM x100 | 0 |
| Gemini-3.7-Flash Task-13 RLM x100 | 0.053619 |
| Claude-Sonnet-5 Task-13 RLM x100 | 0.142984 |
| GPT-5-mini Task-13 RLM x100 | 0.022214 |
| **Total** | **0.218817** |

They verify that every exact model can complete the final RLM request path, that
v18 records a normal result and usage, and that the two provider transports
produce compatible artifacts. They are genuine v18 cells: successful jobs are
kept and skipped during the later full launch, so their time and money are not
wasted.

The likely duration is several minutes per job, but no defensible tight estimate
exists because no successful RLM job has yet been recorded in the final setup.
Each pilot has one question. Its configured upper bound is roughly 35 minutes
plus the final answer call: 30 minutes before cutoff and up to five minutes for
an already-running tool. Sequential worst case is therefore roughly 3.5 hours
for all six. At `--max-parallel 2`, the scheduling reservations permit two at a
time, making the configured worst case roughly 1.75 hours in three waves. The
expected time should be much lower for a 100-row, single-question Tier-3 task.

These six pilots are **recommended but not mathematically necessary**. The
timeout and metrics path can be—and is—tested deterministically without paid
models. Skipping the pilots saves at most CHF 0.22, but moves discovery of a
model-specific response or artifact problem into the large run. Because the
pilots count toward the final experiment, the prudent choice is to run them.

A second full-corpus Task-16 stress-test sweep is not necessary. Running Task 16
once for Qwen, Gemini, Claude, and GPT would have a planned paid cost of
CHF 1.863 total, could take hours, and would repeat failure modes already used to
design the safeguards. Such jobs should now be treated as ordinary production
cells, not disposable pilots.

## Final release check

Before starting the six small pilots or any larger selection:

1. confirm a clean checkout at the pushed `main` commit;
2. run the complete test suite;
3. regenerate in check-only mode and validate both experiment files;
4. run `plan` and compare the dataset fingerprints;
5. verify no pilot has missing calls, tokens, cost, result, or resource metrics;
6. verify `results.rlm_timeout_finalizations` is present, even when its value is
   zero;
7. stop and inspect any authentication error, repeated provider failure,
   unparsed answer, memory termination, or unexpected timeout finalization.
