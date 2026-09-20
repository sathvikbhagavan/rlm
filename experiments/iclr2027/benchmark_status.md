# ICLR 2027 benchmark execution status

This is the shared coordination record for the final benchmark. It says who
owns each part, where it is running, what is complete, and what must happen
next. Update this file whenever a phase starts, stops, or materially changes.

Last consolidated: **2026-09-21 00:38 Europe/Zurich**

Dashboard/assignment implementation at consolidation: `17cd68b`

## Current execution and held-work queue

This table supersedes older point-in-time counts later in this document. The
dashboard intentionally keeps five scientific sections; transport changes,
Slurm shards and repairs are folded into their parent experiment.

| Work | Where / provider | Current state at consolidation | Next action |
| --- | --- | --- | --- |
| DeepSeek full RLM | Transfer from `liacpc14` to Jed / paid OpenRouter | The local launcher was stopped audibly on September 20 at 12:42. Its active non-Docker cell is preserved as interrupted. The continuation has 60 successes, 1 interrupted failure and 218 pending; exactly 195 unfinished non-Docker cells move to Jed and 24 Docker cells stay local. | Repartition the 195 non-Docker cells into 12 disjoint one-worker Jed shards for a sub-10-hour target. |
| GLM full RLM | Jed / paid OpenRouter plus earlier SwissAI results | Running. Folded dashboard: 317 succeeded, 61 failed, 1 running, 2 stale, 69 pending. | Let the disjoint Jed shards finish, then make a failed-only repair set. Keep 45 Docker cells for `liacpc14`. |
| Qwen matched cardinality | Jed / paid OpenRouter | Running under Amin, not Sathvik: 169 succeeded, 4 running and 552 pending after folding the newest two-cell repair. | Stop the four launchers through their tested shutdown path, rebuild the exact unfinished set, and repartition it into about 20 disjoint one-worker shards for a sub-10-hour target. |
| Oracle-predicate Qwen | Jed / paid OpenRouter | 70/75 succeeded. Five full-corpus cells failed in the latest repair. | Inspect their Jed stderr/artifacts, classify the common cause, then retry only the five failed cells. |
| Task-16 prospective decomposition | `liacpc14` / Claude OpenRouter and Qwen | Pilot succeeded; 1/30 jobs complete. The remaining 29 Docker-required, 28--30 GiB jobs are in the explicit local queue. | Run locally one at a time after the 14 Qwen/Gemini repairs. |
| Qwen/Gemini full RLM repair | `liacpc14` / OpenRouter | Qwen Task-16 r02 attempt 2 is the final attempt and has completed 6/10 questions. Its prediction artifact is advancing, so recent stall mail was false. The remaining five Qwen cells are scheduled for exactly one attempt each regardless of r02's outcome; Gemini's eight cells have not started. At 00:29 the shared account had `$408.63`, while this key had `$55.13` before its `$1,200` cap. | Do not retry r02 or any of the other five Qwen repairs after their scheduled attempt. Preserve failures as unsuccessful cells and continue. Revise the cost estimate before releasing Gemini. |
| GLM Docker RLM | `liacpc14` / OpenRouter | 45 jobs held. | Run locally after the prospective and Qwen/Gemini Docker repairs, unless paper priority changes. |
| DeepSeek CodeAct repair | Not launched | 57 failures: 48 timeouts, 6 context overflows, 2 interrupted attempts and 1 other provider error. | Retry transient/interrupted cells; first correct and pilot the six context-overflow configurations. |
| GLM CodeAct | Jed, later | One pilot failed; 299 jobs held. | Diagnose/revise the release pilot before broad execution. |

The recurring local Docker queue, which must appear in every execution-status
handoff until empty, is: (1) six Qwen repairs (**r02 final attempt active; five
scheduled once with no retries**), (2)
eight Gemini repairs, (3) 29 prospective Task-16 jobs, (4) 24 remaining
DeepSeek Docker jobs, and (5) 45 GLM Docker jobs. Only one large Docker RLM cell
runs at a time.

The durable controller `rxn-local-docker-queue-v1` is armed at Git `57e5bba`.
It waits for the active Qwen attempt, then advances through those five phases
in order. Every never-attempted cell receives at most one attempt; scientific
failures are preserved and never selected for retry. Each phase receives a
task-notify watcher and dashboard publication. The controller waits rather
than spending if either the shared OpenRouter account or the API key has less
than USD 20 available. Its durable state and log are
`~/.local/state/rxnhaystack/local-docker-queue.json` and
`~/.local/state/rxnhaystack/local-docker-queue.log`.

Completed since the previous consolidation: GPT matched cardinality is 725/725;
Claude oracle-predicate is 75/75; Claude full CodeAct is 300/300; Claude full
RLM is 450/450 across non-Docker and Docker rows; and GPT full RLM is 450/450.

## How to read the counts

- **Verified** means read directly from that machine's SQLite ledger or from a
  machine report containing ledger counts.
- **Reported** means a collaborator supplied the count, but the ledger has not
  yet been inspected during consolidation.
- **Unknown** is deliberately not treated as complete.
- **Done** should mean a successful job with required metrics and artifacts,
  not merely a request that returned or a W&B run that closed.
- Each full model has 1,050 tracked jobs: 300 LLM, 300 CodeAct, and 450 RLM.
  RLM consists of 405 jobs that do not require Docker and 45 Tier-4 jobs that
  do require Docker.

The machine-local SQLite ledgers remain the authoritative result records. This
document is the authoritative assignment and coordination record. W&B is a
useful common view, but its projects are divided by task and a closed W&B run
does not by itself prove that scientific metrics were produced.

## Assignment by person and machine

| Person | Machine | Assigned work | Must not duplicate |
| --- | --- | --- | --- |
| Amin | `liacpc14` | Active paid DeepSeek continuation; Docker-required catch-up and prospective study | Do not duplicate active Jed shards |
| Sathvik | `liacpc15` | Qwen and Gemini full benchmarks | DeepSeek, GLM, Claude, GPT, or the new Qwen matched Jed shards |
| Amin and Sathvik | Jed CPU cluster | GLM RLM, Qwen matched cardinality, oracle-predicate controls, and approved non-Docker rescue shards | Keep every shard disjoint from active `liacpc14` work |
| Amin | Kuma | Claude and GPT-5-mini full benchmarks; GPT matched-cardinality; GPT repair pilots | SwissAI models and Sathvik's models |

No owner should launch another machine for the same run names without first
changing this table and checking both ledgers for overlap.

## Full-benchmark status

### Execution queue agreed on September 19

The remaining work is assigned by execution requirement rather than by model:

1. **Jed, non-Docker parallel work:** move the unfinished paid-OpenRouter
   DeepSeek RLM continuation to disjoint Slurm shards; replace the slow SwissAI
   GLM non-Docker RLM continuation with exact `z-ai/glm-5.2` OpenRouter shards;
   retry the four Claude Task-14 RLM cells; run the GPT-5-mini failed-only
   matched-cardinality recovery through direct OpenAI; run Qwen matched
   cardinality through paid OpenRouter; and complete the Qwen/Claude
   oracle-predicate model control through OpenRouter. Every shard must have a
   non-overlapping selector list, an isolated ledger/artifact directory, a hard
   API budget, dashboard reporting, and task-notify monitoring.
2. **`liacpc14`, Docker work:** after the active GPT recovery releases memory,
   retry the six missing Qwen and eight missing Gemini Docker RLM cells, run the
   remaining Task-16 prospective-decomposition cells through OpenRouter, and
   later run the 45 GLM Docker RLM cells. Only one full-corpus Docker RLM cell
   should be active at a time unless a measured pilot justifies more.
3. **CodeAct:** GLM CodeAct remains a separate later phase. DeepSeek CodeAct's
   57 failed cells also require a classified failed-only repair rather than an
   indiscriminate rerun.

Neither Jed nor Kuma currently provides Docker on compute nodes. Apptainer is
installed, but the benchmark has no validated Apptainer backend; Docker-required
work therefore stays on `liacpc14` until that changes.

| Model | Owner / machine | LLM | CodeAct | RLM | Confidence |
| --- | --- | ---: | ---: | ---: | --- |
| DeepSeek V4 Flash | Amin / `liacpc14` | 300 succeeded | Finished: 243 succeeded, 57 failed | Original ledger: 171 succeeded, 10 failed/interrupted, 269 pending across all 450; the 279 unfinished cells are now in a paid-OpenRouter continuation with one pilot running | Verified locally at 12:35 September 19 |
| GLM 5.2 | shared / Jed | 300 succeeded in reusable v28 results on `liacpc14`; an archive reporter now publishes them to the dashboard | Release pilot failed; remaining 299 held | The slow SwissAI non-Docker continuation is to be replaced by disjoint paid-OpenRouter `z-ai/glm-5.2` shards on Jed; 45 Docker jobs remain assigned to `liacpc14` | GLM LLM verified directly from the v28 ledger; RLM transition pending |
| Qwen 3.5 | Sathvik / `liacpc15` | 300 succeeded | 300 succeeded | 444 succeeded; six cells have no successful archived record | Verified from Sathvik's checksummed 2,086-result archive |
| Gemini Flash | Sathvik / `liacpc15` | 300 succeeded | 300 succeeded | 442 succeeded; eight cells remain failed in the older dashboard source | Verified from Sathvik's archive and prior dashboard source |
| Claude Haiku | Amin / Kuma and `liacpc14` | 300 succeeded | 300 exact cells succeeded; the missing v34 Task-16 x500 cell completed locally with exit code 0 and valid metrics | Kuma: 401/405 succeeded and 4 failed, assigned to a failed-only Jed retry; local Docker: 45/45 succeeded after the memory-failed cell passed on retry | Dashboard and local ledgers verified September 19 |
| GPT-5-mini | Amin / Kuma and `liacpc14` | 300 succeeded | 300 succeeded | Original non-Docker: 384/405 succeeded; direct recovery: 19 succeeded, one running, one pending; direct-OpenAI Docker: 45/45 succeeded | All three ledgers verified and folded in the dashboard |

Dashboard interpretation was tightened at 20:40. Transport-specific GPT and
DeepSeek completion records are folded into their original full-benchmark run
IDs. `Attempted` no longer implies success; the terminal-coverage bar is green
for successes and red for failures. Execution state (`running`, `paused`, `not
started`, or complete) is separate from reporter freshness.

### DeepSeek details on `liacpc14`

- At 12:35 on September 19, the two six-RPM SwissAI launchers were stopped.
  Their active attempts were preserved as intentionally interrupted, not lost.
- All 279 cells without a successful source result were mapped one-to-one into
  `iclr2027-deepseek-paid-openrouter-continuation-v1`. It uses the exact paid
  model `deepseek/deepseek-v4-flash-0731`, never the free variant. The live
  OpenRouter key had USD 598.05 remaining before launch. The measured-token
  estimate is about CHF 5.61; the deliberately conservative experiment-file
  estimate is CHF 11.56 and its hard budget is CHF 40.
- A single paid pilot is running. If it succeeds, the remaining cells release
  automatically, one at a time while the GPT recovery also uses this host.
- The paid pilot passed and eight cells have succeeded for CHF 0.58; one is
  running and 270 remain pending. This is intentionally one-at-a-time while
  the GPT recovery shares `liacpc14`. A dedicated `task-notify` watcher was
  attached at 19:19 after discovering that only the dashboard reporter—not an
  email watcher—had initially been configured.

- CodeAct finished with 243 successful and 57 failed jobs.
- Active phase: RLM, one worker, six SwissAI request starts per minute at most.
- Formerly stalled run:
  `full-deepseek-v4-flash-tier2-task3-rlm-x100-r05`.
- The CodeAct failures are retained for a later failed-only review/retry. Do
  not retry them while they would compete with the active RLM phase.
- RLM started automatically at `2026-09-16T21:13:47Z`. Thirty-four jobs
  succeeded and 415 were pending at consolidation. The current job had been
  marked running for more than eight hours, but its last model/iteration event
  was at `2026-09-17T03:21:37Z`. Four HTTPS sockets were in `CLOSE-WAIT`; only
  resource samples continued. The configured 1,800-second RLM timeout is
  checked between iterations and therefore did not interrupt this hung client
  call. It was cleanly interrupted and retained as failed with return code 143.
  Commit `6f99e12` adds and tests a launcher-owned wall-time boundary outside
  the model SDK. The phase resumed with `--max-run-seconds 21600`; the next job,
  `full-deepseek-v4-flash-tier2-task3-rlm-x500-r01`, started successfully.
- The 57 CodeAct failures were not blindly retried because they would consume
  the same SwissAI quota as the active RLM phase and include non-retryable
  configurations. Their first-error taxonomy is: 20 workflow timeouts, 17 API
  timeouts, 11 generic timeouts, six context-window 400 errors, one provider
  status error, and two interrupted attempts. Retry transient classes only
  after RLM; change the six context-overflow configurations before retrying.

### GLM details on Jed

- Jed preparation passed: 435 tests passed and 10 skipped before the new host
  cap was added; dataset checksums and infrastructure smokes matched.
- Jed has no Docker. Its GLM scope is therefore 300 CodeAct jobs and 405
  non-Docker RLM jobs.
- Jed and `liacpc14` use the same SwissAI credential. Commit `781c0f6` adds the
  execution-only host cap. `liacpc14` is capped at six requests per minute and
  Jed must also be capped at six, leaving three requests per minute of
  headroom below the shared limit of 15.
- Slurm job `66534424` is running on `jst043` with the six-request-per-minute
  host cap. After about nine hours it had processed 28/405 jobs: 22 succeeded,
  six failed, one was active, and 376 had not started.
- The active run in that report was
  `full-glm-5.2-tier2-task2-rlm-xfull-r04`, question 3/6.
- Successful runs recorded 23.35 million tokens, 1,452 calls, and W&B records.
  There were no HTTP 429 errors and no RLM trajectory-timeout finalizations.
  Four jobs failed on HTTP 500 and two on API request timeout; retain them for
  failed-only retry after the pending set finishes.
- `full-glm-5.2-tier4-task16-codeact-x500-r01` failed its release check, so the
  remaining 299 CodeAct jobs correctly stayed held. Its exact error still
  needs to be recorded before deciding whether to revise or retry the pilot.

### GPT and Claude details on Kuma

- The sole local Claude Docker failure,
  `full-claude-haiku-4.5-tier4-task16-rlm-x500-r01`, succeeded on attempt two.
  It completed all ten questions in 29m58s, cost CHF 3.84, and peaked at only
  530.65 MiB combined memory. Local Claude Docker coverage is therefore 45/45.
- The 21 failed GPT non-Docker cells are being rerun through direct OpenAI,
  not through the OpenRouter path that returned policy refusals. The exact set
  is two Task-13 memory aborts, four Task-14 failures, and fifteen Task-15
  failures. Task 13 receives an 8 GiB local-tool allowance. A Task-15 pilot
  succeeded; the Task-13 memory pilot is running, after which the remaining
  cells release automatically. Estimated cost is CHF 1.74 with a CHF 20 cap.
- Both Task-13 memory cells succeeded, peaking at 4.40 and 4.36 GiB. At 19:20,
  19/21 recovery cells had succeeded for CHF 1.39, one Task-15 full-corpus cell
  was running, and the final cell was pending. Its dedicated email watcher was
  attached at 19:19.
- The sole absent Claude CodeAct v34 cell is
  `full-claude-haiku-4.5-tier4-task16-codeact-x500-r01`. A successful v25
  result exists, but v34 added an explicit workflow-timeout setting. Rather
  than silently transplant it, the exact v34 cell was launched locally at
  20:37 with a CHF 5 notification ceiling and is running.
- Kuma's four terminal Claude RLM failures have not been converted into
  successes: 401 succeeded plus four failed equals 405 terminal outcomes.
  They remain a four-cell failed-only retry set.

- Claude was explicitly left untouched by the GPT repair work. Its current RLM
  ledger count still needs a fresh read.
- The original GPT missing-usage audit found no recoverable scientific result:
  19 full jobs representing 68 trajectories and 505 matched jobs representing
  985 audited trajectories were classified irrecoverable. Original ledgers
  were unchanged by the read-only audit.
- The first staged GPT Task-14 pilot made three requests. All returned empty
  choices with HTTP 403 policy-refusal errors, so the script stopped before
  Task 15 and both memory pilots, as intended.
- Kuma's read-only diagnosis found high-confidence upstream OpenAI policy
  refusal rather than an OpenRouter balance or key failure. The preserved
  errors identify `error_type=refusal` and `provider_code=invalid_request` and
  link to OpenAI policy guidance. OpenRouter's key endpoint remained healthy
  with USD 217.29 of its USD 500 limit available.
- All three generation-metadata and content lookups returned HTTP 404, so cost,
  tokens, request IDs, and the retrospective router pipeline remain unknown.
  Future calls on the review branch request and sanitize OpenRouter routing
  metadata.
- GPT repair branches are held and unmerged:
  `fix/gpt-missing-usage-recovery` at `655f835` and the combined
  `review/gpt-retry-definitions` at `012be90`.
- The updated review branch stops explicit 403 and other non-transient 4xx
  responses after one request, disables hidden SDK retries, retains bounded
  retries for transient failures, and passes 471 tests with 10 skipped. It is
  not yet merged.
- Do not submit the 19-job or 505-job retries through the same upstream OpenAI
  route. OpenRouter currently lists Azure as another provider for the same
  `openai/gpt-5-mini` model slug. One provenance-recorded Azure-only pilot is a
  reasonable next test; changing provider routing must be explicit in the
  retry definition and final methods.

## Matched-cardinality status

| Model | Owner / machine | Status | Next action |
| --- | --- | --- | --- |
| GPT-5-mini | Amin / Kuma | 220/725 succeeded; 505 irrecoverable failed jobs remain held | Diagnose the new OpenRouter 403, validate the repair, then retry only genuinely authorized work |
| Qwen 3.5 | Sathvik / `liacpc15` | Unknown | Obtain exact ledger counts and confirm whether it has started |

No other model belongs in the matched-cardinality experiment.

## Docker-required work

Each model has 45 RLM jobs for Tier-4 Tasks 16, 17, and 17b.

| Machine | Docker status | Consequence |
| --- | --- | --- |
| `liacpc14` | Available and tested | Claude and direct-OpenAI GPT Docker are 45/45 complete; unfinished DeepSeek Docker cells are included in the paid continuation |
| `liacpc15` | Not yet recorded here | Sathvik must report whether Qwen/Gemini Docker jobs ran |
| Jed | Unavailable | Cannot run the 45 GLM or DeepSeek Docker jobs |
| Kuma | Unavailable; Apptainer is not an approved silent substitute | Cannot run the 45 GPT or Claude Docker jobs |

Docker catch-up must receive an explicit non-overlapping assignment after the
ordinary 405-job RLM phases are reconciled. Do not launch all held jobs merely
because `liacpc14` supports Docker; first import or preserve the other
machines' ledgers and confirm which run IDs remain.

## Paper blockers, not merely missing leaderboard cells

1. **The central causal claim is still confounded.** The oracle-predicate
   control is implemented and has cleared its complete parity and deterministic
   execution gates, but its model cells have not run and matched cardinality is
   incomplete. Without completed model results, corpus size, answer
   cardinality, chemical diversity, and abstraction difficulty remain
   entangled.
2. **The obvious competing systems are absent.** There is no completed
   retrieval-augmented or non-recursive map-reduce baseline. A six-model
   replication does not answer whether RLM recursion is necessary.
3. **Prospective validity is unresolved experimentally.** Task 16 target-name,
   target-structure, and target-structure-plus-class conditions have not been
   run. An author chemist approved the three final-step class descriptions on
   September 17, correcting the lactam description to "Boc deprotection of
   secondary amine," so the label-review gate is now cleared.
4. **The final benchmark matrix is incomplete.** Several RLM phases are active,
   GLM CodeAct is blocked, GPT retries are policy-blocked, transient failures
   remain, and Docker coverage is unresolved. The paper can tolerate disclosed
   failures, but not ambiguous denominators or silent missing cells.
5. **Statistics and efficiency evidence are not yet consolidated.** Final
   analysis must include uncertainty across questions/tasks and calls, tokens,
   latency, tool time, cost, memory, and failure rates.
6. **Claims and figures need tightening even if no more experiments finish.**
   Context size has different operational meanings across methods; prospective
   ground truth is non-exhaustive; and broad “chemical reasoning” language must
   be narrowed to what the tasks establish.

The first three items are acceptance-critical experimental blockers. Items four
through six can be addressed partly through transparent reporting and narrower
claims, but they cannot be ignored.

### Control audit findings on September 17

- A five-configuration oracle-predicate control is feasible using existing
  answer-free chemistry evaluators: Tier-3 Tasks 6, 10, and 23 and Tier-4 Tasks
  13 and 14. The initial design covers 16 questions with Qwen and Claude at
  x100, x500, and full scale. Only oracle cells need new inference because the
  normal cells already exist.
- A universal retrieval baseline is not well-defined for this benchmark.
  Tiers 1--3 ask for exhaustive sets, many queries have no natural query
  reaction, and Tier-4 chains can connect individually dissimilar reactions.
  Task-specific SMARTS retrieval would encode the oracle predicate. The paper
  should explain this precisely rather than merely saying RAG is future work.
- Flat non-recursive map-and-union is a relevant baseline for row-separable
  Tiers 1--3, but it is expensive: three questions, two models, five repetitions,
  and 500-row chunks require 7,350 model calls. Implement and mock-test it now,
  then run only one 245-call pilot before deciding whether the publication set
  is worth the deadline time and cost.
- At least one completed full-context DeepSeek RLM trajectory used direct
  Python scanning and no `llm_query`/recursive subcall. The paper must report
  observed root/subcall counts and must not describe every successful RLM run
  as recursive decomposition without trajectory evidence.
- Prospective decomposition applies only to Tier-4 Task 16. Tasks 17 and 17b
  are explicit multi-constraint chain retrieval, not prospective synthesis.
- Current Task 16 is not actually name-only: it supplies descriptions that
  disclose route/final-transformation information, two descriptions contradict
  the withheld reactions, and exact target products remain elsewhere in the
  full corpus for four of ten targets. New name/structure/class controls must
  remove every exact-target-product reaction first. Existing Task-16 results
  remain a separately labelled legacy condition.

## Priority through the full-paper deadline

The abstract deadline is September 18 and the full-paper deadline is September
25. The abstract should use claims already supported by completed evidence; it
must not wait for every leaderboard cell. Work that can run unattended should
continue while the acceptance-critical controls below are implemented.

### P0-A: protect and finish work already running

1. Keep DeepSeek RLM on `liacpc14`, GLM RLM on Jed, and the active Qwen,
   Gemini, and Claude RLM phases running. Do not restart whole phases or create
   duplicate ledgers.
2. Obtain exact Qwen, Gemini, Claude, and matched-Qwen ledger counts immediately.
   We cannot plan remaining capacity from “near the end.”
3. Confirm whether Sathvik's SwissAI credential differs from Amin's. If it is
   the same credential, include `liacpc15` in the shared 15-RPM allocation
   before making any further SwissAI launch.
4. Treat the GPT HTTP 403 as an upstream OpenAI policy block. Test at most one
   explicitly routed Azure GPT-5-mini pilot before deciding whether the same
   model can be completed without changing prompts. Do not send the 19 full or
   505 matched retries through the already-refusing OpenAI route.

### P0-B: acceptance-critical controls to validate and complete

The external review ranks these above polishing a six-model leaderboard. Begin
their implementation in parallel with the unattended full runs:

1. **Oracle-predicate control:** a small, representative Tier-3/Tier-4 subset
   comparing normal RLM with validated executable chemistry predicates and a
   deterministic executor ceiling.
2. **Retrieval and map-reduce baselines:** explain why generic top-k retrieval
   is not a neutral comparator for exhaustive-set queries, and validate a
   non-recursive chunk-and-union baseline on representative row-separable
   Tier-1--3 questions.
3. **Prospective-task decomposition:** target name only versus target structure
   versus target structure plus final transformation class for Task 16 only.
   Treat Tasks 17 and 17b as constrained chain retrieval, not prospective
   synthesis.
4. **Matched cardinality:** finish or launch Qwen matched-cardinality and unblock
   GPT matched-cardinality. This is already specified and should not be
   redesigned.
5. **Human validation:** the tested `human_eval/` application exists, but the
   study is not complete. Export prospective false positives, assign available
   chemistry reviewers, and start review as soon as stable candidate outputs
   exist.

Use representative subsets and two contrasting models first. Do not expand a
control to all six models until its small version works and its result changes
the paper's conclusion.

### Overnight implementation assignments: September 17

The following work uses isolated branches and makes no model calls until tests
and leakage audits pass:

| Branch | Work | Launch tonight? |
| --- | --- | --- |
| `feature/oracle-predicate-control` at `7649255` | Five-task oracle prompts, deterministic ceiling, generated experiment and parity/leakage tests | Merged to main at `3646060`; full parity and deterministic gates passed at `b8f6120`; model pilots pending |
| `feature/task16-prospective-decomposition` at `6cc7719` | Correct Task-16 name/structure/class conditions, remove exact-target leakage, fix taxonomy and prepare human-review export | Merged to main at `6a85c01`; label approval encoded at `3546de4`; one Claude class-arm pilot running on `liacpc14` |
| `feature/flat-map-reduce-baseline` at `475d018` | Resumable Tier-1--3 flat mapper/union reducer, artifacts and mocked experiment definition | Shelved on its remote branch by author decision; do not merge or launch before post-submission review |

The prospective definition contains 30 jobs and 90 trajectories, with an
estimated API cost of CHF 2.73 and a CHF 15 ceiling. The map-and-union
definition contains 30 jobs but expands to 7,350 mapper calls because each job
must cover 245 chunks. Real-corpus calibration estimates about 12.5 million
input tokens per job; its Qwen/Gemini paid ceiling is CHF 180 within a CHF 200
budget. This work is now explicitly post-submission: do not launch even the
pilot unless the author reopens it after the ICLR deadline.

The single prospective release pilot
`prospective-claude-haiku-4.5-task16-structure_plus_class-rlm-xfull-r01`
completed successfully on Docker-capable `liacpc14`. It evaluated all three
targets, produced valid metrics and `task16-predictions.json`, synchronized to
W&B, and recorded complete provider and resource accounting. It made 31 calls,
used 814,558 tokens, cost CHF 1.0163, ran for 2,156 seconds, and peaked at
4,074.09 MiB combined host-plus-Docker memory. Macro-F1 and exact match were
both 1/3: one target was exactly solved and two returned no parsed chains. No
timeout or memory boundary was reached. The other 29 prospective jobs remain
pending until their revised cost and machine schedule are approved; the CHF
0.18 estimate materially understated this pilot.

At 00:55 Europe/Zurich on September 18, the local Docker catch-up plan launched
separate serial Claude and DeepSeek workers on `liacpc14`. By 10:26, 29 Docker
jobs had succeeded with no queue failures. Claude completed 23/45: all 15 x100
jobs, all five Task-17 x500 repetitions, and three of five Task-17b x500
repetitions. Those jobs made 845 model calls, used 24,153,936 tokens, recorded
CHF 29.9779, and peaked at 616.8 MiB combined process-tree-plus-Docker memory.
The queue checked the live OpenRouter allowance before every cell and stopped
cleanly at 05:23 when it observed USD 47.70 remaining, below the protected USD
50 reserve. A later read-only check showed USD 42.12 remaining; the key is also
used by other project processes, so the additional balance movement is not
attributed to this stopped queue. The other 22 Claude Docker jobs remain
pending and the queue will resume without repeating its 23 successes after a
budget decision or key-limit increase.

DeepSeek completed all five Task-16 x100 repetitions and Task-16 x500 r01;
Task-16 x500 r02 was active. Its six successful Docker jobs made 1,458 calls,
used 79,873,974 tokens, cost CHF 0.00 through SwissAI, and peaked at 708.2 MiB.
The original DeepSeek worker simultaneously progressed through the non-Docker
queue. Together, DeepSeek RLM advanced from the previously documented 34
successes to 116 successes; two independent cells were active, two failures
were preserved, and 330 remained pending. Both workers share the same local
six-request-per-minute SwissAI limiter, so the second worker overlaps local
computation and waiting but does not increase the machine's account request
rate. At the latest check the machine used 7.5 GiB of 61 GiB RAM with 53 GiB
available, and the active Docker container was healthy.

GPT's 45 Docker cells now run through a distinct, provenance-preserving direct
OpenAI experiment rather than silently changing the transport of the v34 run
IDs. The `.openai_api_key_liac` credential is a valid OpenAI project key: model
access and a minimal paid completion succeeded. Numeric project balance is not
available to that key because both OpenAI billing endpoints returned HTTP 403,
so the experiment enforces its own CHF 30 hard budget and calculates cost from
official GPT-5-mini input, cached-input and output token prices. Commit
`c6c56ef` adds this transport, its generated 45-cell definition and its staged
runner. The complete suite passed 487 tests with 11 skipped.

The direct-OpenAI release cell
`direct-openai-gpt-5-mini-tier4-task17-rlm-x100-r01` passed: five questions,
39 calls, 606,136 tokens, CHF 0.0759, 372 seconds, 462.16 MiB peak combined
memory, complete W&B and artifacts, and no timeout or policy refusal. This
demonstrates that the earlier empty-choice HTTP 403 is specific to the
OpenRouter/upstream route or its account-policy context, not a universal block
on the unchanged benchmark task. The remaining 44 cells released
automatically; Task-16 x100 r01 was active at consolidation. The tmux session
is `rxn-gpt-direct-openai-docker-v1`, using the isolated clone
`/home/amin/rlm/rlm-gpt-direct-openai` and ledger
`artifacts/iclr2027-gpt5mini-direct-openai-docker-v1/ledger.sqlite3` there.

Claude's guarded queue was resumed with an explicit USD 3 reserve after author
approval. It advanced to 24 succeeded, one running and 20 pending; recorded
Docker cost was CHF 30.7043 and OpenRouter reported USD 40.39 remaining. The
queue still checks that allowance before every new cell and will stop before
the reserve rather than repeatedly submit after the account limit is reached.
The relevant Claude and DeepSeek tmux sessions are
`rxn-claude-docker-catchup-v34-resume` and
`rxn-deepseek-docker-catchup-v34`.

The Claude Docker queue ultimately completed 44/45 cells. The sole failure,
`full-claude-haiku-4.5-tier4-task16-rlm-x500-r01`, completed six of ten
questions before model-generated code in the seventh question allocated Docker
memory from about 0.7 GiB to 20.4 GiB in approximately ten seconds. Host RSS
remained about 494 MiB. The launcher correctly terminated the attempt at
20,902 MiB combined usage against its 20,480 MiB limit. Four other repetitions
of the same configuration succeeded at only 529--542 MiB, so this is a
stochastic runaway tool computation rather than normal configuration memory.
The second attempt started at 11:47 Europe/Zurich on September 19 with the same
safety limit; it remained healthy below 505 MiB during the initial audit. Its
tmux session is `rxn-claude-docker-retry-task16-x500-r01`, and a dedicated
`task-notify` watcher will report the terminal result.

DeepSeek's 405-cell non-Docker scope had 150 successes, six retained failures,
one active cell and 248 pending at 11:50 on September 19. The 150 successes
represent 752 question trajectories, 9,695 calls, 273,877,125 input tokens and
10,154,562 output tokens. OpenRouter currently lists the exact
`deepseek/deepseek-v4-flash-0731` at USD 0.04 per million input tokens and USD
0.08 per million output tokens, plus a zero-priced `:free` variant. Applying
the completed per-trajectory usage to the remaining 448 trajectories,
including failed-only retries, estimates USD 7.01 or CHF 5.61; reserve CHF 12
for task-mix variance. A provider switch must use new provenance-preserving run
identities rather than rewriting the active SwissAI attempts.

The shared dashboard merger previously stopped refreshing when a legacy source
contained the running form of an attempt and the current reporter contained
the terminal form of that same attempt. Commit `6d1743b` now reconciles that
normal monotonic state transition while continuing to reject conflicting
terminal records. Thirty-five focused dashboard tests pass. The viewer
successfully regenerated at 11:50 and now includes all reported local, Jed and
Kuma succeeded/running/failed/pending states, including the 45/45 direct-OpenAI
GPT Docker result and the active Claude retry. Sathvik's Qwen/Gemini LLM and
CodeAct ledgers remain explicitly *unreported*, not silently counted as
pending evidence: Amin's viewer receives HTTP 403 for
`sathvikbhagavan-epfl/rxnhaystack-dashboard`. Complete cross-machine coverage
requires that project to grant Amin read access or Sathvik to publish the
snapshots to a mutually readable W&B dashboard project.

The oracle definition contains 150 model jobs and 480 question trajectories,
estimated at CHF 17.90 with a CHF 30 ceiling, plus 15 deterministic jobs and
48 zero-API evaluations. At exact commit `b8f6120`, Jed's complete 122,456-row
audit achieved exact parity for every predicate. This includes all 719 frozen
Task-10 Mitsunobu rows with zero missing and zero additional entries. The
persisted parity report has SHA-256
`7800e871bc56a668e5bc6636a1f8248ab09f3af486c6092effa318c0762d09e1`.
All 15 deterministic jobs then succeeded on their first attempts, with
macro-F1 and exact-match accuracy both 1.0, zero model calls, zero tokens and
zero API cost. Campaign wall time was 46 minutes 22 seconds; maximum traced
run memory was 340.01 MiB and the Slurm step peak was 394.12 MiB. The executor
ledger SHA-256 is
`1c2472d15599ee51b4d69b2c74b435f79c4d10ba80330c8960ecf65e83bbcd2c`.
Both release gates are therefore passed. Model work must still begin with one
exact Qwen x100 repetition and one exact Claude x100 repetition before the
remaining oracle cells are released.

Those two pilots started on `liacpc14` at approximately 16:36 Europe/Zurich:
`oracle-qwen3.5-397b-tier3-task6-x100-r01` and
`oracle-claude-haiku-4.5-tier3-task6-x100-r01`. They are the only selected
oracle model runs completed successfully on their first attempts. Qwen handled
four questions in 369 seconds using 25 calls and 141,401 tokens, scored 0.75
macro-F1/exact match, cost CHF 0.00, and peaked at 378.79 MiB. Claude handled
the same four questions in 51 seconds using eight calls and 53,986 tokens,
scored 0.50 macro-F1/exact match, cost CHF 0.0691, and peaked at 379.80 MiB.
Both produced metrics, resource traces and W&B records without timeout or
memory failure. These scores are scientific outcomes, not release failures:
the pilots establish that each model received and executed the answer-free
helper, while showing that orchestration errors remain even with the chemistry
abstraction supplied. The other 148 runs remain pending. The Qwen pilot shared
the existing machine-wide six-request-per-minute SwissAI limiter with the
DeepSeek worker, so it did not increase `liacpc14` beyond its assigned quota.
Do not duplicate either r01 selector on another machine.

Jed then received the 74 non-duplicating Claude oracle runs. Its required r02
pilot passed with three of four exact answers, nine calls, 49,806 tokens, CHF
0.0665 cost and a synchronized W&B run. The guarded continuation completed 50
of the 74 assigned jobs successfully; five stopped while starting W&B and 19
were not launched because the CHF 30 experiment budget gate refused further
admission. Across the 50 successes Jed recorded 920 calls, 21,024,131 tokens,
CHF 25.2577, and 87 of 167 exact answers. All successes synchronized to W&B;
none hit a workflow, launcher-wall-time or memory limit. Command time was 3
hours 55 minutes and peak traced memory was 930.30 MiB. Including the separate
local r01 result, Claude oracle coverage is therefore 51 succeeded, five
failed before scientific completion, and 19 pending. The reported Jed ledger
SHA-256 is
`6aad4f1180204422ae25cc6d22c6db1e93f157ed50d575d440584e1ae565f834`;
the final report SHA-256 is
`76f1d39ccb8e08d2b37c9b9cfd04ffdc45aec2f42fb42e70f5dccc6f4c3047d1`.

The CHF 17.61 estimate for these 74 jobs was not reliable: completed oracle
trajectories were materially longer than the historical trajectories used for
planning. Do not restart the phase or simply raise the budget. First audit the
exact five W&B failures and 19 budget-stopped run IDs, obtain a task/scale-aware
remaining-cost estimate, and confirm failed-only/pending-only resumption will
skip all 50 successes and the excluded local r01.

The project lead confirms that the reused original ground-truth predicates were
previously reviewed by a chemist. Preserve the reviewer/date/version record for
the paper. Separately, an author chemist approved the three new Task-16
final-transformation class labels on September 17, with the lactam wording
changed from "spirocyclic amine" to "secondary amine." The project lead must
add the reviewer's name to the internal manuscript record.

Separately, Sathvik should queue matched Qwen behind the current full-Qwen RLM
only after reporting his SwissAI key fingerprint and exact current ledger. Kuma
may prepare a staged Azure-only GPT diagnostic using unchanged prompts; one
pilot may run, but the 524-job release remains held.

### P1: complete the comparison without blocking P0 controls

1. Resolve the GLM CodeAct release failure. If it is an infrastructure failure,
   retry one pilot; if it is a stable model/provider limitation, record it and
   avoid spending days forcing all 299 jobs.
2. Run the two provenance-linked GPT memory pilots after the 403 diagnosis.
3. Run failed-only retries for transient HTTP 500/timeouts after each phase's
   pending jobs finish. Do not automatically retry policy failures, malformed
   scientific responses, or all 57 DeepSeek CodeAct failures as one group.
4. Reconcile and run the Docker-required Tier-4 RLM cells on a tested Docker
   machine. Prioritize a scientifically representative subset before attempting
   every held model/cell.

### P2: analysis required before manuscript claims are frozen

1. Consolidate ledgers without overwriting attempts and verify that every
   plotted cell has the expected number of questions and repetitions.
2. Report tokens, calls, end-to-end time, tool time, cost, peak memory, failure
   rate, and timeout-finalization rate—not only task score.
3. Compute uncertainty across questions/tasks as well as repetitions, and use
   paired comparisons where conditions share questions.
4. Build a failure taxonomy separating provider failures, resource limits,
   malformed responses, retrieval/execution mistakes, abstraction mistakes,
   and prospective alternatives.
5. Add negative cases or narrow the calibration claims if a defensible negative
   subset cannot be completed in time.

### P3: only after the result matrix is frozen

Regenerate plots and tables from scripts, complete the manuscript, run a
chemist/ML red-team read, prepare the reproducibility release, and post the
matching paper/code version to arXiv. Paper writing should proceed now, but
cosmetic plot work must not consume the time reserved for P0 controls.

## Ways to shorten the remaining runtime

These are ordered by expected benefit without weakening the experiment.

1. **Use two workers within one adequately sized node.** GLM has averaged about
   1,452 calls in 9.15 hours, or 2.6 calls per minute, well below its six-RPM
   Jed share. A resumed Jed allocation with about 8--10 CPUs, 60--64 GiB, and
   `--max-parallel 2` could overlap one RLM trajectory's model wait or tool
   execution with another. The manifest's 48-GiB parallel-memory budget will
   still serialize incompatible pairs. Test five to ten jobs before retaining
   this setting.
2. **Pilot a second local DeepSeek worker only on low-memory, non-Docker jobs.**
   `liacpc14` has enough memory for carefully selected 16-GiB plus 30-GiB
   reservations, and both workers would share the same six-RPM node limiter.
   Do not start this until selectors and combined memory have been mechanically
   checked. Earlier unconstrained CodeAct accelerators increased timeouts, so
   this must be a measured RLM pilot rather than an assumption.
3. **Keep one allocation per SwissAI host.** Multiple Slurm nodes do not share
   `/tmp`; each could independently consume the full host share. More nodes are
   unsafe unless quota is statically divided again or a proven distributed
   limiter is introduced.
4. **Retry only transient failures.** The six early GLM failures are candidates
   after pending work finishes. HTTP 403 policy failures and malformed
   scientific outputs are not. Failed-only selection prevents hundreds of
   completed jobs from being repeated.
5. **Use narrow causal subsets.** Oracle predicates, retrieval/map-reduce, and
   prospective decomposition should first cover representative hard/easy tasks
   with one open and one closed model. This can answer the paper's causal
   questions much faster than another complete six-model sweep.
6. **Separate coding from inference.** Existing cluster jobs can run unattended
   while independent clean branches implement the oracle, retrieval, and
   prospective controls and prepare human-review candidates. Do not modify
   active execution clones.
7. **Stop low-information work.** If a release pilot repeatedly demonstrates a
   stable provider or model limitation, report the limitation and redirect time
   to acceptance-critical controls instead of forcing hundreds of identical
   failures.

## Active blockers and decisions

1. **GPT upstream policy rejection:** the current OpenAI route refuses the RLM
   request before producing a choice. The key and spending balance are healthy.
   One Azure-only GPT-5-mini pilot may test the same model through another
   declared provider; otherwise GPT refusals must be reported rather than
   hidden by prompt changes or model substitution.
2. **GLM CodeAct release:** the demanding Task-16 x500 cell must succeed on Jed
   before broad GLM CodeAct is released. GLM non-Docker RLM is independently
   eligible to run.
3. **SwissAI shared quota:** `liacpc14` and the single Jed allocation each own
   six requests per minute. Separate Jed allocations or nodes must not each
   claim another six.
4. **Remote counts:** Qwen, Gemini, Claude RLM, and matched Qwen require fresh
   ledger summaries. Jed GLM is now tracked from Slurm job `66534424`.
5. **Failures are not erased:** DeepSeek failures, GPT policy failures, memory
   failures, and all interrupted attempts remain in their respective ledgers.

## Next reports required

### From Sathvik on `liacpc15`

Send one ledger-derived table containing, for Qwen and Gemini separately:

```text
method | succeeded | failed | running | pending
LLM
CodeAct
RLM
```

Also send the same table for matched Qwen, and report whether Docker is
available and whether Tasks 16, 17, and 17b were included in the RLM count.

### From Jed Codex

Keep Slurm job `66534424` running. Send periodic GLM RLM ledger counts, observed
jobs/hour, projected completion time, and the exact GLM CodeAct release-pilot
error. Before changing resources, prepare a resume-safe two-worker Slurm draft
and estimate speedup from the current trace; do not interrupt the active job
solely because more CPUs exist.

### From Kuma Codex

Send fresh Claude LLM/CodeAct/RLM ledger counts and the status of its Slurm
allocation. For GPT, prepare one provenance-clean Azure-only GPT-5-mini pilot
using the unchanged Task-14 request and the reviewed non-transient-error
handling. Report its exact provider-routing configuration, estimated cost, and
tests before submission. Do not submit the 19-job or 505-job groups merely to
produce fresher counts.

### From `liacpc14`

Refresh DeepSeek from
`artifacts/iclr2027-six-model-full-v34/ledger.sqlite3`. When CodeAct ends,
record its final success/failure totals and confirm that RLM actually started.

## Updating this file

Every update should include:

1. local time and machine;
2. Git commit;
3. experiment name and ledger path;
4. exact succeeded/failed/running/pending counts by method;
5. active scheduler or `tmux` job;
6. newly observed failure causes;
7. API cost and remaining balance when applicable;
8. whether Docker-required jobs are included or held.

Commit status-only updates with a message such as:

```text
Update benchmark execution status
```

Do not edit generated TOML files just to update this coordination record.

## Email completion notifications

`task-notify` is now a standalone utility at
<https://github.com/amansouri3476/tasknotify>, installed for Amin under
`~/.local/bin`. RxnHaystack contains only this project-specific coordination
record and examples. The reusable implementation is intentionally not coupled
to this repository, so it can also monitor CoQ, Catelier, and other projects.

In addition to completion, failure, launcher termination, and cost-limit mail,
the utility now sends hourly updates by default. It labels a message
`PROGRESS` if ledger counters, the active run, or a configured progress file
changed; `STATUS` after one unchanged interval; and `STALL` after two hours
without an observable change. A stall alert is explicitly not proof of a hang,
because a legitimate model request can remain inside one trajectory for a long
time. Per-watcher settings are `--status-interval-seconds` and
`--stall-after-seconds`; setting the first to zero disables periodic mail.
Version 0.3.0 sends an immediate `STARTED` message when a watcher attaches and
stores its email `Message-ID` in the private watcher state. Every later update
and terminal alert is a reply under that same stable-subject conversation.
Version 0.3.1 standardizes every new subject as
`[tasknotify] Project | Task | execution-machine`; outcome remains in the body
so progress and terminal mail do not acquire changing titles.

Six durable benchmark watchers are configured on `liacpc14`:

- `task-notify-claude-docker` watches all 45 Claude Docker RLM cells, launcher
  PID `3147146`, and the guarded OpenRouter queue log;
- `task-notify-gpt-docker` watches all 45 direct-OpenAI GPT Docker RLM cells,
  launcher PID `3156222`, and the CHF 30 experiment ceiling.
- `task-notify-deepseek-nondocker` watches the non-overlapping 405-cell
  DeepSeek RLM scope and launcher PID `2698386`;
- `task-notify-deepseek-docker` watches the 45-cell DeepSeek Docker scope and
  launcher PID `2791520`.
- `task-notify-deepseek-paid` watches the 279-cell paid OpenRouter continuation
  with its CHF 40 ceiling;
- `task-notify-gpt-nondocker-recovery` watches the exact 21-cell direct-OpenAI
  recovery with its CHF 20 ceiling.

All four use the canonical chat label `RxnHaystack ICLR 2027 master chat` and chat
machine `liacpc14`. Their mode-600 states are under
`~/.local/state/task-notify/`. SMTP delivery to `amansouri3476@gmail.com` has
been tested successfully. Credentials stay in private files outside Git. If
SMTP is temporarily unavailable, a terminal message remains pending and the
watcher retries it rather than discarding it.

Commit `7091e0e` adds native Slurm observation through `--slurm-job-id`. It
checks `squeue` while an allocation is live, falls back to `sacct` afterward,
and records the scheduler job ID and final state in the email. This is the
preferred attachment mechanism for Jed and Kuma.
