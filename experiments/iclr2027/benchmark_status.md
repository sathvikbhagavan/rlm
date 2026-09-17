# ICLR 2027 benchmark execution status

This is the shared coordination record for the final benchmark. It says who
owns each part, where it is running, what is complete, and what must happen
next. Update this file whenever a phase starts, stops, or materially changes.

Last consolidated: **2026-09-17 16:34 Europe/Zurich**

Repository commit at consolidation: `b8f6120c0a3764579b0ca46cb6ec90943d39e949`

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
| Amin | `liacpc14` | DeepSeek full benchmark; later Docker-required catch-up where assigned | GLM work assigned to Jed; Qwen/Gemini owned by Sathvik |
| Sathvik | `liacpc15` | Qwen and Gemini full benchmarks; Qwen matched-cardinality | DeepSeek, GLM, Claude, GPT |
| Amin and Sathvik | Jed CPU cluster | GLM CodeAct and GLM non-Docker RLM in one quota-coordinated allocation | DeepSeek remains on `liacpc14`; do not launch a second independent Jed SwissAI allocation |
| Amin | Kuma | Claude and GPT-5-mini full benchmarks; GPT matched-cardinality; GPT repair pilots | SwissAI models and Sathvik's models |

No owner should launch another machine for the same run names without first
changing this table and checking both ledgers for overlap.

## Full-benchmark status

| Model | Owner / machine | LLM | CodeAct | RLM | Confidence |
| --- | --- | ---: | ---: | ---: | --- |
| DeepSeek V4 Flash | Amin / `liacpc14` | 300 succeeded | Finished: 243 succeeded, 57 failed | 34 succeeded, 1 failed after clean interruption, 1 running, 414 pending | Verified locally after guarded restart |
| GLM 5.2 | shared / Jed | 300 succeeded in reusable v28 results on `liacpc14` | Release pilot failed; remaining 299 held | 22 succeeded, 6 failed, 1 running, 376 pending among 405 non-Docker jobs; 45 Docker jobs held | GLM LLM verified locally; RLM from Jed report |
| Qwen 3.5 | Sathvik / `liacpc15` | Reported complete, nominally 300 | Reported complete, nominally 300 | Reported near completion; exact success/failure/running/pending split missing | Reported by Sathvik through Amin |
| Gemini Flash | Sathvik / `liacpc15` | Reported complete; exact ledger count missing | Reported complete; exact ledger count missing | Reported launched; progress and outcome counts unknown | Unverified collaborator report |
| Claude Haiku | Amin / Kuma | 300 succeeded | Effectively 300 succeeded: 299 assigned cells plus one compatible prior cell | Last confirmed: 163/405 succeeded, 1 running, 241 pending; 45 Docker jobs held | Kuma ledger report, but RLM count is stale and needs refresh |
| GPT-5-mini | Amin / Kuma | 300 succeeded | 300 succeeded | 384/405 succeeded; 19 missing-usage/empty-response failures and 2 memory failures; 45 Docker jobs held | Verified by Kuma ledger audit |

### DeepSeek details on `liacpc14`

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
| `liacpc14` | Available and tested | Candidate machine for assigned Docker catch-up |
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
3. **Prospective validity is unresolved.** Task 16 target-name, target-structure,
   and target-structure-plus-class conditions have not been run, and the available
   chemist-review application has not yet produced annotations.
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
| `feature/task16-prospective-decomposition` at `6cc7719` | Correct Task-16 name/structure/class conditions, remove exact-target leakage, fix taxonomy and prepare human-review export | Merged to main at `6a85c01`; class arm held for chemist approval |
| `feature/flat-map-reduce-baseline` at `475d018` | Resumable Tier-1--3 flat mapper/union reducer, artifacts and mocked experiment definition | Shelved on its remote branch by author decision; do not merge or launch before post-submission review |

The prospective definition contains 30 jobs and 90 trajectories, with an
estimated API cost of CHF 2.73 and a CHF 15 ceiling. The map-and-union
definition contains 30 jobs but expands to 7,350 mapper calls because each job
must cover 245 chunks. Real-corpus calibration estimates about 12.5 million
input tokens per job; its Qwen/Gemini paid ceiling is CHF 180 within a CHF 200
budget. This work is now explicitly post-submission: do not launch even the
pilot unless the author reopens it after the ICLR deadline.

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

The project lead confirms that the reused original ground-truth predicates were
previously reviewed by a chemist. Preserve the reviewer/date/version record for
the paper. This prior validation is distinct from the still-pending review of
the three new Task-16 final-transformation class labels.

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
