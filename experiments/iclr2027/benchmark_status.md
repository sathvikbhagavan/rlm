# ICLR 2027 benchmark execution status

This is the shared coordination record for the final benchmark. It says who
owns each part, where it is running, what is complete, and what must happen
next. Update this file whenever a phase starts, stops, or materially changes.

Last consolidated: **2026-09-17 01:05 Europe/Zurich**

Repository commit at consolidation: `70c885ed1c900622dead52ef4ef0ecd629882fbc`

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
| DeepSeek V4 Flash | Amin / `liacpc14` | 300 succeeded | Finished: 243 succeeded, 57 failed | 9 succeeded, 1 running, 440 pending | Verified locally at consolidation time |
| GLM 5.2 | shared / Jed | 300 succeeded in reusable v28 results on `liacpc14` | Release pilot failed; remaining 299 held | 22 succeeded, 6 failed, 1 running, 376 pending among 405 non-Docker jobs; 45 Docker jobs held | GLM LLM verified locally; RLM from Jed report |
| Qwen 3.5 | Sathvik / `liacpc15` | Reported complete, nominally 300 | Reported complete, nominally 300 | Reported near completion; exact success/failure/running/pending split missing | Reported by Sathvik through Amin |
| Gemini Flash | Sathvik / `liacpc15` | Reported complete; exact ledger count missing | Reported complete; exact ledger count missing | Reported launched; progress and outcome counts unknown | Unverified collaborator report |
| Claude Haiku | Amin / Kuma | 300 succeeded | Effectively 300 succeeded: 299 assigned cells plus one compatible prior cell | Last confirmed: 163/405 succeeded, 1 running, 241 pending; 45 Docker jobs held | Kuma ledger report, but RLM count is stale and needs refresh |
| GPT-5-mini | Amin / Kuma | 300 succeeded | 300 succeeded | 384/405 succeeded; 19 missing-usage/empty-response failures and 2 memory failures; 45 Docker jobs held | Verified by Kuma ledger audit |

### DeepSeek details on `liacpc14`

- CodeAct finished with 243 successful and 57 failed jobs.
- Active phase: RLM, one worker, six SwissAI request starts per minute at most.
- Active run at consolidation:
  `full-deepseek-v4-flash-tier1-task1-rlm-x500-r05`.
- The CodeAct failures are retained for a later failed-only review/retry. Do
  not retry them while they would compete with the active RLM phase.
- RLM started automatically at `2026-09-16T21:13:47Z`. Nine jobs succeeded,
  one was active, 440 were pending, and none had failed at consolidation.

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
- GPT repair branches are held and unmerged:
  `fix/gpt-missing-usage-recovery` at `655f835` and the combined
  `review/gpt-retry-definitions` at `62f113a`.
- Do not submit the 19-job or 505-job retries until the source of the 403 is
  identified. Querying the three preserved generation IDs is read-only and is
  the next diagnostic step.

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
4. Diagnose the GPT HTTP 403 using preserved generation metadata. Do not spend
   on the 19 full or 505 matched retries until routing/policy is understood.

### P0-B: acceptance-critical controls not yet implemented

The external review ranks these above polishing a six-model leaderboard. Begin
their implementation in parallel with the unattended full runs:

1. **Oracle-predicate control:** a small, representative Tier-3/Tier-4 subset
   comparing normal RLM with validated executable chemistry predicates and a
   deterministic executor ceiling.
2. **Retrieval and map-reduce baselines:** retrieval followed by LLM/CodeAct,
   plus a non-recursive chunk-and-merge baseline on the same representative
   tasks.
3. **Prospective-task decomposition:** target name only versus target structure
   versus target structure plus final transformation class for Tasks 16, 17,
   and 17b.
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

## Active blockers and decisions

1. **GPT policy rejection:** determine whether HTTP 403 came from an OpenRouter
   workspace guardrail, upstream OpenAI policy, an allowlist, or a spending/key
   restriction. Deterministic 403 responses should not receive transport
   retries.
2. **GLM CodeAct release:** the demanding Task-16 x500 cell must succeed on Jed
   before broad GLM CodeAct is released. GLM non-Docker RLM is independently
   eligible to run.
3. **SwissAI shared quota:** `liacpc14` and the single Jed allocation each own
   six requests per minute. Separate Jed allocations or nodes must not each
   claim another six.
4. **Remote counts:** Qwen, Gemini, Claude RLM, Jed GLM, and matched Qwen require
   fresh ledger summaries.
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

Report whether commit `781c0f6` was pulled, the combined Slurm job ID, GLM RLM
counts, the GLM CodeAct release-cell result, and whether broad CodeAct opened.

### From Kuma Codex

Send fresh Claude LLM/CodeAct/RLM ledger counts and the status of its Slurm
allocation. For GPT, report the read-only generation-metadata diagnosis for
the three Task-14 policy-refusal IDs. Do not submit held retries merely to
produce a fresher count.

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
