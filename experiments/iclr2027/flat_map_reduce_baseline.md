# Fixed map-and-union baseline

This experiment asks whether the useful part of RLM is specifically adaptive,
recursive orchestration, or whether a fixed exhaustive decomposition is enough.
It is deliberately separate from the six-model benchmark and must not be
launched until its one-job release check has passed.

## What it does

For one question, the runner:

1. reads the full cleaned dataset in its recorded order;
2. divides all 122,456 indexed rows into deterministic, non-overlapping chunks;
3. sends the unchanged chemistry question and one chunk to an independent
   one-shot mapper;
4. requires the mapper to return only indices present in that chunk; and
5. deterministically unions the valid partial sets.

There is no model-based reduce call and no recursive call. Every row is covered
exactly once. At the tested default of 500 rows, a run has

```text
ceil(122,456 / 500) = 245 mapper calls
```

The last chunk contains 456 rows. Chunk size remains configurable through
`RXNHAYSTACK_MAP_REDUCE_CHUNK_SIZE`; 500 is the structural-test default because
it matches an existing benchmark context size. A real one-chunk release check
must still establish that the selected provider accepts its prompt size before
the 245-call pilot is approved.

Responses outside a mapper's assigned chunk are not silently removed from the
evaluation. They are recorded as invalid claims and added to the false-positive
count. Duplicate valid indices are unioned once. A malformed response receives
at most three total attempts; deterministic client errors such as HTTP 403 are
not retried. Transient timeouts, HTTP 429, and appropriate 5xx failures are
bounded and auditable.

The checkpoint is written atomically after every successful chunk. Relaunching
the same experiment skips completed chunks. Its identity includes the corpus,
question, model, and chunk size, so a checkpoint cannot be reused after one of
those inputs changes. If any chunk remains missing, the job fails instead of
publishing a score over partial corpus coverage.

Each successful job writes normal launcher metrics plus:

- `map-reduce-checkpoint.json`, with every completed mapper result;
- `map-reduce-details.json`, with predictions, score, per-chunk latency,
  attempts, tokens, cost availability, and invalid claims; and
- the standard resource trace and W&B summary.

Unknown paid-provider cost remains `null`, never zero. Under the current main
branch's archival contract, the scientific details are preserved but the job
then remains failed until accounting is available. The baseline does not depend
on the separate, unmerged GPT usage-recovery work.

## Publication-sized design

The small design selects one question at each row-separable difficulty level:

- Tier 1 Task 1, question 1: exact product lookup;
- Tier 2 Task 5, weight >100 Da and ring delta 1: conjunctive descriptors; and
- Tier 3 Task 10, Wittig olefination: chemistry-dependent classification.

It uses Qwen3.5-397B-A17B and Gemini 3.7 Flash with five repetitions. Both are
already part of the main paper comparison. Gemini is the closed-model default
because this exhaustive baseline repeats unusually large inputs and Gemini's
recorded input price is lower than Claude Haiku's:

```text
3 questions × 2 models × 5 repetitions = 30 tracked jobs
30 jobs × 245 chunks = 7,350 mapper calls
```

For either model, the minimum work is 3,675 calls. At a six-request-per-minute
host cap, the SwissAI half requires at least 10.2 hours of request starts,
before inference time and retries. The generated Gemini allowance is CHF 180
within a CHF 200 experiment ceiling; it is deliberately conservative and must
be replaced by a measured one-job estimate before broad paid launch.

Generate and inspect the experiment with:

```bash
uv run --frozen python experiments/iclr2027/generate_flat_map_reduce_experiment.py --check
uv run --frozen rxnhaystack validate experiments/iclr2027/flat-map-reduce-experiment.toml
uv run --frozen rxnhaystack plan experiments/iclr2027/flat-map-reduce-experiment.toml \
  --select 'flat-map-reduce-qwen3.5-397b-tier1-task1-q01-xfull-c500-r01'
```

Do not run all 30 jobs initially. The release sequence is:

1. mocked unit and integration tests;
2. one provider request containing a representative 500-row chunk;
3. one complete 245-chunk Qwen job;
4. review accuracy, failure rate, tokens, elapsed time, and peak memory; and
5. only then approve the remaining open and paid jobs.

## Why Tier 4 is excluded

Tier 1--3 answers are unions of independent per-reaction decisions. Tier-4
chains may join reactions stored in different chunks. A chunk-local mapper
therefore cannot enumerate the correct result, and a task-specific graph reducer
would inject an engineered solution. Oracle-predicate and prospective-task
controls address Tier 4 more cleanly.

## Why there is no universal retrieval baseline

This benchmark asks for the complete satisfying set, not the most relevant few
records. A top-k retriever changes the question and imposes a hard recall ceiling.
Several property and mechanism questions have no query molecule to embed, while
Tier-4 answers are multi-reaction paths whose individual steps need not resemble
the natural-language query. A task-specific SMARTS or fingerprint filter would
also encode much of the chemistry predicate and become an oracle-predicate
condition rather than a neutral retriever.

The existing 100- and 500-row contexts are random, ground-truth-aware samples;
they must not be described as retrieval. The paper should explain that a single
benchmark-wide RAG baseline is ill-defined, avoid claiming that retrieval is
generally irrelevant, and leave task-specific candidate-recall studies as a
separate question.
