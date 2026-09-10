# RxnHaystack human validation

This isolated, local-first FastAPI application supports three distinct studies:
an independent human benchmark baseline, a benchmark/ground-truth audit, and a
blinded prospective-answer plausibility review. SQLite autosave, timing, revision
history, dataset inspection, deterministic assignment, and one-file export work
without OpenRouter, W&B, or any API credential.

## Start in one command

From the repository root:

```bash
./human_eval/run.sh
```

Open <http://127.0.0.1:8765>. The first launch verifies the clean corpus, builds
the canonical bundle, and indexes 122,456 reactions. Later launches resume the
same anonymous profile and SQLite state. The server binds only to loopback. A
non-loopback `--host` is rejected unless `--allow-network` is explicit.

Drafts autosave every 10 seconds and on normal tab/background transitions. The
question screen's **Save and pause** button performs a confirmed save and closes the
timing session. Keep `human_eval/local_state/` to resume after restarting the app.

Dependencies are supplied from `human_eval/requirements.txt` with `uv --frozen`;
the main lockfile is not changed. Override the corpus with
`RXNHAYSTACK_CLEANED_DATASET=/absolute/path`.

## Reproducible commands

```bash
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli build-bundle
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli validate-bundle
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli index-dataset
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli serve
```

Generated bundles and local state are ignored. `questions.jsonl` contains no
answer values or evidence indices. Protected values are under the bundle's
`admin/` directory and are never sent by baseline routes before submission.

## Assigned studies and candidate packs

Free-browse mode is always available. Install a deterministic stratified study:

```bash
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli install-study human_eval/examples/study_manifest.json
# Then open /questions?mode=baseline&study_id=pilot-baseline-v1
```

The manifest may list multiple `annotators`, each with category `quotas`; stable
annotator-specific randomization naturally supports overlap and double annotation.

Import prospective candidates as an administrator:

```bash
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli import-candidates human_eval/examples/candidate_pack.json
```

Original IDs and source/evaluator metadata are written separately with mode 0600
under `local_state/admin/`. Browser records use opaque derived IDs.

## Export and tests

Use **Export** in the browser or:

```bash
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli export
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m pytest -q human_eval/tests
```

The ZIP is also a portable checkpoint. On a fresh workspace use **Restore** in the
browser, or:

```bash
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli restore rxnhaystack_human_annotation_rxh-….zip
```

Restore verifies all manifest checksums and safely merges repeated imports. It
refuses to mix an export with an existing different annotator's work.

## Updating questions or ground truth

Annotation state is independent of the generated bundle. Building or serving a new
bundle does not delete drafts, submissions, timing, or revision history as long as
the same `--state` directory is retained. Every new autosave/revision is stamped
with the active public-question, ground-truth, and dataset checksums, so work from
different suite versions remains distinguishable.

Before changing a live study, archive the old canonical bundle—including its
protected `admin/` component—and build the update to a new directory such as
`human_eval/generated/canonical-v2`. Do not overwrite the old bundle needed to
score its submissions. Restart the application with both the new bundle and the
unchanged state directory:

```bash
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli serve \
  --bundle human_eval/generated/canonical-v2 \
  --state human_eval/local_state
```

Stable question IDs resume the existing answer; changed or removed IDs leave the
old response safely in SQLite/export history. A semantic change should receive a
new bundle version and study manifest rather than being silently treated as the
same study.

See [ANNOTATOR_INSTRUCTIONS.md](ANNOTATOR_INSTRUCTIONS.md),
[STUDY_DESIGN.md](STUDY_DESIGN.md), [SCHEMA.md](SCHEMA.md),
[CANDIDATE_PACK.md](CANDIDATE_PACK.md), [ANALYSIS.md](ANALYSIS.md), and the
ready-to-send [INVITATION.md](INVITATION.md). The separate
[SPECIALIST_REVIEW_INSTRUCTIONS.md](SPECIALIST_REVIEW_INSTRUCTIONS.md) should only
be sent to people explicitly assigned audit or candidate-route work.
