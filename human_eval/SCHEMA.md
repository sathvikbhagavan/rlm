# Schema and data dictionary

## Canonical bundle

`questions.jsonl` is annotator-safe: `schema_version`, stable `question_id`, `tier`,
`category`, `subcategory`, `canonical_prompt`, `answer_type`, `source_files`, corpus
checksum, protected `ground_truth_ref`, suggested minutes, scoring metadata, and
non-answer metadata. It contains no answer representation or reaction evidence.

`admin/ground_truth.jsonl` maps the protected reference to `representation`,
`relevant_reaction_indices`, evaluator metadata, and source files. Large answers
occur once here. `manifest.json` freezes file checksums, taxonomy, audit sampling,
known source inconsistencies, and excluded historical modules.

Answer types are `index_set`, `reaction_chains`, `reaction_pair_set`, `smiles_set`,
and `single_chain`. Exact submitted text is retained in `answer_exact`; a conservative
line/comma tokenization is also saved as `answer_entries`.

## SQLite and exports

`profiles` holds a random ID and optional non-PII expertise JSON. `drafts` holds the
latest autosave; `revisions` is append-only history; `timing_sessions` holds active,
wall, start/end, and state; `studies`/`assignments` freeze allocation; `candidates`
contains only blinded public payloads.

Each draft and revision includes `annotation_context_json`, which freezes the active
schema/bundle version and the public-question, protected-ground-truth, and dataset
checksums. This lets an updated suite share the same SQLite state without erasing or
misattributing earlier work. Legacy rows created before this field was introduced
remain valid with an empty context and should be reported as version-unresolved.

Annotation payloads contain mode-specific judgments, confidence, rationale,
abstention, issue tags, tool categories, verification, and offline minutes. Timestamps
are timezone-aware ISO 8601 UTC. ZIP checksums cover each member.
When editable metadata is prefilled from a completed question of the same
tier/category/subcategory, `prefill_source_question_id` records its provenance.

Restore accepts only manifest-declared members with safe relative names, verifies
each size and SHA-256 checksum, enforces compressed/uncompressed limits, and merges
rows idempotently. It cannot replace a workspace that already contains another
anonymous annotator's work.
