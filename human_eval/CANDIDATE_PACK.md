# Candidate-pack specification

Required top-level fields are `schema_version`, unique `pack_id`, semantic `version`,
integer `seed`, and `candidates`. Each candidate requires a unique source
`candidate_id`, canonical `question_id`, and structured `candidate_answer`. Include
`reaction_indices` where available. Optional administrator-only fields include
`model_id`, `run_id`, `prompting_method`, `evaluator_outcome`, `control_type`, and
`duplicate_group`.

Import recursively removes identity/evaluator/control metadata from public payloads,
derives opaque IDs, and deterministically shuffles. The separate mode-0600 map binds
opaque IDs to original IDs and hidden fields. Never distribute that file to
annotators. Use a new pack version whenever candidates or metadata change.

Controls are ordinary hidden candidates: label known-good routes `positive`, known-
bad routes `negative`; duplicate reliability items share `duplicate_group`. Assign
the same item to at least two anonymous IDs for double annotation. Candidate content
itself must not mention a model or evaluator decision.
