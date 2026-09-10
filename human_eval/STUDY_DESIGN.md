# Study design

## Estimands

The baseline estimates independently checked human benchmark performance and time.
The audit estimates benchmark defect prevalence/severity. Prospective review
estimates the fraction of evaluator-rejected routes that chemistry-aware humans
consider plausible, retaining uncertainty as its own outcome.

These modes must not be pooled. Baseline must precede any audit exposure for a
given person/question. Record recruitment, assignment manifest, bundle checksum,
corpus checksum, software commit, dates, and annotator eligibility before launch.
Freeze and retain every bundle used for collection. If questions or ground truth
change, issue a new bundle/study version and analyze each version against its own
archived protected ground truth; never silently rescore old submissions against a
new answer merely because the stable question ID is unchanged.

## Assignment

Use deterministic category quotas rather than requiring all 100 questions. The
study JSON stores its ID, schema, mode, seed, annotators, and quotas. Each anonymous
ID receives a stable stratified shuffle. Plan overlap explicitly: at least two
annotators per prospective item and a prespecified audit subset are recommended.
Candidate packs may contain positive controls, negative controls, and hidden
duplicate groups; report control performance and duplicate consistency separately.

For large answers the frozen audit rule is SHA-256 ranking without replacement over
canonical JSON entries, seed `20270910`, first 25 entries. The bundle manifest saves
this method and seed. Changing either requires a new manifest version.

## Timing and missingness

Active time accumulates browser heartbeats and caps a heartbeat contribution at 60
seconds. Wall time spans start-to-pause sessions. Offline/tool time is self-reported.
Analyze these separately. Abstentions and time-limit outcomes stay in denominators
as reported outcomes but not as guessed wrong answers. “Uncertain” remains a
prospective category. Never impute unsubmitted assignments as incorrect.
The independent-baseline estimand always uses the first submitted payload. Later
post-answer-reveal revisions remain available for benchmark-diagnostic analysis but
must not replace the frozen first answer in accuracy calculations.

## Source-selection decision requiring author confirmation

The declared Tier-3 taxonomy totals 35. Historical task 9 and tasks 13/14/16/17 are
therefore excluded. Functional-group ground-truth modules contain 15 entries, but
four label/description entries are commented out in current runners. Extraction
keeps all 15 and mechanically uses their stored key and SMIRKS for the missing
display text. This is recorded in every affected item and the bundle manifest; the
authors should confirm those four prompts before the definitive study freeze.
