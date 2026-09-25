# Trace-level failure review

This review complements the aggregate score and set-error analysis. It asks
*why* a run failed by following the attempted computation through generated
code, tool observations, intermediate strategy changes, and the scored answer.
The reviewed examples are recorded in `reviewed_trace_causes.csv`.

## Scope and safeguards

- The inventory contains all 6,150 jobs in the five paper-model arms.
- The current corrected-score queue invalidates historical scores for 1,179 of
  those jobs. They are excluded from scientific failure analysis until exact
  rescoring completes.
- The remaining 4,971 jobs contain 16,601 question-level trajectories. Among
  them, 2,429 successful jobs have a non-perfect Tier-3 or Tier-4 score and are
  eligible for trace review.
- A deterministic sample draws four candidates for every model--interface
  pair, giving 60 jobs: 12 per model and 20 per interface. Forty-three traces
  are available locally. The 17 inaccessible traces belong to collaborator-run
  Qwen and Gemini projects for which the local W&B key lacks permission.
- Automated regular expressions retrieve possible evidence such as exceptions,
  time limits, literal-string proxies, unsupported tool wrappers, and empty
  parses. They do not assign scientific causes. Every row in the reviewed table
  was read manually and cites direct evidence from its trace.

The sample is stratified for model and interface coverage and deliberately
emphasizes low-scoring, task-diverse examples. Its cause counts are therefore
not prevalence estimates for the benchmark. Population-level claims continue
to use all valid scores; the review explains mechanisms.

## Recurrent mechanisms

1. **The intended computation sometimes never ran.** Claude Haiku 4.5 emitted
   Python inside unsupported `python` or `bash` action wrappers in two CodeAct
   examples, received no tool observation, and then described the unexecuted
   program as successful. DeepSeek V4 Flash separately parsed zero reactions in
   a graph task after misreading the indexed record format.
2. **Textual or narrow structural proxies replaced the required chemistry.**
   GPT-5 mini and Claude Haiku 4.5 searched SMILES fields for literal reagent
   names such as HATU and T3P. GPT-5 mini used tautomer-specific fused-ring
   SMARTS, while route runs used target-specific substrings or generic amine
   patterns. These shortcuts failed even though positives were seeded into the
   sampled contexts.
3. **Several chemistry operations require correspondence, not local pattern
   detection.** DeepSeek V4 Flash correctly noticed that bond-change detection
   requires atom correspondence, but its canonical-rank and MCS substitutes did
   not establish one and triggered invalid RDKit calls.
4. **Graph construction is sensitive to chemically irrelevant shared
   components.** A Gemini 3.7 Flash trace linked reactions through TFA because a
   simple heavy-atom threshold treated it as a persistent intermediate. The
   model diagnosed and patched that instance, illustrating why exact shared
   SMILES alone does not establish a meaningful synthetic dependency.
5. **Full-corpus access does not guarantee stable chemical rules or exhaustive
   enumeration.** Within one Gemini 3.7 Flash RLM trace, two chain families were
   exact, one recovered only 1 of 37 valid chains, and another produced 80
   candidates with zero overlap. Qwen 3.5 showed the same task dependence, with
   223 candidates for a 9-chain Suzuki family and no valid Boc--Buchwald chain.
6. **Infrastructure must remain separate from scientific error.** One Claude
   Haiku 4.5 RLM example includes a refused connection to the model endpoint;
   it is an execution failure, not evidence of weak chemical reasoning. Another
   archived Qwen 3.5 run lacks the referenced verbose trace, so its cause is
   explicitly left unclassified.

## Relation to the workshop examples

The earlier examples highlighted brittle SMARTS, incomplete enumeration, and
route-construction errors. The expanded review confirms those mechanisms across
additional models and adds failures that the old taxonomy obscured: unsupported
tool-call syntax, parsing the corpus as zero records, pseudo-execution without
an observation, common-reagent graph links, and loss of subcall decisions at
aggregation. The old examples remain useful illustrations, but this review is
broader and more careful about causal attribution.

## Remaining work before paper inclusion

The corrected task families must be rescored before their traces can re-enter
the scientific analysis. The inaccessible collaborator traces should also be
exported if we want the final reviewed sample to have identical local evidence
coverage for every model--interface pair. Until then, the paper should report
the unaffected reviewed mechanisms without presenting their counts as corpus
frequencies.
