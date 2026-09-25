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

## Mapping failures to the benchmark's diagnostic objective

RxnHaystack is designed to locate the first broken link in a capability chain:

1. **Dataset access:** can the system reach the relevant records?
2. **Faithful structured execution:** can it parse the records and execute the
   required scan, join, or transformation without losing observations?
3. **Chemical abstraction:** can it formulate the structural or mechanistic
   rule that defines a valid answer?
4. **Relational orchestration:** can it preserve that rule while enumerating
   and combining paths or recursive subproblems?
5. **Target and route reasoning:** can it connect a target description to a
   compatible multi-step synthesis path?

The traces become informative only when read against the benchmark controls.
In the sampled contexts, every scored positive is already present, so literal
reagent searches and brittle SMARTS are post-access failures of chemical
abstraction. Full-corpus RLM traces control access more strongly: family-level
under- and over-enumeration after all 122,456 reactions are exposed implicates
rule induction and orchestration. The chemistry-rule-supplied control removes
rule induction, while the deterministic executor removes model orchestration;
the remaining gaps separate those two capabilities. Mechanical graph tasks
provide an internal comparison for chemically constrained graph tasks, and the
Task-16 decomposition isolates target representation and final-step knowledge
within route construction.

This mapping changes the paper's interpretation of the examples. A failed
CodeAct run is not evidence that tools are unhelpful: unsupported action syntax
or a zero-record parse shows that executable *access* was available but
structured execution failed. A full-corpus RLM run that exactly solves two
chain families yet overgenerates another does not exhibit a generic context
failure; it shows transformation-dependent abstraction and enumeration. A
connection refusal supports neither conclusion and remains an infrastructure
failure. The complete mechanism-to-objective mapping is recorded in
`diagnostic_objective_map.csv`.

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
