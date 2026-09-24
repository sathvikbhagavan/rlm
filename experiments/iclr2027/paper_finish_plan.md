# RxnHaystack paper finish plan

## Original list from Amin (verbatim)

- We need to have a proper figure 1. It could be a cute llama with lab coat looking in a haystack for a reaction. Maybe we just need the schematic though, or a mix of two, not sure. But figure 1 is something we need.
- <span style="color: #1a7f37;">We need to follow the assessment, and make sure if the claim of mechanical vs. reasoning is causally separated and has evidence? For that, which plot do we need?</span> **✅ Completed**
- The story should also emphasize the structured data aspect of this task in scientific discovery, something that might be more relevant when we step away from domains like math. Perhaps we could come up with more examples, like time-series (cite our adaptive time-series work), data in biology, I don't know, but make it make sense more and show the impact and need for it through some other examples.
- The assessment has found the failures insightful, we should redo the failure analysis now that we have so many models and arms, either automatically, or having codex finding the root causes, writing code for it, and running it on the traces. Failure analysis is important.
- 🟢 <span style="color: #1a7f37;">We should properly and briefly explain RAG is superseded by our baselines constructions. Essentially at any context length we put the ground-truths in, so it's already a ceiling for what a RAG (e.g. based on DRFP) could achieve. Because in 100, 500 out of 120k, we're putting the answer reaction along with the others. So we don't try RAG since we think it's subsumed.</span> **✅ Completed with oracle-recall wording**
- <span style="color: #1a7f37;">The assessment talks about the highest value move being oracle-predicate. Do we have results for that? What do those results tell us? We should add those results and the prose to overleaf to the proper location.</span> **✅ Completed**
- We should think about the presentation/framing as to what makes it a good benchmark, and show those qualities. I'm not sure what these qualities are, but I can think of:
  - A benchmark should be able to separate models, otherwise, it's not a good benchmark
  - It should show the existence of a gap (where we get from our human annotations)
  - It'd be better if it's connected to real-world, and is actually useful for humans. So it'd be great if we could explain how some of the difficult questions of our benchmark are things that experimental chemists would benefit from if solved reliably.
  - Any other suggestions are welcome!
  - Lastly, this paper is not at all about the strength of RLMs or how they good they are, we don't care, we just adopted them as a recent attempt at very long-context task
- <span style="color: #1a7f37;">The assessment writes "... the strongest evidence ..." right before section 3, does it still hold with our results?</span> **✅ Completed with qualified wording**
- Then moving to weaknesses:
  - <span style="color: #1a7f37;">Has our experiment answered W1?</span> **✅ Key cardinality confound answered for GPT-5-mini Tier 1--3; scope stated**
  - <span style="color: #1a7f37;">W2 is an explanation I mentioned above</span> **✅ Oracle-recall/RAG explanation added**
  - <span style="color: #1a7f37;">W3 is addressed</span> **✅ Four terminal RLM model arms summarized; two provisional arms excluded**
  - <span style="color: #1a7f37;">W4, do we need more clarification?</span> **✅ Replaced broad wording with operation-specific demands**
  - <span style="color: #1a7f37;">W5, is it too important? Do any of our claims rest on it? I don't think so.</span> **✅ Central claims do not rely on it; legacy-condition confound stated**
  - <span style="color: #1a7f37;">W6, we could point to it as a limitation, and mention it in the paper (always appreciated)</span> **✅ Added as a limitation**
  - W7 W8 W9: to be ignored for now, and delegated to later
  - <span style="color: #1a7f37;">W10: We should have plots of tool-call, tokens, latency, tool-time, all of it should enrich the appendix</span> **✅ Six-metric appendix figure added**
- Then moving to experiments:
  - <span style="color: #1a7f37;">A: Do we have it? Is it in the paper? If not, we should.</span> **✅ Completed: chemistry-rule control and deterministic ceiling are in the main results and appendix**
  - <span style="color: #1a7f37;">B: We have it, does it answer the question? Is it embedded in the paper?</span> **✅ Completed: the GPT-5-mini two-factor control is central to the causal-controls result**
  - <span style="color: #1a7f37;">C: Explain why not (DRFP, etc.)</span> **✅ Completed with oracle-recall wording**
  - <span style="color: #1a7f37;">D: Do we have it?</span> **✅ Completed: 30/30 jobs, 90 trajectories, and the controlled result is in Methods and Results**
  - E: I have it differently actually. We have human annotations for different chunks of the 100 questions, not their annotation for false positives.. **⏳ Not complete: one 20-question export is available locally; collaborator exports and expertise metadata still need consolidation**
  - <span style="color: #1a7f37;">F: Do our results support that?</span> **✅ Completed: the capability ordering is reported across Qwen, Gemini, GPT-5 mini, and Claude**
- 🟢 <span style="color: #1a7f37;">What should we release as artifact of this benchmark? Just the set of questions? The answers? Should we hold out anything for not being contaminated? The codes to obtain the ground-truth? The interface to obtain more human annotations?</span> **✅ Release policy completed: publish the current benchmark in full and build a separate private extension for future contamination-resistant evaluation**

### Current focus

**Just completed:** the assessment's strongest-evidence claim and W1--W6/W10
have been audited against the frozen results. The paper now uses the qualified
multi-model claim, explains the oracle-recall/RAG scope, separates specific
operation types, records the legacy route-task confounds as limitations, and
includes calls, tokens, latency, tool time, wall time, and memory in the
appendix. W7--W9 remain deliberately deferred as requested.

Updated September 23, 2026. This is the working document for turning the
workshop paper into the ICLR submission. It should be updated whenever an
analysis is accepted, a figure is frozen, or prose is pushed to Overleaf.

The active manuscript is `overleaf/iclr2027.tex` and its included files. The
September 7 critique is `RxnHaystack_ICLR_2027_Review_and_20_Day_Plan.tex`.
Plotting inputs and their final/provisional rules are documented in
`paper_plots/gold/iclr2027/README.md`.

## Paper identity

This is a benchmark and scientific-diagnosis paper, not an RLM advocacy paper.
RLM is one recent mechanism for interacting with contexts that do not fit in a
single prompt. The paper should ask:

> When an agent works over a very large structured scientific dataset, what
> fails first: access to the data, execution of the required computation, or
> construction and preservation of the right domain abstraction?

The intended answer is nuanced:

- systematic decomposition provides access to computations that ordinary
  prompt-bound interfaces cannot perform over the full corpus;
- fixed-cardinality evidence shows that corpus scale itself can degrade
  domain-heavy performance even when the number of required answers is held
  constant;
- supplying the benchmark's validated chemistry predicate helps at full scale,
  but does not make RLM perfect;
- deterministic execution of the same predicates is perfect, showing that
  faithful orchestration remains a separate bottleneck;
- route construction and multi-constraint chemistry remain substantially
  harder than mechanically specified graph operations across model families.

Every claim must distinguish deterministic structural execution, learned
chemical classification, graph orchestration, and prospective route reasoning.
Avoid using "chemical reasoning" as if these were a single capability.

## Rules for every result and figure

1. Terminal failed jobs score zero. Running, stale, and pending jobs are not
   silently treated as zero; they make an arm provisional.
2. Report coverage next to performance whenever an arm is incomplete.
3. Average over questions, not merely over task scripts whose question counts
   differ.
4. State what the horizontal axis means for each interface. Prompt context size
   and RLM chunk size are not identical information budgets.
5. Separate scientific failures from provider, accounting, timeout, memory, and
   transport failures.
6. Generate every number and plot from checked scripts and frozen CSV files.
7. Put the source snapshot time and checksums in the plotting record.
8. Do not describe a mixed control as confirming a cleaner story than the data
   support.

## Figure plan

### Figure 1: structured scientific data as a haystack

Figure 1 should be attractive, but its first job is explanation. The preferred
design combines a restrained visual metaphor with a serious schematic:

- a small lab-coated llama searching reaction cards in a haystack;
- the 122,456-row reaction corpus represented as structured reaction records,
  not natural-language documents;
- three access patterns: a prompt receives a sampled subset, CodeAct executes
  over that subset, and RLM can interact with the full corpus through recursive
  calls and tools;
- representative operations progressing from lookup and aggregation to
  chemistry predicates and reaction-graph paths;
- the two diagnostic axes: mechanical/orchestration demand and domain-
  abstraction demand.

The llama should be a visual entry point, not most of the panel. A reviewer
should understand the benchmark and its scientific question without reading
the caption. Prepare both a clean schematic-only version and the mixed version,
then choose after inspecting them at one-column and two-column size.

### Main benchmark figure

Show all six models and the three interfaces, with finality and coverage made
visible. The principal view should communicate:

- model separation;
- degradation from 100 to 500 rows for prompt-bound interfaces;
- RLM full-corpus results without implying that its `x` axis is identical to
  the prompt context-size axis;
- the contrast between mechanically specified and domain-heavy task groups.

Use the gold plotting records. Do not copy numbers from the dashboard by hand.

### Causal-controls figure

This should be a central two-panel figure, not an appendix afterthought.

**Panel A: matched cardinality.** At fixed positive cardinality `k=1`, plot
question-weighted F1 against corpus size. At fixed corpus size `N=5,000`, plot
F1 against `k in {1,5,20}` using exactly the tasks eligible for all three
cardinalities. Show Tier 2 and Tier 3 separately.

Current GPT-5-mini audit:

| Control | Tier 2 F1 | Tier 3 F1 |
| --- | ---: | ---: |
| `N=100, k=1` | 0.970 | 0.671 |
| `N=500, k=1` | 1.000 | 0.558 |
| `N=5,000, k=1` | 0.990 | 0.337 |
| `N=50,000, k=1` | 0.961 | 0.206 |
| `N=full, k=1` | 0.980 | 0.212 |
| `N=5,000, k=5` | 0.990 | 0.416 |
| `N=5,000, k=20` | 0.981 | 0.525 |

This answers the primary W1 confound for GPT-5 mini: Tier-3 degradation
persists while answer cardinality is fixed, whereas increasing cardinality at
fixed scale does not cause the decline. The Qwen arm must not be presented as
complete until its provider and scientific failures are audited.

**Panel B: oracle predicates.** Plot ordinary RLM, oracle-predicate RLM, and the
deterministic executor across corpus size, with task-level detail available in
the appendix. Use question weighting across the 16 questions.

Current full-corpus audit:

| Model | Ordinary RLM | Oracle-predicate RLM | Deterministic executor |
| --- | ---: | ---: | ---: |
| Claude Haiku 4.5 | 0.307 | 0.572 | 1.000 |
| Qwen 3.5 | 0.551 | 0.756 | 1.000 |

The correct conclusion is not "the oracle solves everything." Supplying the
label function produces a substantial full-corpus gain, especially on
predicate-heavy Tier-3 tasks, but the remaining gap to deterministic execution
shows that applying the rule exhaustively and preserving it through agent
orchestration are separate capabilities.

### Prospective decomposition figure

For each model, compare target-name only, target structure, and target structure
plus final-transformation class. The exact target-producing reactions are
excluded from the searchable corpus. Include score, timeout rate, calls, and
wall time. This is a targeted validity analysis; it is not the foundation of
the paper's main claim.

### Failure-analysis figure

Use a stacked distribution or matrix of verified failure causes by model,
interface, tier/task group, and scale. Pair it with a few compact trace excerpts.
Infrastructure failures must be shown separately from scientific failures.

### Appendix efficiency figures

Include, at minimum:

- model calls per trajectory;
- input, output, and total tokens per trajectory;
- end-to-end wall time per successful trajectory;
- cumulative model latency;
- tool execution time;
- peak process-tree plus Docker memory;
- paid API cost, clearly excluding free-access models from paid-model means;
- coverage and the fraction of jobs with unknown accounting.

Efficiency plots must never make a failed method look efficient merely because
it stopped early or produced no answer.

## Does the original strongest evidence still hold?

Yes qualitatively, but the workshop wording must be weakened and generalized.
Across the four currently terminal six-model arms (Qwen, Gemini, GPT-5 mini,
and Claude), preliminary question-weighted full-corpus RLM means are:

| Task group | Across-model mean F1 | Model range |
| --- | ---: | ---: |
| Bond changes | 0.865 | 0.611--1.000 |
| Stereochemistry | 0.888 | 0.660--1.000 |
| Mechanisms | 0.360 | 0.207--0.563 |
| Mechanical graph operations | 0.828 | 0.395--1.000 |
| Chemically constrained graph operations | 0.577 | 0.293--0.840 |
| Route and multi-constraint tasks | 0.061 | 0.020--0.131 |

Thus the capability split reproduces across model families. It is no longer
accurate to say that RLM is universally perfect on mechanical graph tasks:
Claude is an important counterexample. The stronger defensible claim is that
mechanically specified operations are consistently much easier than mechanism,
route, and multi-constraint reasoning, with meaningful model heterogeneity.

## Structured data and scientific-discovery framing

The introduction should explain why this is not another natural-language
needle-in-a-haystack task. Scientific agents must act on structured objects and
relations while preserving domain semantics. Examples include:

- reaction tables and synthesis graphs in chemistry;
- time-series trajectories from adaptive experiments, where an agent must
  compare dynamical responses and update hypotheses;
- longitudinal omics, perturbation screens, and cell-state transition graphs;
- materials databases linking composition, processing conditions, structure,
  and measured properties;
- astronomical catalogues and sensor streams requiring joins, temporal
  aggregation, event detection, and physical constraints;
- clinical event sequences, where temporal order, codes, measurements, and
  missingness are part of the problem rather than surrounding prose.

The direct related example is *Time-Series as Feedback: Evaluating Adaptive
Reasoning in LLM Agents* by Kuroki, Mansouri, and Schwaller, which evaluates
agents that choose experiments and interpret time-series feedback in a kinetic
mechanism-identification loop:
<https://openreview.net/pdf?id=N3vDM8CafN>.

The purpose of these examples is not to claim that chemistry results transfer
automatically. They motivate a broader benchmark category: reasoning over large
structured scientific environments, where access, execution, and domain
abstraction can fail independently.

## Why the benchmark is useful

The paper should explicitly evaluate RxnHaystack against benchmark-quality
criteria:

1. **Discrimination:** it separates models, interfaces, scales, and task groups
   rather than producing universal saturation.
2. **Construct validity:** matched-cardinality and oracle-predicate controls test
   whether the intended scale and abstraction factors actually explain the
   observed gaps.
3. **Diagnosticity:** task groups and traces reveal *how* a system fails, not
   only whether its final answer is wrong.
4. **Reliability:** deterministic ground truth, fixed data checksums, pinned
   chemistry software, fixed seeds, repeated runs, and preserved ledgers make
   results auditable.
5. **Ecological relevance:** questions operate over a real patent-reaction
   corpus and include operations useful in reaction-database curation, reaction
   classification, protecting-group analysis, route search, and synthesis
   planning.
6. **Human relevance:** difficult tasks correspond to work chemists would value
   if it were reliable, while human baselines expose the time and expertise such
   work requires.
7. **Calibrated breadth:** the benchmark spans deterministic scans, chemistry
   predicates, graph joins, and route construction without pretending they are
   one homogeneous skill.
8. **Extensibility:** question generators, scorers, and the annotation interface
   allow additional targets, predicates, negatives, and independent human
   audits.
9. **Resource transparency:** calls, tokens, tool time, latency, memory, cost,
   and failures can all be compared.

Current human evidence is one local export containing 20 completed baseline
questions from one annotator, not a false-positive plausibility study. Its
first-submission macro-F1 is 0.148, with one abstention and substantial tool and
offline time. This is promising evidence of a human--model/task gap, but it must
not be generalized until all collaborator exports, expertise metadata, and
question assignments are consolidated. Human evaluation E from the September 7
assessment has therefore been changed: the paper will report the actual
question-chunk baseline study, not claim that prospective false positives were
human-validated.

## Retrieval/RAG positioning

The benchmark deliberately removes the ordinary retrieval bottleneck at
`x=100` and `x=500`: known positives are deterministically seeded into the
candidate context and evaluation is against the answers present in that
context. This is best described as an **oracle-recall** or
**retrieval-elided** condition. It asks whether the model can recognize and
exhaustively return relevant structured records after retrieval has succeeded.

This supports the following argument:

- a DRFP or other similarity retriever cannot improve recall beyond having the
  relevant answers already present under the same candidate budget;
- observed failures therefore cannot be attributed solely to the retriever
  missing every answer;
- evaluating retrieval quality itself is a distinct systems question from the
  benchmark's present focus.

Do not claim that this universally "supersedes RAG." A learned retriever would
change the negative distribution, may retrieve useful supporting records, and
introduces precision/recall trade-offs. State these limitations explicitly.
The defensible paper claim is that RxnHaystack evaluates an oracle-retrieval
upper-bound condition and therefore prioritizes post-retrieval structured
reasoning over comparing retriever architectures. Because map-and-union was
shelved, do not claim that recursion itself is necessary or superior to every
systematic decomposition.

## Failure-analysis work package

The workshop failure analysis was insightful but too anecdotal and limited to
one model. Redo it across the completed matrix.

### Automated extraction

Build a trace-analysis script that extracts, for every attempt:

- generated code and tool calls;
- chemistry primitives used (SMARTS, SMIRKS, RDKit, string/reagent proxies);
- enumeration coverage and early stopping;
- exceptions, parse failures, timeouts, memory stops, and provider failures;
- answer size, precision, recall, and false-positive/false-negative counts;
- changes of strategy within a trajectory;
- calls, tokens, latency, tool time, wall time, and peak memory.

### Failure labels

Use a documented, testable taxonomy:

1. correct structural predicate, incomplete enumeration;
2. heuristic proxy substituted for a structural predicate;
3. wrong chemical transformation or mechanism abstraction;
4. incorrect graph link, join, or persistence constraint;
5. target interpretation or route-construction failure;
6. invalid or misused RDKit/code operation;
7. answer parsing or formatting failure;
8. premature stopping, iteration exhaustion, or timeout;
9. memory/resource failure;
10. provider, transport, or missing-accounting failure.

Rules can identify obvious infrastructure and code signatures automatically.
Scientific root causes need a stratified trace review. Coding assistants may
prepare evidence and propose labels, but authors should verify a sample and
report the rubric rather than treating an opaque LLM judge as ground truth.
Sample successes as controls, not only failures.

### Deliverables

- machine-readable per-attempt taxonomy;
- coverage and agreement audit for the labels;
- one aggregate failure figure;
- a small table of representative verified traces;
- prose connecting failure mechanisms to the benchmark's diagnostic value.

## Assessment weaknesses: current disposition

| Assessment item | Current disposition | Paper action |
| --- | --- | --- |
| W1 scale/cardinality confound | Key answer-count confound answered for GPT-5 mini Tier 1--3 | Central matched-cardinality panel and precise scope added |
| W2 missing RAG | Retrieval recall is intentionally elided, but RAG is not universally subsumed | Oracle-recall design and realistic-retriever caveat added |
| W3 one model | Addressed by six-model matrix; two full-corpus arms remain provisional | Four terminal RLM arms summarized with model ranges |
| W4 broad "chemical reasoning" | Addressed in current prose | Operation-specific abstraction and orchestration terminology used |
| W5 name-to-structure confound | Does not support the central paper claim; controlled study running | Legacy-condition confound stated; decomposition remains a validity analysis |
| W6 non-exhaustive prospective ground truth | Unresolved and real | Limitation now distinguishes exact-route recovery from plausible-route validity |
| W7 sparse families | Deferred | State scope; do not imply broad chemistry generalization |
| W8 no negatives | Deferred | State scope and future extension |
| W9 incomparable `x` axes | Deferred experimentally, cheap to clarify | Fix captions and methods language |
| W10 cost-only efficiency | Addressed | Six-panel calls/tokens/latency/tool/wall/memory figure added to appendix |

## Assessment experiments: current disposition

| Experiment | Status | What it currently says |
| --- | --- | --- |
| A. Oracle predicate | 150/150 model jobs and 15/15 deterministic jobs complete | Predicate supply helps at full scale, but orchestration remains imperfect |
| B. Matched cardinality | GPT 725/725 complete; Qwen excluded from this paper analysis | GPT causally separates corpus scale from answer cardinality |
| C. Retrieval and map-reduce | Not run; map-and-union deliberately shelved | Explain oracle-recall setting and narrow claims about recursion |
| D. Prospective decomposition | 30/30 jobs complete; 90 trajectories | Final-step information helps, while structure alone does not; route recovery remains difficult |
| E. Human validation | One 20-question baseline export is local; consolidation pending | Report only after exports, expertise, and assignments are verified |
| F. Multi-model replication | Complete for the four-model full-corpus comparison | Capability split reproduces, with important model heterogeneity |

## Artifact release plan

The detailed public-facing policy and packaging checklist are maintained in
[`benchmark_release.md`](benchmark_release.md).

The current 100 questions are already part of the research process and should
not be presented as a contamination-proof hidden test. Reproducibility is more
valuable than pretending they remain secret.

Release:

- all 100 public benchmark questions, prompts, schemas, and task metadata;
- context builders and frozen sampling seeds;
- data-reconstruction instructions, row counts, checksums, and pinned RDKit
  version;
- ground-truth generation code and versioned answer files;
- scorers and exact metric definitions;
- executable predicates used in the oracle study, with parity tests;
- experiment definitions, model/provider settings, timeouts, memory limits, and
  retry policy;
- sanitized per-run metrics, finality/coverage records, and plotting scripts;
- selected sanitized traces supporting the failure analysis;
- the human-annotation application, instructions, schemas, analysis code, and
  anonymized annotations for which release permission exists;
- a model card/data sheet describing intended use, limitations, licenses, and
  known benchmark errors.

For future contamination resistance, create a separate extension rather than
withholding the evidence used in this paper:

- keep question-generator templates public;
- reserve new targets, thresholds, reaction classes, and random seeds after the
  paper is frozen;
- evaluate the private extension through an evaluation service or periodically
  refreshed test release;
- report public-development and private-extension scores separately.

Do not withhold the current answers or generation code merely to create a
leaderboard. That would weaken scientific auditability without undoing existing
exposure.

## Writing and analysis queue

Each numbered item ends only after four deliverables exist: checked analysis,
publication-quality plot/table, manuscript prose, and a compiled Overleaf push.

1. **Freeze the claim/evidence table.** Record every headline claim, exact
   supporting experiment, finality, aggregation rule, and permitted wording.
2. **Design Figure 1.** Produce schematic-only and llama-plus-schematic drafts;
   inspect at paper size and choose one.
3. **Rewrite the six-model benchmark section.** Replace the GPT-only results
   text and old figures with the gold multi-model analysis.
4. **Completed: causal-controls figure and section.** Matched cardinality and
   executable chemistry-rule controls directly answer W1 and separate
   chemistry-rule inference from execution.
5. **Completed: prospective decomposition.** The three information conditions
   are analyzed over 90 trajectories and reported in Methods and Results.
6. **Consolidate human annotations.** Import all collaborator exports, verify
   expertise and assignments, run the existing analysis, and report the actual
   study design.
7. **Implement and run the multi-model failure analysis.** Freeze the taxonomy,
   audit labels, create the figure, and select trace examples.
8. **Completed: efficiency appendix.** Calls, tokens, latency, tool time, wall
   time, memory, cost, accounting coverage, and early-failure caveats are
   reported.
9. **Rewrite framing and related work.** Structured scientific data, benchmark
   qualities, RAG/oracle-recall positioning, time-series and other scientific
   examples, and an explicit statement that this is not an RLM paper.
10. **Rewrite methods and limitations.** Operation-specific terminology,
    provider/model settings, resource limits, scoring of failures, W5/W6, and
    deferred W7--W9 extensions.
11. **Prepare the release.** Public benchmark package, reproducibility commands,
    anonymous supplement, licenses, and optional future private extension.
12. **Final audit.** Recompute every number, check figure/table consistency,
    complete the ICLR checklist, compile, obtain chemistry and ML red-team
    reviews, and push the final Overleaf version.

## Immediate next step

Start with the claim/evidence table and causal-controls plot specification,
because they determine the title, abstract, Figure 1 labels, introduction, and
results order. In parallel, Figure 1 can be sketched and failure traces can be
indexed without waiting for the remaining Docker jobs.
