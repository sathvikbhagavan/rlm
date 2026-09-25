# Tier 3 and Tier 4 canonical-answer audit

Audit date: 2026-09-25

Dataset SHA-256: `9f9b2e71676e3e8f132b495b3fc62e2fdec01bc5b43f7be3dc09ef279c351b14`

This audit covers all 70 canonical Tier 3 and Tier 4 questions included in the
human bundle. Superseded or unused task modules that are not among the 100
canonical questions are outside its scope.

## Method

- Rebuilt answers from the clean 122,456-record dataset and compared semantic
  sets, rather than comparing serialized files.
- Used the benchmark environment (RDKit 2025.09.6) for the original Tier 3
  bond-perception contract and the isolated human environment (RDKit
  2026.03.6) to detect version drift.
- Re-ran the complete Tier 4 graph miners under RDKit 2026.03.6 and compared
  every stored chain, pair, prefix, and molecule.
- Reviewed each canonical prompt against its extractor and human-evaluation
  answer type. Exact regeneration alone is not considered proof that a prompt
  and predicate agree.

## Results

| Canonical family | Questions | Result |
| --- | ---: | --- |
| T3 Tasks 6, 7, 8 | 15 | Task 8 exact; Tasks 6 and 7 corrected as described below |
| T3 Tasks 10 and 10b | 10 | Task 10 Wittig corrected below; the other nine subquestions regenerate exactly |
| T3 Tasks 11, 12, 15 | 3 | Tasks 11 and 15 exact in both checked environments; Task 12 exact under its benchmark RDKit 2025.09.6 contract |
| T3 Tasks 18, 19, 20 | 3 | Tasks 19 and 20 exact; Task 18 corrected as described below |
| T3 Tasks 21 and 22 | 2 | Exact regeneration |
| T3 Tasks 23 and 24 | 2 | Task 24 exact; Task 23 corrected as described below |
| T4 Tasks 11, 12, 12b | 5 | All complete-dataset answers regenerate exactly |
| T4 Tasks 13, 14, 15 | 10 | Task 13 definition clarified; Task 14 exact; Task 15 alternatives and scoring corrected below |
| T4 Task 16 | 10 | Every stored full chain and derived prefix regenerates exactly |
| T4 Tasks 17 and 17b | 10 | Every stored sequential-template chain regenerates exactly |

## Corrections

### Tier 3 Task 6

The old acyl-chloride SMIRKS began with a wildcard `[*]-C(=O)Cl`. That admitted
chloroformates and carbamoyl chlorides, whose products are carbamates and ureas,
despite the prompt specifying amide formation from an acyl chloride. The corrected
predicate requires a carbon substituent, `[#6]-C(=O)Cl`. The primary-amine answer
changes from 1,347 to 942 reactions (the disabled secondary-amine definition changes
from 833 to 633). Reviewer 1 returned 935 of the 942 corrected primary-amine records
with no false positives.

### Tier 3 Task 7

Two independent defects were corrected. First, the old reaction-template matcher
applied a single-reactant transformation only once. It therefore omitted records in
which the same requested transformation occurred at multiple sites and the stored
product contained the fully transformed molecule. Reaction 3907, where both primary
alcohols are oxidized to a dicarboxylic acid, is the reported example. Second, the
templates define connectivity but the matcher compared isomeric SMILES, rejecting
valid products whose newly formed stereocenter was recorded explicitly. Reviewer 1
identified seven such Grignard examples; exhaustive regeneration found ten.

The final bundle-1.6 counts are:

- Grignard ketone to tertiary alcohol: 33 to 43;
- Grignard aldehyde to secondary alcohol: 73 to 78;
- nitrile to amine: 274 to 275;
- nitro groups to amines: 2,053 to 2,067;
- alcohol to azide: 120 to 133;
- alcohol to carboxylic acid: 63 to 64.

### Tier 3 Task 10 (Wittig)

The first-stage Wittig templates used a wildcard for the atom bound to phosphorus.
Consequently `P=S` thionating reagents matched the supposed ylide/phosphorane and the
cascade mislabeled carbonyl-to-thiocarbonyl conversions as Wittig olefinations. The
atom is now restricted to carbon in both charge-separated and double-bonded forms.
The answer changes from 99 to 45 reactions. Reviewer 1 returned 42 of the corrected
45 with no false positives.

### Tier 3 Task 18

The old ring-fragment matcher retained substituent-dependent atom properties.
It therefore classified ordinary substitution and coupling on an existing ring
as construction of a new ring system, contrary to the prompt. Reaction 3 is a
minimal dataset example. The predicate now compares induced ring-atom graphs
using element/aromaticity atom labels and internal bond labels, ignoring
external substituent properties. The answer changes from 46,528 to 17,022
reactions.

### Tier 3 Task 23

The prompt requested uppercase absolute R/S stereocenters, while the old
predicate also admitted lowercase pseudoasymmetric r/s assignments. The prompt
and evaluator now explicitly retain uppercase R/S only. The answer changes
from 1,456 to 1,410 reactions.

### Tier 4 Task 12b

This was corrected in bundle 1.3.0. The former human extractor exposed eight
support hubs selected for sampled model contexts as though they were the
exhaustive answer. The complete-dataset answer contains 2,091 molecules and
includes `BrCc1cccc(Br)c1`.

### Tier 4 Task 15

The prompt asks for any one valid chain, but the human evaluator previously
required the annotator to submit the complete set of 44 to 199 alternatives.
The new `one_of_reaction_chains` answer type accepts exactly one stored valid
chain. A separate 200-solution traversal cap also made two stored alternative
sets incomplete. Exhaustive mining now stores 299 quinoline chains (formerly
199) and 241 indole chains (formerly 184). Benzothiazole (44) and
benzimidazole (142) are unchanged.

### Tier 4 Task 13

The existing SMARTS `[CX3](=O)[OX2H1]` consistently defines
`carboxylic_acid` as neutral, protonated R-C(=O)-OH. Gabriel's recomputation
reproduced all 550 stored chains, so the answer is unchanged. The prompt now
states the SMARTS contract and explicitly excludes carboxylate anions and their
salts; this removes an avoidable representational ambiguity.

## Historical model rescoring ledger

Keep raw model outputs and context artifacts fixed when applying these entries.
Do not silently mix corrected scores with scores produced under an older oracle
or prompt version.

### Correction provenance and affected-run identity

| Change | Correction commit | Historical-run selection |
| --- | --- | --- |
| T4 Task 12b exhaustive human answer | `793b08a` | Human submissions using bundles before 1.3.0; model scores are unaffected. |
| T3 Tasks 18 and 23; T4 Task 15 one-chain human evaluator | `5eefa91` | Every archived run whose canonical task is `tier3/task18` or `tier3/task23`; the human evaluator change does not alter model outputs. |
| T3 Task 7; T4 Task 13 prompt; T4 Task 15 exhaustive alternatives | `ffbbe41` | Every archived `tier3/task7` and `tier4/task15` run; Task 13 remains the original-prompt condition. |
| T3 Tasks 6 and 10; final connectivity-only Task 7 correction | `bfb038a` | Every archived run whose canonical task is `tier3/task6`, `tier3/task7`, or `tier3/task10`. |
| Task 15 score-field normalization | `778c3fe` | Every `tier4/task15` run selects `macro_reaction_f1`; all other tasks select `macro_f1`. |

Exact affected run IDs and their old values are retained row by row in
`paper_plots/gold/iclr2027/full_benchmark_records.csv`,
`codeact_x1000_records.csv`, `rlm_x1000_records.csv`, and the causal-control
records. Affected rows carry correction ID
`rxnhaystack-ground-truth-2026-09-25-v2`, preserve the former score in
`original_f1`, and use `score_correction_status` to distinguish invalidated
historical scores from affected jobs without a historical score. The source
manifest checksums these tables and the correction overlay.

| Feedback source | Task | Required action for historical model results | Reason |
| --- | --- | --- | --- |
| Theo | T3 Task 18 | Rescore every LLM, RLM, and CodeAct output at every context size. | Ground-truth membership changed from 46,528 to 17,022. |
| Theo | T3 Task 23 | Rescore every LLM, RLM, and CodeAct output at every context size. | Ground-truth membership changed from 1,456 to 1,410. |
| Theo | T4 Task 12b | No model rescore; re-evaluate human submissions made against bundles before 1.3.0. | Model runners compute the answer from each supplied context; the defect was the human extractor treating eight sampling hubs as exhaustive. |
| Gabriel | T3 Task 7 | Rescore all model outputs for nitrile-to-amine, nitro-to-amine, alcohol-to-azide, and alcohol-to-carboxylic-acid. The two Grignard questions need no score change. | Corrected multi-site matching adds 14 records across four answer sets. |
| Gabriel | T4 Task 13 | No score-only correction. Preserve old results as the original-prompt condition; rerun the primary-alcohol-to-carboxylic-acid question if results under the clarified definition are needed. | Ground truth is unchanged; only the neutral-acid contract is now explicit. |
| Gabriel | T4 Task 15 | Rescore Task 15 raw outputs, at minimum the quinoline and indole questions. Report the corrected `macro_reaction_f1` and index-match fields; valid-path/objective-length correctness is unchanged. | The accepted alternative sets expand from 199 to 299 and from 184 to 241. |

The Task-7 entry above is superseded by bundle 1.6.0: connectivity-only comparison
also changes both Grignard answers and adds further nitro/azide records. Two additional
entries apply:

| Feedback source | Task | Required action for historical model results | Reason |
| --- | --- | --- | --- |
| Reviewer 1 | T3 Task 6 | Rescore the acyl-chloride/primary-amine subquestion in every historical arm. | Corrected membership changes from 1,347 to 942. |
| Reviewer 1 | T3 Task 10 | Rescore the Wittig subquestion in every historical arm. | Corrected membership changes from 99 to 45. |

### Applied historical-score policy

The sanitized experiment records retain counts and aggregate metrics but do not
consistently retain the predicted index sets required for exact rescoring. The recovery
pipeline therefore reads the preserved W&B console logs, reconstructs each historical
sampled context using the original runner-specific settings, extracts raw predictions
where present, and validates every reconstruction against the originally reported
per-question and macro scores. Tier-4 Task 15 is checked against the historical chain
sets before its expanded alternatives are scored. The nine affected deterministic
executor cells are rerun against the corrected predicates rather than assumed correct.

`paper_plots/gold/corrected_score_recoveries.json` now freezes 931 unique exact
rescores. These restore 588 main-benchmark rows, 19 CodeAct-x1000 rows, 43 RLM-x1000
rows, 216 completed causal-control rows, and 96 provisional Qwen control rows. Some
run IDs occur in both the main and control tables, so table applications intentionally
outnumber unique recoveries. Restored rows are labeled `corrected_exact_rescore`, keep
their historical value in `original_f1`, and carry the recovery ID in `sources`.

Of 1,743 unique affected runs, 812 remain unresolved: 480 logs are in a W&B entity to
which the current account receives HTTP 403, and 332 accessible logs do not preserve
enough prediction detail to determine the corrected overlap from aggregate
precision/recall/counts alone. No corrected score is guessed or imputed. Consequently,
595 main-benchmark, 60 CodeAct-x1000, 14 RLM-x1000, 148 completed control, and 54
provisional Qwen control rows remain `historical_score_invalidated`; the respective
tables also contain 77, 41, 3, 0, and 25 affected jobs without a historical score.
Status, cost, token, timing, and memory evidence remains valid. The cached source logs,
extracted prediction ledger, historical Task-15 chain pack, and all input/output
checksums make the completed recovery reproducible without exposing credentials.

## Reviewer-source audit

Theo's supplied archive is preserved as a sanitized, checksummed source tree at
`human_eval/reviewer_code/rxh-e39c32021e72`; the embedded dataset, pickle caches,
bytecode, and cache directories are excluded. Twelve of his thirteen submitted
answers exactly match bundle 1.6.0. The Tier-4 Task-13 mismatch is not a benchmark
defect: his code includes alkyl fluorides although the contract uses Cl/Br/I and joins
successive reactions whenever any product is consumed later, allowing molecule
identity to switch between steps instead of following one coherent molecule chain.
His saved Tier-2 molecular-weight script also contains a debug `exit()` before its
calculation, so that archived file alone cannot reproduce the submitted answer. His
Task-23 source calls `FindMolChiralCenters(..., includeUnassigned=False)` without the
uppercase-CIP filter used for the corrected R/S contract; the submitted answer was
manually corrected and does match the canonical set, but the saved script by itself
does not document that last filtering step. Several unattempted assigned-question
files are empty placeholders. None of these source-archive limitations changes the
scored first submissions. The complete adjudication of all reviewer claims is in
`REVIEWER_FEEDBACK_AUDIT.csv`.

## Version-sensitive record

Tier 3 Task 12 differs by one record across the two inspected RDKit releases.
For reaction 55015, RDKit 2025.09.6 perceives an unusual aromatic C-O bond in
the reactant, while RDKit 2026.03.6 perceives it as single. Because the prompt
distinguishes aromatic and single bonds, the stored benchmark answer retains
55015 and now names RDKit 2025.09.6 as its bond-perception contract. This is a
specified representation choice, not a silent chemical correction.

## Remaining study-design limitation

Task 16's stored prefixes are internally correct, but the benchmark constructs
question contexts that withhold terminal target-forming reactions. The
standalone human application also provides a browser and download for the full
dataset, so those terminal reactions are physically available to annotators.
Task 16 human-baseline results should either use a controlled evidence pack
that actually excludes the terminal reactions or be reported as a less-blind
full-dataset diagnostic. This does not affect Tasks 17/17b, whose human answers
are defined over the complete dataset.

## Non-canonical artifact retained for provenance

`tier4/task13_fg_hardcoded_chains.json` still contains an older
`primary_alcohol->tertiary_amide` mined payload. It is not listed in Task 13's
`FIXED_QUESTIONS`, is not extracted into the 100-question bundle, and cannot
affect human scoring. It has been left untouched rather than deleting historical
evidence during this audit.
