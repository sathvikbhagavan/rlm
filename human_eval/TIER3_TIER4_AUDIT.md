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
| T3 Tasks 6, 7, 8 | 15 | Tasks 6 and 8 exact; Task 7 corrected as described below |
| T3 Tasks 10 and 10b | 10 | Exact regeneration for every subquestion |
| T3 Tasks 11, 12, 15 | 3 | Tasks 11 and 15 exact in both checked environments; Task 12 exact under its benchmark RDKit 2025.09.6 contract |
| T3 Tasks 18, 19, 20 | 3 | Tasks 19 and 20 exact; Task 18 corrected as described below |
| T3 Tasks 21 and 22 | 2 | Exact regeneration |
| T3 Tasks 23 and 24 | 2 | Task 24 exact; Task 23 corrected as described below |
| T4 Tasks 11, 12, 12b | 5 | All complete-dataset answers regenerate exactly |
| T4 Tasks 13, 14, 15 | 10 | Task 13 definition clarified; Task 14 exact; Task 15 alternatives and scoring corrected below |
| T4 Task 16 | 10 | Every stored full chain and derived prefix regenerates exactly |
| T4 Tasks 17 and 17b | 10 | Every stored sequential-template chain regenerates exactly |

## Corrections

### Tier 3 Task 7

The old reaction-template matcher applied a single-reactant transformation only
once. It therefore omitted records in which the same requested transformation
occurred at multiple sites and the stored product contained the fully
transformed molecule. Reaction 3907, where both primary alcohols are oxidized
to a dicarboxylic acid, is the reported example. The matcher now follows
successive single-reactant template applications up to the number of matching
sites and accepts the recorded final product. Four subquestions gain records:

- nitrile to amine: 274 to 275 (adds 43529);
- nitro groups to amines: 2,053 to 2,064 (adds 11 records);
- alcohol to azide: 120 to 121 (adds 9683);
- alcohol to carboxylic acid: 63 to 64 (adds 3907).

Both Grignard subquestions are unchanged.

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

| Feedback source | Task | Required action for historical model results | Reason |
| --- | --- | --- | --- |
| Theo | T3 Task 18 | Rescore every LLM, RLM, and CodeAct output at every context size. | Ground-truth membership changed from 46,528 to 17,022. |
| Theo | T3 Task 23 | Rescore every LLM, RLM, and CodeAct output at every context size. | Ground-truth membership changed from 1,456 to 1,410. |
| Theo | T4 Task 12b | No model rescore; re-evaluate human submissions made against bundles before 1.3.0. | Model runners compute the answer from each supplied context; the defect was the human extractor treating eight sampling hubs as exhaustive. |
| Gabriel | T3 Task 7 | Rescore all model outputs for nitrile-to-amine, nitro-to-amine, alcohol-to-azide, and alcohol-to-carboxylic-acid. The two Grignard questions need no score change. | Corrected multi-site matching adds 14 records across four answer sets. |
| Gabriel | T4 Task 13 | No score-only correction. Preserve old results as the original-prompt condition; rerun the primary-alcohol-to-carboxylic-acid question if results under the clarified definition are needed. | Ground truth is unchanged; only the neutral-acid contract is now explicit. |
| Gabriel | T4 Task 15 | Rescore Task 15 raw outputs, at minimum the quinoline and indole questions. Report the corrected `macro_reaction_f1` and index-match fields; valid-path/objective-length correctness is unchanged. | The accepted alternative sets expand from 199 to 299 and from 184 to 241. |

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
