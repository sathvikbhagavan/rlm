# Canonical bundle changelog

## rxnhaystack-human-1.2.0 — September 14, 2026

Updated the empirically informed suggested stopping limits to 5 minutes for Tier 1,
10 minutes for Tiers 2–3, and 15 minutes for Tier 4. Clarified, without changing
the predicate or answer, that the Tier-2 ring-count maximum over all component
pairs is equivalently the maximum product ring count minus the minimum reactant
ring count. Stable question IDs, ground truths, and the dataset are unchanged.

## rxnhaystack-human-1.1.0 — September 12, 2026

Rebuilt from Sathvik Bhagavan's source commit `8987a3c` (“fix: last 10 questions
of tier4”). Stable question IDs and the dataset are unchanged.

Prompts changed for all 10 sequential-template questions:

- `rxh-t4-task17-boc-buchwald`
- `rxh-t4-task17-nitro-reduction-amide`
- `rxh-t4-task17-suzuki-carbonyl-reduction`
- `rxh-t4-task17-wittig-hydrogenation`
- `rxh-t4-task17-alcohol-oxidation-reductive-amination`
- `rxh-t4-task17b-azide-staudinger`
- `rxh-t4-task17b-aryl-brom-negishi`
- `rxh-t4-task17b-boronic-suzuki`
- `rxh-t4-task17b-boc-deprot-buchwald`
- `rxh-t4-task17b-aryl-brom-suzuki-ester-red`

Stored answer representations changed for five:

- `rxh-t4-task17-boc-buchwald`
- `rxh-t4-task17-nitro-reduction-amide`
- `rxh-t4-task17-wittig-hydrogenation`
- `rxh-t4-task17-alcohol-oxidation-reductive-amination`
- `rxh-t4-task17b-boronic-suzuki`

Bundle 1.0 remains archived locally for version-correct analysis of any earlier
submissions. The application marks restored or local drafts whose saved bundle
context differs from the active bundle.
