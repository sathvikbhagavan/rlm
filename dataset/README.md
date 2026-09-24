# RxnHaystack dataset

RxnHaystack uses the 2023 USPTO reaction-SMILES release on Figshare
([DOI 10.6084/m9.figshare.24921555.v1](https://doi.org/10.6084/m9.figshare.24921555.v1)).
The upstream record is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and should be cited as:
Rik van der Lingen (2023), *Reaction SMILES USPTO year 2023*, Figshare,
DOI 10.6084/m9.figshare.24921555.v1.
Dataset files are stored outside Git by default under `~/datasets/rxnhaystack`.

Reconstruct and verify the benchmark corpus without changing the project lockfile:

```bash
uv run --frozen \
  --with-requirements dataset/requirements.txt \
  python -m dataset.prepare
```

Use `--verify-only` to check existing files. Override the location with
`--data-dir`, `RXNHAYSTACK_DATA_DIR`, or the file-specific
`RXNHAYSTACK_RAW_DATASET` and `RXNHAYSTACK_CLEANED_DATASET` variables.

Expected provenance:

| Artifact | Lines | SHA-256 |
| --- | ---: | --- |
| Raw | 137,261 | `8ba53c8aa5a513bdef651fb06f3ed1392ac3056c5f472087bdc52e820440b791` |
| Cleaned | 122,456 | `9f9b2e71676e3e8f132b495b3fc62e2fdec01bc5b43f7be3dc09ef279c351b14` |

The cleaning procedure sanitizes each molecule with RDKit, removes atom maps,
emits canonical isomeric SMILES, sorts molecules within each reaction field,
and removes duplicate canonical `(reactants, reagents, products)` triples.
