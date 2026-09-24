# RxnHaystack benchmark release policy

This document defines the public artifact described in the ICLR 2027 paper. It
separates the fully reproducible benchmark used in the paper from a future
private evaluation extension.

## Decision

Release the current RxnHaystack benchmark in full. The public artifact should
include the 100 questions, answer files, ground-truth generators, context
builders, scorers, and evaluation interface. Withholding the answers or the
generation code would not make this benchmark contamination-resistant: the
questions, experimental design, and aggregate results are already disclosed by
the paper. It would, however, make the reported results harder to audit.

The current benchmark is therefore a transparent, versioned research benchmark
and public development set. A separate extension should provide private or
rotating evaluation instances for future contamination-sensitive comparisons.

## Public benchmark artifact

### Questions and evaluation

Release:

- all 100 question instances, stable identifiers, prompts, answer schemas, and
  task metadata;
- the exact answer sets used in the paper;
- ground-truth generation code and parity tests;
- context construction code, frozen sampling seeds, and in-context answer
  construction;
- task-specific parsers, scorers, aggregation rules, and failure-scoring policy;
- the executable chemistry rules and deterministic executor used by the
  chemistry-rule control;
- the prospective-route control prompts, approved transformation descriptions,
  target-removal rule, and scoring protocol.

Answers and generation code belong together. The frozen answer files make the
paper immediately reproducible, while the generators expose the scientific
definitions and allow independent verification or extension.

### Dataset

The source dataset is the public *Reaction SMILES USPTO year 2023* release,
DOI `10.6084/m9.figshare.24921555.v1`, licensed CC BY 4.0. Keep the 23 MB source
and cleaned reaction files outside the Git repository. Release:

- the upstream DOI, citation, and license;
- the download and atomic reconstruction script;
- the complete cleaning procedure and pinned RDKit version;
- raw and cleaned row counts and SHA-256 checksums;
- a versioned cleaned-data archive outside Git if convenient, carrying the
  upstream attribution and CC BY 4.0 notice.

This provides byte-identical reconstruction without making the software
repository carry a second copy of the dataset.

### Reproducible execution and evidence

Release:

- the locked Python environment and installation instructions;
- experiment definitions, model identifiers, prompts, generation parameters,
  timeouts, memory limits, retry policy, and budget ceilings;
- the portable launcher, ledger schema, and result-file contract;
- sanitized per-run scientific metrics, coverage/finality records, frozen
  aggregate CSV files, plotting scripts, and source manifests;
- selected sanitized traces supporting the reported failure analysis;
- tests for dataset reconstruction, question generation, scoring, controls,
  aggregation, and plots.

Do not release API keys, credential hashes, authorization headers, private W&B
metadata, unrestricted provider responses, local paths that identify users, or
raw traces containing secrets. Publish a machine-readable inventory with a
SHA-256 checksum for every released file.

### Human evaluation

Release the local annotation application, study definitions, instructions,
assignment logic, schemas, import/export tools, and analysis code. Release
annotations only after:

1. annotators have agreed to public use;
2. identities and unnecessary free-form metadata have been removed;
3. expertise categories and question assignments can be reported without
   identifying individuals; and
4. the anonymized export passes a manual privacy review.

The public artifact should clearly distinguish the independent benchmark
baseline, ground-truth audit, and prospective-route plausibility review. They
answer different scientific questions and must not be pooled into one metric.

## Contamination-resistant extension

Do not hold back a subset of the current 100 questions and describe it as
private. Instead, create new instances after the public benchmark is frozen:

- reserve new target molecules, thresholds, reaction classes, task parameters,
  and sampling seeds;
- keep the private instance parameters and answer sets outside the public
  repository;
- expose the same public schemas and scoring contract;
- evaluate through a hosted service or periodically refreshed test release;
- limit submissions and record model/checkpoint dates;
- report public-development and private-extension scores separately; and
- release retired private sets later for auditability, replacing them with new
  instances.

This design preserves a transparent scientific artifact while providing a
credible test of future systems that could have encountered the public
questions during training.

## Packaging before submission

The anonymous supplement should contain the software and benchmark files above,
excluding identifying Git history and private annotation exports. Before the
archival public release:

- assign a semantic benchmark version and immutable release tag;
- archive the release under a DOI;
- add a citation file, benchmark/data sheet, third-party notices, and known-
  issues list;
- verify the repository license and dataset attribution;
- run the documented reconstruction and one end-to-end scoring smoke test from
  a fresh checkout; and
- compare every released aggregate against the final paper tables and figures.

The release is complete only when an independent user can reconstruct the
dataset, regenerate the 100 questions and answers, score a prediction file, and
reproduce the reported aggregate tables without access to private services.
