# October 2026 follow-up assignments

These five manifests complete first-pass human coverage of all 100 canonical
questions when combined with the three frozen exports in
`paper_plots/gold/iclr2027/human_validation`. The final questions in each
manifest are validation cross-checks selected for known evaluator sensitivity
or an earlier non-exact answer.

| Annotator | Core | Validation cross-checks | Estimated core time | Estimated cross-check time |
| --- | ---: | ---: | ---: | ---: |
| Tom | 10 | 1 | 2.5--3 hours | 15--20 minutes |
| Theo | 5 | 2 | 65--75 minutes | 25--30 minutes |
| Gabriel | 12 | 2 | 95--105 minutes | 20--25 minutes |
| Sarina | 10 | 2 | 100--115 minutes | 20--25 minutes |
| Vu | 19 | 3 | 4--4.5 hours | 40--50 minutes |

The estimates use the bundle's suggested times and the medians from completed
exports. Active browser and self-reported offline time were recorded separately,
so these are planning ranges rather than additive measurements.

After pulling the current repository, install the personalized manifest before
starting the application:

```bash
uv run --frozen --with-requirements human_eval/requirements.txt \
  python -m human_eval.cli install-study \
  human_eval/studies/2026-10-followup/NAME.json
./human_eval/run.sh
```

Open the assigned-study URL using the `study_id` from the manifest, for example:

```text
http://127.0.0.1:8765/questions?mode=baseline&study_id=rxnhaystack-followup-tom-2026-10-v1
```

The ordered `question_ids` field places core coverage first and validation
cross-checks last. Both sections are part of the requested contribution. If an
annotator reaches their time limit, the preserved ordering ensures that new
benchmark coverage is collected before additional replication.

Each annotator must export the resulting ZIP and send that single file to Amin.
Do not inspect protected ground-truth or `human_eval/generated/*/admin` files.
