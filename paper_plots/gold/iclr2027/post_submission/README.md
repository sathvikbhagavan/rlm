# Post-submission corrected-rescore queue

This is internal audit material, not manuscript prose. It freezes every run whose exact corrected score is still pending. The submission tables carry the preserved historical score for these rows so terminal trajectory denominators remain complete.

- Unique queued runs: **201**
- Recovery ledger: `rxnhaystack-corrected-score-recovery-2026-09-25-v2`
- Submission score freeze: `rxnhaystack-iclr2027-submission-score-freeze-v1`
- Configured sampling seed: **42** for every queued run; `repetition` records the independent model repetition (`r01` through `r05`).

After submission: recover or rerun exactly these run IDs against the corrected bundle, replace the carried score with an exact rescore, regenerate every gold table and figure, and compare the resulting manuscript claims against the frozen submission manifest.

## By experiment

| Value | Runs |
| --- | ---: |
| `causal_control` | 32 |
| `causal_control;main_benchmark` | 10 |
| `main_benchmark` | 136 |
| `provisional_control` | 17 |
| `x1000_extension` | 6 |

## By model

| Value | Runs |
| --- | ---: |
| `claude-haiku-4.5` | 30 |
| `deepseek-v4-flash` | 15 |
| `gemini-3.7-flash` | 32 |
| `glm-5.2` | 31 |
| `gpt-5-mini` | 61 |
| `qwen3.5` | 32 |

## By task

| Value | Runs |
| --- | ---: |
| `tier3/task10` | 19 |
| `tier3/task18` | 119 |
| `tier3/task23` | 45 |
| `tier3/task7` | 18 |

## By method

| Value | Runs |
| --- | ---: |
| `llm` | 59 |
| `rlm` | 142 |
