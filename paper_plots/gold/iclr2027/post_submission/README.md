# Post-submission corrected-rescore queue

This is internal audit material, not manuscript prose. It freezes every run whose score is excluded from submission-time aggregates. Historical values are retained only for reproducibility; every listed run requires a post-submission audit against the corrected human bundle.

- Unique queued runs: **1743**
- Recovery ledger: `rxnhaystack-corrected-score-recovery-2026-09-25-v2`
- Submission score freeze: `rxnhaystack-iclr2027-submission-score-freeze-v1`
- Configured sampling seed: **42** for every queued run; `repetition` records the independent model repetition (`r01` through `r05`).

After submission: recover or rerun exactly these run IDs against the corrected bundle, replace the carried score with an exact rescore, regenerate every gold table and figure, and compare the resulting manuscript claims against the frozen submission manifest.

## By experiment

| Value | Runs |
| --- | ---: |
| `causal_control` | 274 |
| `causal_control;main_benchmark` | 90 |
| `causal_control;matched_control` | 150 |
| `main_benchmark` | 1093 |
| `x1000_extension` | 136 |

## By model

| Value | Runs |
| --- | ---: |
| `claude-haiku-4.5` | 255 |
| `deepseek-v4-flash` | 243 |
| `deterministic-executor` | 9 |
| `gemini-3.7-flash` | 267 |
| `glm-5.2` | 140 |
| `gpt-5-mini` | 394 |
| `qwen3.5` | 435 |

## By task

| Value | Runs |
| --- | ---: |
| `tier3/task10` | 315 |
| `tier3/task18` | 292 |
| `tier3/task23` | 330 |
| `tier3/task6` | 309 |
| `tier3/task7` | 276 |
| `tier4/task15` | 221 |

## By method

| Value | Runs |
| --- | ---: |
| `codeact` | 372 |
| `llm` | 360 |
| `rlm` | 1011 |
