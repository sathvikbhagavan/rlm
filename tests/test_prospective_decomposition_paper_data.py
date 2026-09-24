from __future__ import annotations

from paper_plots.scripts.build_prospective_decomposition import aggregate_rows, build_rows


def _run(model: str, condition: str, repetition: int, f1: float) -> dict:
    return {
        "model": model,
        "condition": condition,
        "repetition": repetition,
        "run_id": f"{model}-{condition}-{repetition}",
        "status": "succeeded",
        "sources": ["test"],
        "result_updated_at": "2026-09-24T00:00:00+00:00",
        "attempts": [
            {
                "status": "succeeded",
                "started_at": "2026-09-24T00:00:00+00:00",
                "metrics": {
                    "results": {
                        "queries_evaluated": 3,
                        "macro_f1": f1,
                        "macro_precision": f1,
                        "macro_recall": f1,
                        "exact_match_accuracy": f1,
                    }
                },
            }
        ],
    }


def test_complete_prospective_grid_and_aggregation() -> None:
    models = (
        "RCP-AIaaS/Qwen/Qwen3.5-397B-A17B",
        "anthropic/claude-haiku-4.5",
    )
    conditions = ("name_only", "structure_only", "structure_plus_class")
    runs = [
        _run(model, condition, repetition, 0.25 if condition == "structure_plus_class" else 0.0)
        for model in models
        for condition in conditions
        for repetition in range(1, 6)
    ]

    rows = build_rows({"runs": runs})
    aggregates = aggregate_rows(rows)

    assert len(rows) == 30
    assert sum(row["question_count"] for row in rows) == 90
    assert len(aggregates) == 6
    class_rows = [row for row in aggregates if row["condition"] == "structure_plus_class"]
    assert {row["mean_macro_f1"] for row in class_rows} == {0.25}
