from __future__ import annotations

from experiments.iclr2027.import_sathvik_succeeded_pack import nested_metrics


def test_task15_reaction_f1_is_available_under_common_score_name() -> None:
    metrics = nested_metrics(
        {
            "metrics.calls": 3,
            "metrics.results.macro_reaction_f1": 0.75,
            "metrics.resources.peak_combined_memory_mib": 123.0,
        }
    )
    assert metrics["calls"] == 3
    assert metrics["results"]["macro_reaction_f1"] == 0.75
    assert metrics["results"]["macro_f1"] == 0.75
    assert metrics["resources"]["peak_combined_memory_mib"] == 123.0


def test_existing_macro_f1_is_not_overwritten() -> None:
    metrics = nested_metrics(
        {
            "metrics.results.macro_f1": 0.5,
            "metrics.results.macro_reaction_f1": 0.75,
        }
    )
    assert metrics["results"]["macro_f1"] == 0.5
