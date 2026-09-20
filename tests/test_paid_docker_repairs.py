from __future__ import annotations

from experiments.iclr2027.generate_paid_docker_repairs import (
    DEFINITIONS,
    GEMINI_SOURCE_RUN_IDS,
    glm_source_run_ids,
    render,
    selected_source_runs,
)
from rxnhaystack.manifest import load_manifest


def test_paid_docker_repairs_are_exact_transport_only_continuations(tmp_path) -> None:
    expected_counts = {
        "iclr2027-gemini-paid-openrouter-docker-repair-v1": 8,
        "iclr2027-glm-paid-openrouter-docker-v1": 45,
    }
    for definition in DEFINITIONS:
        source_runs = selected_source_runs(definition)
        output = tmp_path / definition.output.name
        output.write_text(render(definition, source_runs), encoding="utf-8")
        repair = load_manifest(output)

        assert len(repair.runs) == expected_counts[definition.campaign]
        assert repair.campaign.name == definition.campaign
        assert {run.env["RXNHAYSTACK_SOURCE_RUN_ID"] for run in repair.runs} == set(
            definition.source_run_ids
        )
        for source, recovered in zip(source_runs, repair.runs, strict=True):
            assert recovered.model == definition.model
            assert recovered.env["RXNHAYSTACK_PROVIDER"] == "openrouter"
            assert recovered.env["RXNHAYSTACK_RLM_OUTPUT_LIMIT"] == "4096"
            assert not any(key.startswith("RXNHAYSTACK_SWISSAI_") for key in recovered.env)
            assert recovered.command == source.command
            assert recovered.memory_reservation_mib == source.memory_reservation_mib
            assert recovered.memory_limit_mib == source.memory_limit_mib


def test_exact_repair_cardinalities() -> None:
    assert len(GEMINI_SOURCE_RUN_IDS) == len(set(GEMINI_SOURCE_RUN_IDS)) == 8
    assert len(glm_source_run_ids()) == len(set(glm_source_run_ids())) == 45
