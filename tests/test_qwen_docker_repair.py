from __future__ import annotations

from experiments.iclr2027.generate_qwen_docker_repair import (
    CAMPAIGN,
    OPENROUTER_MODEL,
    OUTPUT,
    SOURCE_RUN_IDS,
    render,
    selected_source_runs,
)
from rxnhaystack.manifest import load_manifest


def test_qwen_docker_repair_is_exact_and_transport_only(tmp_path) -> None:
    source_runs = selected_source_runs()
    output = tmp_path / OUTPUT.name
    output.write_text(render(source_runs), encoding="utf-8")
    repair = load_manifest(output)

    assert repair.campaign.name == CAMPAIGN
    assert len(repair.runs) == len(SOURCE_RUN_IDS) == 6
    assert {run.env["RXNHAYSTACK_SOURCE_RUN_ID"] for run in repair.runs} == set(SOURCE_RUN_IDS)
    for source, recovered in zip(source_runs, repair.runs, strict=True):
        assert recovered.model == OPENROUTER_MODEL
        assert recovered.env["RXNHAYSTACK_PROVIDER"] == "openrouter"
        assert not any(key.startswith("RXNHAYSTACK_SWISSAI_") for key in recovered.env)
        assert recovered.command == source.command
        assert recovered.memory_reservation_mib == source.memory_reservation_mib
        assert recovered.memory_limit_mib == source.memory_limit_mib
        assert recovered.question_parallelism == source.question_parallelism
        assert (
            recovered.env["RXNHAYSTACK_RLM_OUTPUT_LIMIT"]
            == source.env["RXNHAYSTACK_RLM_OUTPUT_LIMIT"]
        )
