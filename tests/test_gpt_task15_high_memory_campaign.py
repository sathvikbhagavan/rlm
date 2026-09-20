from pathlib import Path

from rxnhaystack.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/iclr2027/gpt-task15-high-memory-campaign.toml"


def test_high_memory_recovery_is_exact_and_bounded() -> None:
    manifest = load_manifest(MANIFEST)

    assert len(manifest.runs) == 1
    run = manifest.runs[0]
    assert run.run_id.endswith("tier4-task15-rlm-xfull-r04")
    assert run.env["RXNHAYSTACK_SOURCE_RUN_ID"] == ("full-gpt-5-mini-tier4-task15-rlm-xfull-r04")
    assert run.env["RXNHAYSTACK_PROVIDER"] == "openai"
    assert run.env["RXNHAYSTACK_RLM_LOCAL_MEMORY_LIMIT_MIB"] == "24576"
    assert run.env["RXNHAYSTACK_RLM_LOCAL_TOOL_MEMORY_LIMIT_MIB"] == "12288"
    assert run.memory_limit_mib == 30720
    assert run.memory_reservation_mib == 28672
    assert manifest.campaign.budget_chf == 5.0
