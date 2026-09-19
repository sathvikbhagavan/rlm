from __future__ import annotations

import subprocess
from pathlib import Path

from rxnhaystack.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "experiments/iclr2027/generate_gpt_direct_openai_recovery_campaign.py"
MANIFEST = ROOT / "experiments/iclr2027/gpt-direct-openai-recovery-campaign.toml"


def test_recovery_campaign_is_current_and_exact() -> None:
    subprocess.run(["uv", "run", "--frozen", "python", str(GENERATOR), "--check"], check=True)
    manifest = load_manifest(MANIFEST)
    runs = manifest.runs

    assert len(runs) == 21
    assert manifest.campaign.required_secrets == ("OPENAI_API_KEY", "WANDB_API_KEY")
    assert {run.task for run in runs} == {"tier4/task13", "tier4/task14", "tier4/task15"}
    assert {run.env["RXNHAYSTACK_PROVIDER"] for run in runs} == {"openai"}
    assert all(run.model == "gpt-5-mini" for run in runs)
    assert sum(run.task == "tier4/task13" for run in runs) == 2
    assert sum(run.task == "tier4/task14" for run in runs) == 4
    assert sum(run.task == "tier4/task15" for run in runs) == 15
    task13 = [run for run in runs if run.task == "tier4/task13"]
    assert {run.env["RXNHAYSTACK_RLM_LOCAL_TOOL_MEMORY_LIMIT_MIB"] for run in task13} == {
        "8192"
    }
    assert len({run.env["RXNHAYSTACK_SOURCE_RUN_ID"] for run in runs}) == 21
