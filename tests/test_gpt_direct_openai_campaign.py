from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rxnhaystack.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "experiments/iclr2027/generate_gpt_direct_openai_docker_campaign.py"
MANIFEST = ROOT / "experiments/iclr2027/gpt-direct-openai-docker-campaign.toml"
RUNNER = ROOT / "experiments/iclr2027/run_gpt_direct_openai_docker.sh"


def test_direct_openai_docker_campaign_is_current_and_exact() -> None:
    subprocess.run(["uv", "run", "--frozen", "python", str(GENERATOR), "--check"], check=True)
    manifest = load_manifest(MANIFEST)
    expanded = manifest.runs

    assert len(expanded) == 45
    assert manifest.estimated_cost_chf == pytest.approx(7.14878, abs=0.01)
    assert manifest.campaign.budget_chf == 30
    assert manifest.campaign.required_secrets == ("OPENAI_API_KEY", "WANDB_API_KEY")
    assert {run.task for run in expanded} == {
        "tier4/task16",
        "tier4/task17",
        "tier4/task17b",
    }
    assert {run.corpus_size for run in expanded} == {100, 500, "full"}
    assert all(run.model == "gpt-5-mini" for run in expanded)
    assert all(run.env["RXNHAYSTACK_PROVIDER"] == "openai" for run in expanded)
    assert len({run.run_id for run in expanded}) == 45


def test_direct_openai_docker_runner_has_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
