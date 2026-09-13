from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from experiments.iclr2027 import (
    generate_full_campaign,
    generate_matched_cardinality_campaign,
)
from rxnhaystack.manifest import ManifestError, load_manifest
from rxnhaystack.runtime import resolve_required_secrets

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / "experiments" / "iclr2027"
REQUIRED_SECRETS = (
    "OPENROUTER_API_KEY",
    "SWISSAI_RESEARCH_API_KEY",
    "WANDB_API_KEY",
)


def test_full_campaign_is_generated_and_covers_six_models() -> None:
    path = CAMPAIGN_DIR / "full-campaign.toml"
    assert path.read_text(encoding="utf-8") == generate_full_campaign.render()
    manifest = load_manifest(path)

    assert len(manifest.runs) == 6_300
    assert len({run.model for run in manifest.runs}) == 6
    assert manifest.estimated_cost_chf == pytest.approx(743.72, abs=0.02)
    assert "anthropic/claude-haiku-4.5" in {run.model for run in manifest.runs}
    assert "anthropic/claude-sonnet-5" not in {run.model for run in manifest.runs}
    assert manifest.campaign.required_secrets == REQUIRED_SECRETS
    assert Counter(run.env["RXNHAYSTACK_PROVIDER"] for run in manifest.runs) == {
        "swissai": 3_150,
        "openrouter": 3_150,
    }
    rlm_runs = [run for run in manifest.runs if run.method == "rlm"]
    assert {run.env["RXNHAYSTACK_RLM_MAX_TIMEOUT_SECONDS"] for run in rlm_runs} == {"1800"}
    codeact_runs = [run for run in manifest.runs if run.method == "codeact"]
    assert {
        run.env["RXNHAYSTACK_CODEACT_OUTPUT_LIMIT"]
        for run in codeact_runs
        if run.model == "CSCS-Inference/zai-org/GLM-5.2"
    } == {"8192"}
    assert {
        run.env["RXNHAYSTACK_CODEACT_OUTPUT_LIMIT"]
        for run in codeact_runs
        if run.model != "CSCS-Inference/zai-org/GLM-5.2"
    } == {"30000"}
    assert {
        run.question_parallelism
        for run in codeact_runs
        if run.model == "CSCS-Inference/zai-org/GLM-5.2"
    } == {1}
    llm_runs = [run for run in manifest.runs if run.method == "llm"]
    swissai_llm_runs = [run for run in llm_runs if run.env["RXNHAYSTACK_PROVIDER"] == "swissai"]
    openrouter_llm_runs = [run for run in llm_runs if run not in swissai_llm_runs]
    assert {run.question_parallelism for run in swissai_llm_runs} == {1}
    assert {run.question_parallelism for run in openrouter_llm_runs} == {4}
    assert {run.env["RXNHAYSTACK_LLM_OUTPUT_LIMIT"] for run in swissai_llm_runs} == {"4096"}
    assert all("RXNHAYSTACK_LLM_OUTPUT_LIMIT" not in run.env for run in openrouter_llm_runs)

    with pytest.raises(ManifestError, match="SWISSAI_RESEARCH_API_KEY"):
        resolve_required_secrets(
            manifest,
            {},
            environ={"OPENROUTER_API_KEY": "x", "WANDB_API_KEY": "y"},
        )


def test_matched_cardinality_campaign_has_two_factor_design() -> None:
    path = CAMPAIGN_DIR / "matched-cardinality-campaign.toml"
    assert path.read_text(encoding="utf-8") == (generate_matched_cardinality_campaign.render())
    manifest = load_manifest(path)

    assert len(manifest.runs) == 1_450
    assert {run.model for run in manifest.runs} == {
        "RCP-AIaaS/Qwen/Qwen3.5-397B-A17B",
        "openai/gpt-5-mini",
    }
    expected_tasks = {
        f"{tier}/task{task_id}"
        for tier, task_ids in generate_matched_cardinality_campaign.ELIGIBLE_TASKS.items()
        for task_id in task_ids
    }
    assert {run.task for run in manifest.runs} == expected_tasks
    assert len(expected_tasks) == 21
    assert not any(task.startswith("tier4/") for task in expected_tasks)
    assert manifest.estimated_cost_chf == pytest.approx(70.98, abs=0.02)
    assert manifest.campaign.required_secrets == REQUIRED_SECRETS

    scale = [run for run in manifest.runs if run.condition.startswith("scale-")]
    assert {run.corpus_size for run in scale} == {100, 500, 5_000, 50_000, "full"}
    assert {run.positive_cardinality for run in scale} == {1}
    assert {run.task for run in scale} == expected_tasks
    cardinality = [run for run in manifest.runs if run.condition.startswith("cardinality-")]
    assert {run.corpus_size for run in cardinality} == {5_000}
    assert {run.positive_cardinality for run in cardinality} == {5, 20}
    assert all(not run.task.startswith("tier1/") for run in cardinality)
    assert all((ROOT / run.command[-1]).is_file() for run in manifest.runs)

    for run in manifest.runs:
        tier, task = run.task.split("/task", 1)
        assert run.positive_cardinality is not None
        assert (
            run.positive_cardinality
            <= generate_matched_cardinality_campaign.MIN_POSITIVES[tier][task]
        )
        assert run.env["RXNHAYSTACK_RLM_MAX_TIMEOUT_SECONDS"] == "1800"
        if run.env["RXNHAYSTACK_PROVIDER"] == "openrouter":
            assert run.env["RXNHAYSTACK_RLM_OUTPUT_LIMIT"] == "4096"
            assert run.env["RXNHAYSTACK_RLM_REASONING_EFFORT"] == "low"
        else:
            assert run.env["RXNHAYSTACK_RLM_OUTPUT_LIMIT"] == "2048"
