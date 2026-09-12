from __future__ import annotations

import runpy
from collections import Counter
from pathlib import Path

import pytest

from rxnhaystack.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "iclr2027" / "baseline-campaign.toml"


def test_baseline_campaign_is_complete_and_within_budget() -> None:
    generator = runpy.run_path(
        str(ROOT / "experiments" / "iclr2027" / "generate_baseline_campaign.py"),
        run_name="campaign_generator_test",
    )
    assert MANIFEST.read_text(encoding="utf-8") == generator["render"]()
    manifest = load_manifest(MANIFEST)
    assert len(manifest.runs) == 1_050
    assert manifest.estimated_cost_chf == pytest.approx(67.32, abs=0.01)
    assert manifest.estimated_cost_chf <= manifest.campaign.budget_chf == 1_500.0
    assert manifest.campaign.require_clean_git
    assert manifest.campaign.required_secrets == ("OPENROUTER_API_KEY", "WANDB_API_KEY")
    assert Counter(run.method for run in manifest.runs) == {
        "llm": 300,
        "codeact": 300,
        "rlm": 450,
    }


def test_baseline_campaign_commands_and_contexts_resolve() -> None:
    manifest = load_manifest(MANIFEST)
    contexts: dict[str, set[int]] = {"llm": set(), "codeact": set(), "rlm": set()}
    for run in manifest.runs:
        script = ROOT / run.command[-1]
        assert script.is_file(), run.run_id
        context = int(run.env["RXNHAYSTACK_CONTEXT_SIZE"])
        contexts[run.method].add(context)
        assert run.corpus_size == ("full" if context == -1 else context)
    assert contexts == {
        "llm": {100, 500},
        "codeact": {100, 500},
        "rlm": {-1, 100, 500},
    }

    selected_task_groups = {(run.task, run.method) for run in manifest.runs}
    for excluded in ("tier3/task11", "tier3/task12", "tier3/task16", "tier3/task19"):
        assert all(task != excluded for task, _method in selected_task_groups)
