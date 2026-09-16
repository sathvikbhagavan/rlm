from __future__ import annotations

from pathlib import Path

import pytest

from experiments.iclr2027 import generate_flat_map_reduce_experiment
from rxnhaystack.manifest import load_manifest
from rxnhaystack.map_reduce_tasks import selected_questions

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "iclr2027"


def test_flat_map_reduce_experiment_is_generated_and_small() -> None:
    path = EXPERIMENT_DIR / "flat-map-reduce-experiment.toml"
    assert path.read_text(encoding="utf-8") == generate_flat_map_reduce_experiment.render()
    manifest = load_manifest(path)

    assert len(manifest.runs) == 30
    assert {run.model for run in manifest.runs} == {
        "RCP-AIaaS/Qwen/Qwen3.5-397B-A17B",
        "google/gemini-3.7-flash",
    }
    assert {run.task for run in manifest.runs} == {
        "tier1/task1",
        "tier2/task5",
        "tier3/task10",
    }
    assert {run.method for run in manifest.runs} == {"map-reduce"}
    assert {run.corpus_size for run in manifest.runs} == {"full"}
    assert manifest.estimated_cost_chf == pytest.approx(180.0)
    assert manifest.campaign.budget_chf == 200.0
    assert all((ROOT / run.command[-1]).is_file() for run in manifest.runs)
    assert all(run.env["RXNHAYSTACK_MAP_REDUCE_CHUNK_SIZE"] == "500" for run in manifest.runs)
    assert {run.env["RXNHAYSTACK_MAP_REDUCE_QUESTION_ID"] for run in manifest.runs} == set(
        selected_questions()
    )
    swissai = [run for run in manifest.runs if run.env["RXNHAYSTACK_PROVIDER"] == "swissai"]
    assert {run.env["RXNHAYSTACK_SWISSAI_RATE_LIMIT_RETRIES"] for run in swissai} == {"0"}


def test_flat_map_reduce_call_arithmetic() -> None:
    generator = generate_flat_map_reduce_experiment

    assert generator.CHUNKS_PER_FULL_CORPUS == 245
    assert len(generator.MODELS) * len(generator.QUESTIONS) * generator.REPETITIONS == 30
    assert (
        len(generator.MODELS)
        * len(generator.QUESTIONS)
        * generator.REPETITIONS
        * generator.CHUNKS_PER_FULL_CORPUS
        == 7_350
    )
