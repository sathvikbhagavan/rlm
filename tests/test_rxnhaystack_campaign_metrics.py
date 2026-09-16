from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rxnhaystack.campaign_metrics import install_campaign_metrics
from rxnhaystack.resources import rlm_trace_callbacks


class FakeWandb:
    def __init__(self) -> None:
        self.run = SimpleNamespace(summary={}, url="https://wandb.invalid/run")
        self.finished = False

    def init(self, *args, **kwargs):
        return self.run

    def log(self, data, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        self.finished = True


class WandbSummaryLike:
    """Mimic W&B Summary: `_as_dict()` exists but `.items()` does not."""

    def __init__(self) -> None:
        self.values = {"macro_f1": 0.625, "_runtime": 10}

    def __getattr__(self, key: str):
        raise KeyError(key)

    def _as_dict(self) -> dict[str, float]:
        return dict(self.values)


def configure_campaign(monkeypatch, tmp_path: Path, *, method: str) -> Path:
    metrics = tmp_path / "metrics.json"
    monkeypatch.setenv("RXNHAYSTACK_RUN_ID", "test-run")
    monkeypatch.setenv("RXNHAYSTACK_METHOD", method)
    monkeypatch.setenv("RXNHAYSTACK_METRICS_PATH", str(metrics))
    monkeypatch.setenv("RXNHAYSTACK_USD_TO_CHF", "0.8")
    monkeypatch.setenv("RXNHAYSTACK_RESOURCE_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv(
        "RXNHAYSTACK_TRAJECTORY_EVENTS_PATH", str(tmp_path / "trajectory-events.jsonl")
    )
    return metrics


def test_capture_writes_complete_llm_metrics(monkeypatch, tmp_path: Path) -> None:
    metrics_path = configure_campaign(monkeypatch, tmp_path, method="llm")
    wandb = FakeWandb()
    capture = install_campaign_metrics(wandb)
    assert capture is not None
    run = wandb.init(project="test")
    run.summary["macro_f1"] = 0.75
    for sample, cost in ((0, 0.1), (1, 0.2)):
        wandb.log(
            {
                f"sample/{sample}/iteration_total_tokens": 30,
                f"sample/{sample}/final_total_input_tokens": 20,
                f"sample/{sample}/final_total_output_tokens": 10,
                f"sample/{sample}/final_total_tokens": 30,
                f"sample/{sample}/final_total_cost_usd": cost,
            }
        )
    wandb.finish()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["calls"] == 2
    assert metrics["input_tokens"] == 40
    assert metrics["output_tokens"] == 20
    assert metrics["cost_usd"] == pytest.approx(0.3)
    assert metrics["cost_chf"] == pytest.approx(0.24)
    assert metrics["results"]["macro_f1"] == 0.75
    assert wandb.finished


def test_capture_reads_current_wandb_summary_object(monkeypatch, tmp_path: Path) -> None:
    metrics_path = configure_campaign(monkeypatch, tmp_path, method="llm")
    wandb = FakeWandb()
    wandb.run.summary = WandbSummaryLike()
    install_campaign_metrics(wandb)
    wandb.init(project="test")
    wandb.log(
        {
            "sample/0/iteration_total_tokens": 3,
            "sample/0/final_total_input_tokens": 2,
            "sample/0/final_total_output_tokens": 1,
            "sample/0/final_total_tokens": 3,
            "sample/0/final_total_cost_usd": 0.01,
        }
    )

    wandb.finish()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["results"]["macro_f1"] == 0.625
    assert metrics["results"]["accounting"]["status"] == "available"


@pytest.mark.parametrize(
    ("stopped_by_timeout", "expected_finalizations"),
    [(False, 0), (True, 1)],
)
def test_capture_uses_exact_recursive_trace_metrics(
    monkeypatch,
    tmp_path: Path,
    stopped_by_timeout: bool,
    expected_finalizations: int,
) -> None:
    metrics_path = configure_campaign(monkeypatch, tmp_path, method="rlm")
    trace_path = Path(str(tmp_path / "trace.jsonl"))
    callbacks = rlm_trace_callbacks(trace_path, sample_id="question-0")
    callbacks["on_completion_metrics"](
        {
            "calls": 3,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.25,
            "execution_time_seconds": 4.0,
            "model_time_seconds": 2.0,
            "tool_time_seconds": 1.5,
            "stopped_by_timeout": stopped_by_timeout,
        }
    )
    wandb = FakeWandb()
    install_campaign_metrics(wandb)
    wandb.init(project="test")
    wandb.log(
        {
            "sample/0/iteration_total_tokens": 30,
            "sample/0/final_total_input_tokens": 20,
            "sample/0/final_total_output_tokens": 10,
            "sample/0/final_total_tokens": 30,
            "sample/0/final_total_cost_usd": 0.25,
        }
    )
    wandb.finish()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["calls"] == 3
    assert metrics["tool_time_seconds"] == 1.5
    assert metrics["results"]["rlm_timeout_finalizations"] == expected_finalizations


def test_capture_preserves_score_with_unknown_provider_cost(monkeypatch, tmp_path: Path) -> None:
    metrics_path = configure_campaign(monkeypatch, tmp_path, method="llm")
    wandb = FakeWandb()
    install_campaign_metrics(wandb)
    wandb.init(project="test", config={"num_questions": 1})
    wandb.log(
        {
            "sample/0/iteration_total_tokens": 30,
            "sample/0/final_total_input_tokens": 20,
            "sample/0/final_total_output_tokens": 10,
            "sample/0/final_total_tokens": 30,
            "sample/0/f1": 0.5,
        }
    )

    wandb.finish()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["cost_usd"] is None
    assert metrics["cost_chf"] is None
    assert metrics["accounting_status"] == "unavailable"
    assert metrics["estimated_cost_chf"] == 0
    trajectory = (tmp_path / "trajectory-events.jsonl").read_text()
    assert '"sample/0/f1":0.5' in trajectory
    assert wandb.finished


def test_capture_rejects_partial_multi_trajectory_job(monkeypatch, tmp_path: Path) -> None:
    configure_campaign(monkeypatch, tmp_path, method="rlm")
    wandb = FakeWandb()
    install_campaign_metrics(wandb)
    wandb.init(project="test", config={"num_questions": 2})
    wandb.log(
        {
            "sample/0/final_total_input_tokens": 20,
            "sample/0/final_total_output_tokens": 10,
            "sample/0/final_total_tokens": 30,
            "sample/0/f1": 0.5,
        }
    )

    with pytest.raises(RuntimeError, match="Incomplete trajectory set: 1/2"):
        wandb.finish()
