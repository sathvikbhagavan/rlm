from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rxnhaystack.campaign_metrics import SCIENTIFIC_SCORE_FIELDS, install_campaign_metrics
from rxnhaystack.resources import rlm_trace_callbacks


class FakeWandb:
    def __init__(self) -> None:
        self.run = SimpleNamespace(summary={}, url="https://wandb.invalid/run")
        self.finished = False
        self.init_kwargs = {}

    def init(self, *args, **kwargs):
        self.init_kwargs = kwargs
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


def final_usage(sample: int = 0) -> dict[str, float]:
    return {
        f"sample/{sample}/iteration_total_tokens": 30,
        f"sample/{sample}/final_total_input_tokens": 20,
        f"sample/{sample}/final_total_output_tokens": 10,
        f"sample/{sample}/final_total_tokens": 30,
        f"sample/{sample}/final_total_cost_usd": 0.01,
    }


def test_full_and_matched_scoring_field_audit_is_registered() -> None:
    # Inventory from every sample/* score emitted by the full-campaign runners,
    # including the same Tier 1--3 runners reused by matched cardinality.
    audited_fields = {
        "count_exact",
        "exact_set_match",
        "f1",
        "index_match",
        "is_correct",
        "is_exact_match",
        "lcs_ratio",
        "normalized_edit_distance",
        "normalized_lcs",
        "objective_length_match",
        "position_accuracy",
        "precision",
        "prefix_match_ratio",
        "reaction_f1",
        "reaction_precision",
        "reaction_recall",
        "recall",
        "valid_path",
    }

    assert audited_fields <= SCIENTIFIC_SCORE_FIELDS


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


def test_capture_links_retry_to_source_run_in_wandb(monkeypatch, tmp_path: Path) -> None:
    configure_campaign(monkeypatch, tmp_path, method="rlm")
    monkeypatch.setenv("RXNHAYSTACK_SOURCE_RUN_ID", "original-run-r02")
    wandb = FakeWandb()
    install_campaign_metrics(wandb)

    wandb.init(project="test")

    assert wandb.init_kwargs["config"]["rxnhaystack_source_run_id"] == "original-run-r02"


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


def test_capture_accepts_actual_task14_metric_shape(monkeypatch, tmp_path: Path) -> None:
    metrics_path = configure_campaign(monkeypatch, tmp_path, method="rlm")
    monkeypatch.setenv("RXNHAYSTACK_TASK", "tier4/task14")
    wandb = FakeWandb()
    install_campaign_metrics(wandb)
    wandb.init(project="test", config={"num_questions": 1})
    payload = {
        "sample_idx": 0,
        "sample/0/pg_label": "Boc_N",
        "sample/0/functional_group": "Boc-protected nitrogen",
        "sample/0/ground_truth_count": 2,
        "sample/0/pred_pair_count": 1,
        "sample/0/precision": 1.0,
        "sample/0/recall": 0.5,
        "sample/0/f1": 2 / 3,
        "sample/0/exact_set_match": 0.0,
        **final_usage(),
    }
    wandb.log(payload)

    wandb.finish()

    assert json.loads(metrics_path.read_text())["results"]["accounting"]["status"] == "available"


def test_capture_accepts_actual_task15_metric_shape(monkeypatch, tmp_path: Path) -> None:
    metrics_path = configure_campaign(monkeypatch, tmp_path, method="rlm")
    monkeypatch.setenv("RXNHAYSTACK_TASK", "tier4/task15")
    wandb = FakeWandb()
    install_campaign_metrics(wandb)
    wandb.init(project="test", config={"num_questions": 1})
    payload = {
        "sample_idx": 0,
        "sample/0/ring_system": "quinoline",
        "sample/0/ground_truth_count": 1,
        "sample/0/pred_reaction_indices": "1015,1016,1017",
        "sample/0/validity_reason": "ok",
        "sample/0/is_correct": 1.0,
        "sample/0/valid_path": 1.0,
        "sample/0/index_match": 1.0,
        "sample/0/objective_length_match": 1.0,
        "sample/0/reaction_precision": 1.0,
        "sample/0/reaction_recall": 1.0,
        "sample/0/reaction_f1": 1.0,
        "sample/0/normalized_lcs": 1.0,
        **final_usage(),
    }
    wandb.log(payload)

    wandb.finish()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["calls"] == 1
    assert metrics["results"]["accounting"]["status"] == "available"
    assert {"reaction_f1", "is_correct"} <= SCIENTIFIC_SCORE_FIELDS


@pytest.mark.parametrize(
    ("task", "score_fields"),
    [
        ("tier2/task3", {"precision": 0.5, "recall": 1.0, "f1": 2 / 3}),
        ("tier3/task10b", {"precision": 1.0, "recall": 1.0, "f1": 1.0}),
    ],
)
def test_capture_accepts_representative_matched_metric_shapes(
    monkeypatch, tmp_path: Path, task: str, score_fields: dict[str, float]
) -> None:
    configure_campaign(monkeypatch, tmp_path, method="rlm")
    monkeypatch.setenv("RXNHAYSTACK_TASK", task)
    wandb = FakeWandb()
    install_campaign_metrics(wandb)
    wandb.init(project="test", config={"num_questions": 1})
    wandb.log(
        {
            **{f"sample/0/{key}": value for key, value in score_fields.items()},
            "sample/0/ground_truth_count": 5,
            "sample/0/predicted_count": 5,
            **final_usage(),
        }
    )

    wandb.finish()

    assert wandb.finished


def test_capture_rejects_nonfinite_or_diagnostic_only_score(monkeypatch, tmp_path: Path) -> None:
    configure_campaign(monkeypatch, tmp_path, method="rlm")
    wandb = FakeWandb()
    install_campaign_metrics(wandb)
    wandb.init(project="test", config={"num_questions": 1})
    wandb.log(
        {
            "sample/0/f1": float("nan"),
            "sample/0/validity_reason": "not a scientific score",
            **final_usage(),
        }
    )

    with pytest.raises(RuntimeError, match="Unscored model responses"):
        wandb.finish()


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
