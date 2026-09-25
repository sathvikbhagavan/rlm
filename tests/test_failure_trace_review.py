import csv
from pathlib import Path

from paper_plots.scripts.build_failure_trace_review import (
    canonical_run_id,
    eligible_wrong_answer,
    signal_counts,
)
from paper_plots.scripts.fetch_failure_review_traces import parse_wandb_url


def test_canonical_run_id_accepts_repair_prefix() -> None:
    known = ("full-qwen-tier4-task16-rlm-xfull-r05",)
    assert (
        canonical_run_id("paid-openrouter-full-qwen-tier4-task16-rlm-xfull-r05", known) == known[0]
    )


def test_pending_or_failed_run_is_not_a_scientific_wrong_answer() -> None:
    row = {
        "run_id": "run",
        "status": "succeeded",
        "method": "rlm",
        "tier": "4",
        "f1": "0.0",
    }
    assert not eligible_wrong_answer(row, {"run"})
    row["status"] = "failed"
    assert not eligible_wrong_answer(row, set())


def test_signals_are_evidence_retrieval_not_mutually_exclusive_labels() -> None:
    text = (
        "Boost.Python.ArgumentError while calling RunReactants; workflow timeout\n"
        '<invoke name="python">\nLoaded 0 reactions\nANSWER: []'
    )
    signals = signal_counts(text)
    assert signals["rdkit_runtime_error"] == 1
    assert signals["generated_smarts"] == 1
    assert signals["iteration_or_time_limit"] == 1
    assert signals["unsupported_action_wrapper"] == 1
    assert signals["zero_record_parse"] == 1
    assert signals["placeholder_or_empty_answer"] == 1


def test_parse_wandb_url_returns_api_path() -> None:
    assert (
        parse_wandb_url("https://wandb.ai/liac/CodeAct-Task1/runs/yfd5qypg")
        == "liac/CodeAct-Task1/yfd5qypg"
    )


def test_reviewed_causes_reference_valid_unaffected_runs() -> None:
    root = Path(__file__).resolve().parents[1] / "paper_plots/gold/iclr2027"
    with (root / "final_arm_records.csv").open(newline="") as stream:
        records = {row["run_id"]: row for row in csv.DictReader(stream)}
    with (root / "post_submission/pending_corrected_rescores.csv").open(newline="") as stream:
        pending = {row["run_id"] for row in csv.DictReader(stream)}
    with (root / "failure_analysis/reviewed_trace_causes.csv").open(newline="") as stream:
        reviewed = list(csv.DictReader(stream))

    assert reviewed
    assert len({row["run_id"] for row in reviewed}) == len(reviewed)
    for row in reviewed:
        source = records[row["run_id"]]
        assert row["run_id"] not in pending
        assert source["model_label"] == row["model"]
        assert source["method"] == row["method"]
        assert source["task"] == row["task"]
        assert source["context"] == row["context"]
        assert row["confidence"] in {"high", "medium", "low"}
        assert row["review_status"] == "reviewed"
        assert row["direct_evidence"]


def test_diagnostic_map_uses_reviewed_examples_and_explicit_controls() -> None:
    root = Path(__file__).resolve().parents[1] / "paper_plots/gold/iclr2027/failure_analysis"
    with (root / "reviewed_trace_causes.csv").open(newline="") as stream:
        reviewed = {row["run_id"] for row in csv.DictReader(stream)}
    with (root / "diagnostic_objective_map.csv").open(newline="") as stream:
        mappings = list(csv.DictReader(stream))

    capabilities = {row["benchmark_capability"] for row in mappings}
    assert {
        "faithful structured execution",
        "chemical abstraction",
        "orchestration and state preservation",
        "relational semantics and graph construction",
        "prospective target interpretation and route construction",
    }.issubset(capabilities)
    for row in mappings:
        examples = row["representative_runs"].split(";")
        assert all(run_id in reviewed for run_id in examples)
        assert row["design_contrast"]
        assert row["supported_interpretation"]
