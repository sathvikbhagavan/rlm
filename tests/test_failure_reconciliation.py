import csv
from pathlib import Path


def test_manual_trace_audit_covers_paper_models_and_interfaces() -> None:
    path = Path("paper_plots/gold/iclr2027/failure_analysis/manual_trace_audit.csv")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["model"] for row in rows} == {
        "Qwen3.5-397B-A17B",
        "DeepSeek-V4-Flash-0731",
        "Gemini 3.7 Flash",
        "GPT-5 mini",
        "Claude Haiku 4.5",
    }
    assert {row["method"] for row in rows} == {"CodeAct", "RLM"}
    assert all(row["review_status"] == "manually verified" for row in rows)
    assert all(len(row["source_sha256"]) == 64 for row in rows)


def test_corrected_tasks_use_valid_rescores() -> None:
    path = Path("paper_plots/gold/iclr2027/failure_analysis/manual_trace_audit.csv")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    corrected = {"tier3/task6", "tier3/task18", "tier4/task15"}
    assert all(
        row["correction_status"] == "corrected_exact_rescore"
        for row in rows
        if row["task"] in corrected
    )
