from __future__ import annotations

import json
from pathlib import Path

from human_eval.analysis import analyze, cohen_kappa, krippendorff_alpha_nominal, set_scores
from human_eval.db import Store
from human_eval.exporting import export_zip
from human_eval.schema import ground_truth_answer_set, submitted_answer_set


def test_known_analysis_values():
    assert set_scores({"1", "2"}, {"2", "3"}) == {
        "exact_match": 0.0,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert cohen_kappa(["yes", "yes", "no", "no"], ["yes", "yes", "no", "no"]) == 1.0
    assert cohen_kappa(["yes", "yes", "no", "no"], ["no", "no", "yes", "yes"]) == -1.0
    assert krippendorff_alpha_nominal({"a": ["x", "x"], "b": ["y", "y"]}) == 1.0
    assert (
        set_scores(
            submitted_answer_set([["2", "1"]], "index_set"),
            ground_truth_answer_set([1, 2], "index_set"),
        )["exact_match"]
        == 1.0
    )


def test_export_analysis_round_trip(tmp_path: Path, tiny_bundle: Path):
    store = Store(tmp_path / "state.sqlite")
    user = store.profile()["annotator_id"]
    store.save_draft(
        user,
        "baseline",
        "rxh-t1-fixture",
        {
            "answer_exact": "987654321",
            "answer_entries": [["987654321"]],
            "confidence": "5",
            "offline_minutes": "2",
        },
        submit=True,
    )
    store.save_draft(
        user,
        "baseline",
        "rxh-t1-fixture",
        {"answer_exact": "wrong", "answer_entries": [["wrong"]]},
        submit=True,
    )
    bundle_manifest = json.loads((tiny_bundle / "manifest.json").read_text())
    export_path = tmp_path / "export.zip"
    export_path.write_bytes(export_zip(store, user, bundle_manifest))
    output = tmp_path / "analysis"
    summary = analyze(
        [export_path],
        tiny_bundle / "questions.jsonl",
        tiny_bundle / "admin/ground_truth.jsonl",
        output,
    )
    assert summary["baseline_macro"]["exact_match"] == 1.0
    assert summary["timing_totals"]["self_reported_offline_minutes"] == 2.0
    assert (output / "item_metrics.csv").is_file()
    assert (output / "summary.json").is_file()
