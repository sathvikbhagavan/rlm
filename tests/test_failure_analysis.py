from paper_plots.scripts.build_failure_analysis import error_signature, outcome


def test_error_signature_distinguishes_set_error_shapes() -> None:
    base = {"corrected_f1": 0.5, "predicted_count": 2}
    assert (
        error_signature(base | {"corrected_precision": 1, "corrected_recall": 0.5})
        == "omissions only"
    )
    assert (
        error_signature(base | {"corrected_precision": 0.5, "corrected_recall": 1})
        == "extra selections only"
    )
    assert (
        error_signature(base | {"corrected_precision": 0.5, "corrected_recall": 0.5})
        == "mixed omissions and extras"
    )
    assert (
        error_signature(
            base | {"predicted_count": 0, "corrected_precision": 0, "corrected_recall": 0}
        )
        == "empty answer"
    )


def test_outcome_keeps_execution_failure_separate() -> None:
    assert outcome({"status": "failed", "f1": "0"}) == "execution failure"
    assert outcome({"status": "succeeded", "f1": "0"}) == "zero score"
    assert outcome({"status": "succeeded", "f1": "0.5"}) == "partial"
    assert outcome({"status": "succeeded", "f1": "1"}) == "exact"
