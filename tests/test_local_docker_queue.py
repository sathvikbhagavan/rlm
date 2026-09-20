from __future__ import annotations

from experiments.iclr2027.run_local_docker_queue import (
    MAIN_ROOT,
    PHASES,
    selected_runs,
)
from rxnhaystack.manifest import load_manifest


def test_local_docker_queue_has_exact_order_and_cardinalities() -> None:
    assert [phase.name for phase in PHASES] == [
        "qwen-repair",
        "gemini-repair",
        "prospective-task16",
        "deepseek-docker",
        "glm-docker",
    ]
    assert [phase.expected_runs for phase in PHASES] == [6, 8, 30, 24, 45]


def test_local_docker_queue_selectors_are_disjoint_and_exact() -> None:
    for phase in PHASES:
        path = phase.manifest_path
        if not path.is_file():
            path = MAIN_ROOT / phase.manifest_relative
        manifest = load_manifest(path)
        runs = selected_runs(phase, manifest)
        assert len(runs) == phase.expected_runs
        assert len({run.run_id for run in runs}) == phase.expected_runs
        assert all(run.method == "rlm" for run in runs)
        if phase.name != "prospective-task16":
            assert all(
                run.task in {"tier4/task16", "tier4/task17", "tier4/task17b"} for run in runs
            )


def test_local_queue_never_requests_failed_retries() -> None:
    source = MAIN_ROOT / "experiments/iclr2027/run_local_docker_queue.py"
    text = source.read_text(encoding="utf-8")
    assert "--retry-failed" not in text
