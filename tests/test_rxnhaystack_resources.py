from __future__ import annotations

import json
import threading
from pathlib import Path

from rxnhaystack.resources import (
    MemoryBudget,
    append_trace_event,
    process_tree_rss_bytes,
    rlm_trace_callbacks,
)


def write_status(path: Path, *, pid: int, parent_pid: int, rss_kib: int) -> None:
    path.mkdir(parents=True)
    (path / "status").write_text(
        f"Name:\ttest\nPid:\t{pid}\nPPid:\t{parent_pid}\nVmRSS:\t{rss_kib} kB\n"
    )


def test_process_tree_rss_includes_descendants_but_not_unrelated_processes(
    tmp_path: Path,
) -> None:
    write_status(tmp_path / "100", pid=100, parent_pid=1, rss_kib=10)
    write_status(tmp_path / "101", pid=101, parent_pid=100, rss_kib=20)
    write_status(tmp_path / "102", pid=102, parent_pid=101, rss_kib=30)
    write_status(tmp_path / "200", pid=200, parent_pid=1, rss_kib=1000)

    assert process_tree_rss_bytes(100, proc_root=tmp_path) == 60 * 1024


def test_memory_budget_blocks_until_a_reservation_is_released() -> None:
    budget = MemoryBudget(100)
    first_acquired = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()

    def first() -> None:
        with budget.reserve(80):
            first_acquired.set()
            release_first.wait(timeout=2)

    def second() -> None:
        first_acquired.wait(timeout=2)
        with budget.reserve(30):
            second_acquired.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_acquired.wait(timeout=2)
    assert not second_acquired.wait(timeout=0.05)
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert second_acquired.is_set()


def test_resource_trace_combines_samples_and_prompt_free_rlm_events(tmp_path: Path) -> None:
    trace = tmp_path / "resource-trace.jsonl"
    append_trace_event(trace, "resource_sample", process_tree_rss_mib=123.5)
    callbacks = rlm_trace_callbacks(trace, sample_id="question-2")
    callbacks["on_iteration_start"](0, 3)
    callbacks["on_subcall_start"](1, "provider/model", "secret prompt must not be logged")
    callbacks["on_subcall_complete"](1, "provider/model", 2.5, "private error detail")
    callbacks["on_iteration_complete"](0, 3, 4.5)
    callbacks["on_iteration_metrics"](
        0,
        3,
        {
            "iteration_calls": 2,
            "iteration_input_tokens": 100,
            "iteration_output_tokens": 20,
            "iteration_cost_usd": 0.01,
            "model_time_s": 2.0,
            "tool_time_s": 1.0,
        },
    )
    callbacks["on_completion_metrics"](
        {
            "calls": 3,
            "input_tokens": 150,
            "output_tokens": 30,
            "cost_usd": 0.02,
            "execution_time_seconds": 5.0,
            "model_time_seconds": 3.0,
            "tool_time_seconds": 1.5,
        }
    )

    text = trace.read_text()
    events = [json.loads(line) for line in text.splitlines()]
    assert [event["event"] for event in events] == [
        "resource_sample",
        "rlm_iteration_started",
        "rlm_subcall_started",
        "rlm_subcall_finished",
        "rlm_iteration_finished",
        "rlm_iteration_metrics",
        "rlm_completion_metrics",
    ]
    assert all("monotonic_seconds" in event for event in events)
    assert events[2]["sample_id"] == "question-2"
    assert events[3]["failed"] is True
    assert events[-1]["calls"] == 3
    assert events[-1]["tool_time_seconds"] == 1.5
    assert "secret prompt" not in text
    assert "private error" not in text
