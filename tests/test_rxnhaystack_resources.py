from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from rxnhaystack.resources import (
    MemoryBudget,
    append_trace_event,
    docker_cgroup_memory_bytes,
    process_tree_rss_bytes,
    rlm_trace_callbacks,
    wait_with_memory_watchdog,
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


def test_docker_cgroup_memory_sums_live_unique_registered_containers(tmp_path: Path) -> None:
    cgroup_root = tmp_path / "cgroup"
    first = cgroup_root / "system.slice/docker-first.scope"
    second = cgroup_root / "system.slice/docker-second.scope"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "memory.current").write_text("10485760\n")
    (second / "memory.current").write_text("20971520\n")
    registry = tmp_path / "docker-cgroups.txt"
    registry.write_text(
        "/system.slice/docker-first.scope\n"
        "/system.slice/docker-second.scope\n"
        "/system.slice/docker-first.scope\n"
        "../../outside\n"
    )

    assert docker_cgroup_memory_bytes(registry, cgroup_root=cgroup_root) == 30 * 1024 * 1024


def test_memory_watchdog_records_host_docker_and_combined_peaks(
    tmp_path: Path, monkeypatch
) -> None:
    docker_bytes = 20 * 1024 * 1024
    monkeypatch.setattr(
        "rxnhaystack.resources.docker_cgroup_memory_bytes",
        lambda _path: docker_bytes,
    )
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(1.0)"],
        text=True,
        start_new_session=True,
    )
    usage = wait_with_memory_watchdog(
        process,
        memory_limit_mib=None,
        poll_interval_seconds=0.02,
        docker_memory_registry_path=tmp_path / "registry",
    )

    assert usage.return_code == 0
    assert usage.peak_host_rss_mib > 0
    assert usage.peak_docker_memory_mib == 20
    assert usage.peak_combined_memory_mib >= usage.peak_host_rss_mib + 20


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
            "code_block_count": 1,
            "had_error": True,
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
            "stopped_by_timeout": True,
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
    assert events[-2]["code_block_count"] == 1
    assert events[-2]["had_error"] is True
    assert events[-1]["calls"] == 3
    assert events[-1]["tool_time_seconds"] == 1.5
    assert events[-1]["stopped_by_timeout"] is True
    assert "secret prompt" not in text
    assert "private error" not in text
