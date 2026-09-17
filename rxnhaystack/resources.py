from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
DOCKER_MEMORY_CGROUP_ENV = "RXNHAYSTACK_DOCKER_MEMORY_CGROUP_PATH"


@dataclass(frozen=True)
class ProcessUsage:
    return_code: int
    peak_combined_memory_mib: float
    peak_host_rss_mib: float
    peak_docker_memory_mib: float
    memory_limit_exceeded: bool
    cancelled: bool = False
    wall_time_limit_exceeded: bool = False


class MemoryBudget:
    """A weighted semaphore for process-level memory reservations."""

    def __init__(self, capacity_mib: int | None) -> None:
        self.capacity_mib = capacity_mib
        self._available_mib = capacity_mib
        self._condition = threading.Condition()

    @contextmanager
    def reserve(self, amount_mib: int | None) -> Iterator[None]:
        if self.capacity_mib is None:
            yield
            return
        if amount_mib is None:
            raise ValueError("A memory reservation is required when a memory budget is active")
        if amount_mib > self.capacity_mib:
            raise ValueError(
                f"Memory reservation {amount_mib} MiB exceeds budget {self.capacity_mib} MiB"
            )
        with self._condition:
            while self._available_mib is not None and self._available_mib < amount_mib:
                self._condition.wait()
            assert self._available_mib is not None
            self._available_mib -= amount_mib
        try:
            yield
        finally:
            with self._condition:
                assert self._available_mib is not None
                self._available_mib += amount_mib
                self._condition.notify_all()


def process_tree_rss_bytes(root_pid: int, *, proc_root: Path = Path("/proc")) -> int:
    """Return aggregate RSS for a Linux process and all current descendants."""

    processes: dict[int, tuple[int, int]] = {}
    # Capture the root before scanning /proc. Very short workers can otherwise
    # exit while the directory is being traversed and incorrectly report a
    # zero peak despite being alive when sampling began.
    root_status = _read_linux_status(proc_root / str(root_pid) / "status")
    if root_status is not None:
        pid, parent_pid, rss_bytes = root_status
        processes[pid] = (parent_pid, rss_bytes)
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        if int(entry.name) == root_pid:
            continue
        status = _read_linux_status(entry / "status")
        if status is not None:
            pid, parent_pid, rss_bytes = status
            processes[pid] = (parent_pid, rss_bytes)

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _) in processes.items():
            if parent_pid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(processes[pid][1] for pid in descendants if pid in processes)


def _read_linux_status(path: Path) -> tuple[int, int, int] | None:
    fields: dict[str, str] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                key, separator, value = line.partition(":")
                if separator and key in {"Pid", "PPid", "VmRSS"}:
                    fields[key] = value.strip()
    except (OSError, UnicodeError):
        return None
    try:
        pid = int(fields["Pid"])
        parent_pid = int(fields["PPid"])
        rss_kib = int(fields.get("VmRSS", "0 kB").split()[0])
    except (KeyError, ValueError):
        return None
    return pid, parent_pid, rss_kib * 1024


def docker_cgroup_memory_bytes(
    registry_path: Path | None,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int:
    """Read a launcher-managed container's cgroup-v2 memory counter."""

    if registry_path is None:
        return 0
    try:
        cgroup_values = registry_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return 0
    root = cgroup_root.resolve()
    total = 0
    for cgroup_value in set(cgroup_values):
        if not cgroup_value:
            continue
        candidate = (root / cgroup_value.lstrip("/") / "memory.current").resolve()
        try:
            candidate.relative_to(root)
            value = int(candidate.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeError, ValueError):
            continue
        total += max(0, value)
    return total


def wait_with_memory_watchdog(
    process: subprocess.Popen[str],
    *,
    memory_limit_mib: int | None,
    poll_interval_seconds: float = 0.2,
    termination_grace_seconds: float = 5.0,
    trace_path: Path | None = None,
    run_id: str | None = None,
    cancellation_event: threading.Event | None = None,
    docker_memory_registry_path: Path | None = None,
    wall_time_limit_seconds: float | None = None,
) -> ProcessUsage:
    """Measure resources and enforce launcher-owned process limits."""

    if memory_limit_mib is not None and not Path("/proc/self/status").is_file():
        raise RuntimeError("Per-run memory limits require a Linux /proc filesystem")
    if wall_time_limit_seconds is not None and wall_time_limit_seconds <= 0:
        raise ValueError("wall_time_limit_seconds must be positive")
    started = time.monotonic()
    peak_rss_bytes = 0
    peak_host_rss_bytes = 0
    peak_docker_memory_bytes = 0
    memory_limit_bytes = memory_limit_mib * MIB if memory_limit_mib is not None else None
    exceeded = False
    cancelled = False
    wall_time_exceeded = False
    if trace_path is not None:
        append_trace_event(
            trace_path,
            "process_started",
            run_id=run_id,
            root_pid=process.pid,
            memory_limit_mib=memory_limit_mib,
            wall_time_limit_seconds=wall_time_limit_seconds,
        )
    try:
        while True:
            host_rss_bytes = process_tree_rss_bytes(process.pid)
            docker_memory_bytes = docker_cgroup_memory_bytes(docker_memory_registry_path)
            rss_bytes = host_rss_bytes + docker_memory_bytes
            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            peak_host_rss_bytes = max(peak_host_rss_bytes, host_rss_bytes)
            peak_docker_memory_bytes = max(peak_docker_memory_bytes, docker_memory_bytes)
            if trace_path is not None:
                append_trace_event(
                    trace_path,
                    "resource_sample",
                    run_id=run_id,
                    root_pid=process.pid,
                    process_tree_rss_mib=host_rss_bytes / MIB,
                    combined_memory_mib=rss_bytes / MIB,
                    host_process_tree_rss_mib=host_rss_bytes / MIB,
                    docker_memory_mib=docker_memory_bytes / MIB,
                )
            return_code = process.poll()
            if (
                return_code is None
                and cancellation_event is not None
                and cancellation_event.is_set()
            ):
                cancelled = True
                if trace_path is not None:
                    append_trace_event(
                        trace_path,
                        "interruption_requested",
                        run_id=run_id,
                        root_pid=process.pid,
                    )
                _terminate_process_group(process, grace_seconds=termination_grace_seconds)
                return_code = process.wait()
            if memory_limit_bytes is not None and rss_bytes > memory_limit_bytes:
                exceeded = True
                if trace_path is not None:
                    append_trace_event(
                        trace_path,
                        "memory_limit_exceeded",
                        run_id=run_id,
                        root_pid=process.pid,
                        process_tree_rss_mib=host_rss_bytes / MIB,
                        combined_memory_mib=rss_bytes / MIB,
                        host_process_tree_rss_mib=host_rss_bytes / MIB,
                        docker_memory_mib=docker_memory_bytes / MIB,
                        memory_limit_mib=memory_limit_mib,
                    )
                _terminate_process_group(process, grace_seconds=termination_grace_seconds)
                return_code = process.wait()
            elapsed_seconds = time.monotonic() - started
            if (
                return_code is None
                and wall_time_limit_seconds is not None
                and elapsed_seconds > wall_time_limit_seconds
            ):
                wall_time_exceeded = True
                if trace_path is not None:
                    append_trace_event(
                        trace_path,
                        "wall_time_limit_exceeded",
                        run_id=run_id,
                        root_pid=process.pid,
                        elapsed_seconds=elapsed_seconds,
                        wall_time_limit_seconds=wall_time_limit_seconds,
                    )
                _terminate_process_group(process, grace_seconds=termination_grace_seconds)
                return_code = process.wait()
            if return_code is not None:
                break
            time.sleep(poll_interval_seconds)
    except BaseException:
        _terminate_process_group(process, grace_seconds=termination_grace_seconds)
        raise
    if trace_path is not None:
        append_trace_event(
            trace_path,
            "process_finished",
            run_id=run_id,
            root_pid=process.pid,
            return_code=return_code,
            peak_process_tree_rss_mib=peak_host_rss_bytes / MIB,
            peak_combined_memory_mib=peak_rss_bytes / MIB,
            peak_host_process_tree_rss_mib=peak_host_rss_bytes / MIB,
            peak_docker_memory_mib=peak_docker_memory_bytes / MIB,
            memory_limit_exceeded=exceeded,
            cancelled=cancelled,
            wall_time_limit_exceeded=wall_time_exceeded,
        )
    return ProcessUsage(
        return_code=return_code,
        peak_combined_memory_mib=peak_rss_bytes / MIB,
        peak_host_rss_mib=peak_host_rss_bytes / MIB,
        peak_docker_memory_mib=peak_docker_memory_bytes / MIB,
        memory_limit_exceeded=exceeded,
        cancelled=cancelled,
        wall_time_limit_exceeded=wall_time_exceeded,
    )


def _terminate_process_group(process: subprocess.Popen[str], *, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def append_trace_event(path: Path, event: str, **fields: Any) -> None:
    """Append one small event atomically to a cross-process JSONL trace."""

    payload = {
        "schema_version": 1,
        "event": event,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "monotonic_seconds": time.monotonic(),
        **fields,
    }
    serialized = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        written = os.write(descriptor, serialized)
        if written != len(serialized):
            raise OSError(f"Short resource-trace write: {written}/{len(serialized)} bytes")
    finally:
        os.close(descriptor)


def rlm_trace_callbacks(
    trace_path: Path,
    *,
    sample_id: str | int,
) -> dict[str, Callable[..., None]]:
    """Build prompt-free callbacks aligned with launcher RSS timestamps."""

    def on_subcall_start(depth: int, model: str, _prompt_preview: str) -> None:
        append_trace_event(
            trace_path,
            "rlm_subcall_started",
            sample_id=sample_id,
            depth=depth,
            model=model,
        )

    def on_subcall_complete(
        depth: int, model: str, duration_seconds: float, error: str | None
    ) -> None:
        append_trace_event(
            trace_path,
            "rlm_subcall_finished",
            sample_id=sample_id,
            depth=depth,
            model=model,
            duration_seconds=duration_seconds,
            failed=error is not None,
        )

    def on_iteration_start(depth: int, iteration: int) -> None:
        append_trace_event(
            trace_path,
            "rlm_iteration_started",
            sample_id=sample_id,
            depth=depth,
            iteration=iteration,
        )

    def on_iteration_complete(depth: int, iteration: int, duration_seconds: float) -> None:
        append_trace_event(
            trace_path,
            "rlm_iteration_finished",
            sample_id=sample_id,
            depth=depth,
            iteration=iteration,
            duration_seconds=duration_seconds,
        )

    def on_iteration_metrics(depth: int, iteration: int, metrics: dict[str, Any]) -> None:
        append_trace_event(
            trace_path,
            "rlm_iteration_metrics",
            sample_id=sample_id,
            depth=depth,
            iteration=iteration,
            calls=int(metrics.get("iteration_calls", 0)),
            input_tokens=int(metrics.get("iteration_input_tokens", 0)),
            output_tokens=int(metrics.get("iteration_output_tokens", 0)),
            cost_usd=float(metrics.get("iteration_cost_usd", 0.0)),
            model_time_seconds=float(metrics.get("model_time_s") or 0.0),
            tool_time_seconds=float(metrics.get("tool_time_s") or 0.0),
            code_block_count=int(metrics.get("code_block_count", 0)),
            had_error=bool(metrics.get("had_error", False)),
        )

    def on_completion_metrics(metrics: dict[str, Any]) -> None:
        append_trace_event(
            trace_path,
            "rlm_completion_metrics",
            sample_id=sample_id,
            calls=int(metrics.get("calls", 0)),
            input_tokens=int(metrics.get("input_tokens", 0)),
            output_tokens=int(metrics.get("output_tokens", 0)),
            cost_usd=float(metrics.get("cost_usd", 0.0)),
            execution_time_seconds=float(metrics.get("execution_time_seconds", 0.0)),
            model_time_seconds=float(metrics.get("model_time_seconds", 0.0)),
            tool_time_seconds=float(metrics.get("tool_time_seconds", 0.0)),
            stopped_by_timeout=bool(metrics.get("stopped_by_timeout", False)),
        )

    return {
        "on_subcall_start": on_subcall_start,
        "on_subcall_complete": on_subcall_complete,
        "on_iteration_start": on_iteration_start,
        "on_iteration_complete": on_iteration_complete,
        "on_iteration_metrics": on_iteration_metrics,
        "on_completion_metrics": on_completion_metrics,
    }
