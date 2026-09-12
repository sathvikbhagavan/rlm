from __future__ import annotations

import time

from rlm.codeact_core import (
    IsolatedCodeExecutor,
    _reported_tool_timeout,
    _tool_isolation_failure,
    make_simple_code_executor,
)


def test_campaign_executor_preserves_state_and_preloaded_lines(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_RUN_ID", "test-run")
    monkeypatch.setenv("RXNHAYSTACK_CODEACT_TOOL_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("RXNHAYSTACK_CODEACT_TOOL_MEMORY_LIMIT_MIB", "4096")
    executor = make_simple_code_executor(extra_locals={"lines": ["first", "second"]})
    assert isinstance(executor, IsolatedCodeExecutor)
    try:
        assert executor.execute("answer = len(lines)") == ""
        assert executor.execute("print(answer, lines[1])").strip() == "2 second"
    finally:
        executor.cleanup()


def test_campaign_executor_terminates_timeout_and_restores_clean_state(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_RUN_ID", "test-run")
    monkeypatch.setenv("RXNHAYSTACK_CODEACT_TOOL_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setenv("RXNHAYSTACK_CODEACT_TOOL_MEMORY_LIMIT_MIB", "4096")
    executor = make_simple_code_executor(extra_locals={"lines": ["preserved"]})
    assert isinstance(executor, IsolatedCodeExecutor)
    try:
        assert executor.execute("temporary = 42") == ""
        started = time.monotonic()
        timed_out = executor.execute("while True:\n    pass")
        assert time.monotonic() - started < 3
        assert "TimeoutError" in timed_out
        assert "state was reset" in timed_out
        assert _tool_isolation_failure(timed_out)
        assert _reported_tool_timeout(timed_out) == "0.2"
        recovered = executor.execute("print(lines[0]); print('temporary' in globals())")
        assert recovered.strip().splitlines() == ["preserved", "False"]
    finally:
        executor.cleanup()


def test_campaign_executor_recovers_from_native_process_exit(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_RUN_ID", "test-run")
    monkeypatch.setenv("RXNHAYSTACK_CODEACT_TOOL_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("RXNHAYSTACK_CODEACT_TOOL_MEMORY_LIMIT_MIB", "4096")
    executor = make_simple_code_executor(extra_locals={"lines": ["preserved"]})
    assert isinstance(executor, IsolatedCodeExecutor)
    try:
        crashed = executor.execute("import os; os._exit(7)")
        assert "isolated code process exited unexpectedly" in crashed
        assert _tool_isolation_failure(crashed)
        assert executor.execute("print(lines[0])").strip() == "preserved"
    finally:
        executor.cleanup()
