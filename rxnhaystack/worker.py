from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rxnhaystack.providers import configure_rlm_for_provider
from rxnhaystack.resources import rlm_trace_callbacks
from rxnhaystack.runtime import WorkerConfig


@dataclass(frozen=True)
class BenchmarkRuntime:
    """Campaign values with explicit standalone-script defaults."""

    model: str
    dataset_path: Path
    seed: int
    question_parallelism: int
    worker_config: WorkerConfig | None

    @classmethod
    def from_defaults(
        cls,
        *,
        model: str,
        dataset_path: str | Path,
        seed: int,
    ) -> BenchmarkRuntime:
        if "RXNHAYSTACK_RUN_ID" not in os.environ:
            return cls(
                model=model,
                dataset_path=Path(dataset_path).expanduser().resolve(),
                seed=seed,
                question_parallelism=1,
                worker_config=None,
            )
        config = WorkerConfig.from_environment()
        return cls(
            model=config.model,
            dataset_path=config.require_dataset().cleaned,
            seed=config.seed,
            question_parallelism=config.question_parallelism,
            worker_config=config,
        )

    @property
    def launched(self) -> bool:
        return self.worker_config is not None

    def instrument_rlm_kwargs(
        self,
        kwargs: dict[str, Any],
        *,
        sample_id: str | int,
    ) -> dict[str, Any]:
        configured = dict(kwargs)
        backend_kwargs = dict(configured.get("backend_kwargs", {}))
        backend_kwargs["model_name"] = self.model
        configured["backend_kwargs"] = backend_kwargs
        if self.worker_config is not None:
            configured.update(
                rlm_trace_callbacks(
                    self.worker_config.resource_trace_path,
                    sample_id=sample_id,
                )
            )
        return configured

    def trace(self, event: str, *, sample_id: str | int, **fields: Any) -> None:
        if self.worker_config is None:
            return
        from rxnhaystack.resources import append_trace_event

        append_trace_event(
            self.worker_config.resource_trace_path,
            event,
            sample_id=sample_id,
            **fields,
        )

    def timed_sample(self, method: str, sample_id: str | int) -> SampleTimer:
        return SampleTimer(runtime=self, method=method, sample_id=sample_id)

    def codeact_callbacks(self, *, sample_id: str | int) -> dict[str, Any]:
        return {
            "on_llm_start": lambda iteration: self.trace(
                "codeact_llm_started",
                sample_id=sample_id,
                iteration=iteration,
            ),
            "on_llm_complete": lambda iteration, duration, failed: self.trace(
                "codeact_llm_finished",
                sample_id=sample_id,
                iteration=iteration,
                duration_seconds=duration,
                failed=failed,
            ),
            "on_tool_start": lambda iteration: self.trace(
                "codeact_tool_started",
                sample_id=sample_id,
                iteration=iteration,
            ),
            "on_tool_complete": lambda iteration, duration, failed: self.trace(
                "codeact_tool_finished",
                sample_id=sample_id,
                iteration=iteration,
                duration_seconds=duration,
                failed=failed,
            ),
        }


@dataclass
class SampleTimer:
    runtime: BenchmarkRuntime
    method: str
    sample_id: str | int
    started: float | None = None
    duration_seconds: float | None = None

    def __enter__(self) -> SampleTimer:
        self.started = time.monotonic()
        self.runtime.trace(
            "question_started",
            sample_id=self.sample_id,
            method=self.method,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self.started is not None
        self.duration_seconds = time.monotonic() - self.started
        self.runtime.trace(
            "question_finished",
            sample_id=self.sample_id,
            method=self.method,
            duration_seconds=self.duration_seconds,
            failed=exc_value is not None,
        )


def instrument_rlm_from_environment(
    kwargs: dict[str, Any], *, sample_id: str | int = "shared"
) -> dict[str, Any]:
    """Add campaign model and resource callbacks while preserving standalone use."""

    configured = configure_rlm_for_provider(kwargs)
    trace_path = os.environ.get("RXNHAYSTACK_RESOURCE_TRACE_PATH")
    if trace_path:
        configured.update(rlm_trace_callbacks(Path(trace_path).resolve(), sample_id=sample_id))
    return configured


def codeact_callbacks_from_environment(*, sample_id: str | int) -> dict[str, Any]:
    """Attach prompt-free CodeAct timing callbacks for launcher-managed runs."""

    trace_path = os.environ.get("RXNHAYSTACK_RESOURCE_TRACE_PATH")
    if not trace_path:
        return {}

    def trace(event: str, **fields: Any) -> None:
        from rxnhaystack.resources import append_trace_event

        append_trace_event(Path(trace_path).resolve(), event, sample_id=sample_id, **fields)

    return {
        "on_llm_start": lambda iteration: trace("codeact_llm_started", iteration=iteration),
        "on_llm_complete": lambda iteration, duration, failed: trace(
            "codeact_llm_finished",
            iteration=iteration,
            duration_seconds=duration,
            failed=failed,
        ),
        "on_tool_start": lambda iteration: trace("codeact_tool_started", iteration=iteration),
        "on_tool_complete": lambda iteration, duration, failed: trace(
            "codeact_tool_finished",
            iteration=iteration,
            duration_seconds=duration,
            failed=failed,
        ),
    }
