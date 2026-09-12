from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from rxnhaystack.metrics import RunMetrics, cost_chf_from_usd, write_run_metrics
from rxnhaystack.providers import benchmark_provider

_FINAL_METRIC = re.compile(
    r"^sample/(?P<sample>[^/]+)/final_total_(?P<kind>input_tokens|output_tokens|tokens|cost_usd)$"
)
_ITERATION_TOTAL = re.compile(r"^sample/[^/]+/iteration_total_tokens$")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _read_trace(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


class CampaignMetricsCapture:
    """Convert the benchmark's existing W&B telemetry into the launcher contract."""

    def __init__(self, wandb: ModuleType) -> None:
        self.wandb = wandb
        self.original_init = wandb.init
        self.original_log = wandb.log
        self.original_finish = wandb.finish
        self.run: Any = None
        self.started: float | None = None
        self.latest: dict[str, Any] = {}
        self.iteration_calls = 0
        self.lock = threading.Lock()
        self.written = False

    def init(self, *args: Any, **kwargs: Any) -> Any:
        config = dict(kwargs.get("config") or {})
        config.update(
            {
                "rxnhaystack_run_id": os.environ["RXNHAYSTACK_RUN_ID"],
                "rxnhaystack_condition": os.environ.get("RXNHAYSTACK_CONDITION"),
                "rxnhaystack_provider": benchmark_provider(),
                "rxnhaystack_repetition": int(os.environ.get("RXNHAYSTACK_REPETITION", "1")),
                "rxnhaystack_question_parallelism": int(
                    os.environ.get("RXNHAYSTACK_QUESTION_PARALLELISM", "1")
                ),
            }
        )
        if "RXNHAYSTACK_CONTEXT_SIZE" in os.environ:
            config["rxnhaystack_context_size"] = int(os.environ["RXNHAYSTACK_CONTEXT_SIZE"])
        if "RXNHAYSTACK_POSITIVE_CARDINALITY" in os.environ:
            config["rxnhaystack_positive_cardinality"] = int(
                os.environ["RXNHAYSTACK_POSITIVE_CARDINALITY"]
            )
        kwargs["config"] = config
        run = self.original_init(*args, **kwargs)
        # W&B replaces its pre-init module functions when a run starts. Capture the
        # live implementations, then restore our observers around them.
        current_log = self.wandb.log
        if getattr(current_log, "__self__", None) is not self:
            self.original_log = current_log
        current_finish = self.wandb.finish
        if getattr(current_finish, "__self__", None) is not self:
            self.original_finish = current_finish
        self.wandb.init = self.init
        self.wandb.log = self.log
        self.wandb.finish = self.finish
        with self.lock:
            self.run = run
            self.started = time.monotonic()
        return run

    def log(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(data, Mapping):
            with self.lock:
                self.latest.update({str(key): value for key, value in data.items()})
                self.iteration_calls += sum(
                    1 for key in data if _ITERATION_TOTAL.fullmatch(str(key))
                )
        return self.original_log(data, *args, **kwargs)

    def _sample_metrics(self) -> dict[str, dict[str, float]]:
        samples: dict[str, dict[str, float]] = {}
        for key, value in self.latest.items():
            match = _FINAL_METRIC.fullmatch(key)
            if match is None:
                continue
            samples.setdefault(match.group("sample"), {})[match.group("kind")] = float(value)
        return samples

    def _summary(self) -> dict[str, Any]:
        summary = getattr(self.run, "summary", None)
        if summary is None:
            return {}
        try:
            items = summary.items()
        except AttributeError:
            return {}
        return {
            str(key): _json_value(value)
            for key, value in items
            if not str(key).startswith(("_", "sample/", "running_"))
            and str(key) not in {"sample_idx", "sample_iteration"}
        }

    def write(self) -> None:
        with self.lock:
            if self.written:
                return
            if self.started is None:
                raise RuntimeError("wandb.finish() was called before wandb.init()")
            samples = self._sample_metrics()
            if not samples:
                raise RuntimeError("No per-sample final token metrics were logged")
            required = {"input_tokens", "output_tokens", "tokens"}
            if benchmark_provider() == "swissai":
                for values in samples.values():
                    values.setdefault("cost_usd", 0.0)
            else:
                required.add("cost_usd")
            incomplete = {
                sample: sorted(required - values.keys())
                for sample, values in samples.items()
                if not required <= values.keys()
            }
            if incomplete:
                raise RuntimeError(f"Incomplete per-sample usage metrics: {incomplete}")

            trace = _read_trace(os.environ.get("RXNHAYSTACK_RESOURCE_TRACE_PATH"))
            rlm_events = [
                event for event in trace if event.get("event") == "rlm_completion_metrics"
            ]
            if rlm_events:
                calls = sum(int(event.get("calls", 0)) for event in rlm_events)
                input_tokens = sum(int(event.get("input_tokens", 0)) for event in rlm_events)
                output_tokens = sum(int(event.get("output_tokens", 0)) for event in rlm_events)
                cost_usd = sum(float(event.get("cost_usd", 0.0)) for event in rlm_events)
                tool_time = sum(float(event.get("tool_time_seconds", 0.0)) for event in rlm_events)
            else:
                calls = self.iteration_calls
                input_tokens = sum(int(values["input_tokens"]) for values in samples.values())
                output_tokens = sum(int(values["output_tokens"]) for values in samples.values())
                cost_usd = sum(values["cost_usd"] for values in samples.values())
                tool_time = sum(
                    float(event.get("duration_seconds", 0.0))
                    for event in trace
                    if event.get("event") == "codeact_tool_finished"
                )

            if calls < 1:
                raise RuntimeError("No successful model calls were captured")
            logged_input = sum(int(values["input_tokens"]) for values in samples.values())
            logged_output = sum(int(values["output_tokens"]) for values in samples.values())
            usage_discrepancy = {
                "input_tokens": input_tokens - logged_input,
                "output_tokens": output_tokens - logged_output,
            }
            results = self._summary()
            if rlm_events and any(usage_discrepancy.values()):
                results["wandb_usage_discrepancy"] = usage_discrepancy

            write_run_metrics(
                RunMetrics(
                    calls=calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    latency_seconds=time.monotonic() - self.started,
                    tool_time_seconds=tool_time,
                    cost_usd=cost_usd,
                    cost_chf=cost_chf_from_usd(cost_usd),
                    wandb_url=getattr(self.run, "url", None),
                    results=results,
                )
            )
            self.written = True

    def finish(self, *args: Any, **kwargs: Any) -> Any:
        metrics_error: BaseException | None = None
        try:
            self.write()
        except BaseException as error:
            metrics_error = error
        try:
            result = self.original_finish(*args, **kwargs)
        finally:
            if metrics_error is not None:
                raise metrics_error
        return result


def install_campaign_metrics(wandb: ModuleType) -> CampaignMetricsCapture | None:
    """Install once for launcher-managed runs; leave standalone scripts unchanged."""

    if "RXNHAYSTACK_RUN_ID" not in os.environ:
        return None
    existing = getattr(wandb, "_rxnhaystack_metrics_capture", None)
    if existing is not None:
        return existing
    capture = CampaignMetricsCapture(wandb)
    wandb.init = capture.init
    wandb.log = capture.log
    wandb.finish = capture.finish
    wandb._rxnhaystack_metrics_capture = capture
    return capture
