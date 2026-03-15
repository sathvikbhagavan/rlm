import atexit
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

_TRACING_INITIALIZED = False
_TRACER_PROVIDER: Any = None
_SHUTDOWN_REGISTERED = False


@dataclass(slots=True)
class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        del key, value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        del attributes

    def record_exception(self, exception: BaseException) -> None:
        del exception

    def set_status(self, status: Any) -> None:
        del status


class _NoOpSpanContextManager:
    def __enter__(self) -> _NoOpSpan:
        return _NoOpSpan()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        del exc_type, exc_value, traceback
        return False


class _NoOpTracer:
    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpanContextManager:
        del name, kwargs
        return _NoOpSpanContextManager()


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def is_tracing_enabled(flag_env_var: str = "RLM_ENABLE_TRACING") -> bool:
    return _parse_bool(os.getenv(flag_env_var), default=False)


def _register_shutdown() -> None:
    global _SHUTDOWN_REGISTERED
    if _SHUTDOWN_REGISTERED:
        return
    atexit.register(shutdown_tracing)
    _SHUTDOWN_REGISTERED = True


def init_tracing(
    project_name: str | None = None,
    *,
    auto_instrument: bool = True,
    batch: bool = False,
    endpoint: str | None = None,
    protocol: str | None = None,
    headers: dict[str, str] | None = None,
) -> bool:
    """
    Initialize Phoenix tracing.

    Returns:
        bool: True if tracing initialization succeeded, False otherwise.
    """
    global _TRACING_INITIALIZED, _TRACER_PROVIDER
    if _TRACING_INITIALIZED:
        return True

    try:
        from phoenix.otel import register
    except Exception:
        return False

    resolved_endpoint = endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or "http://localhost:6006"
    resolved_protocol = protocol or os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL") or "http/protobuf"
    if resolved_endpoint.endswith("/"):
        resolved_endpoint = resolved_endpoint[:-1]
    traces_endpoint = (
        resolved_endpoint
        if resolved_endpoint.endswith("/v1/traces")
        else f"{resolved_endpoint}/v1/traces"
    )

    # Keep OTEL exporter env vars consistent to avoid fallback to localhost:4317 (gRPC default).
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = resolved_endpoint
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = resolved_protocol
    os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = traces_endpoint

    register_kwargs: dict[str, Any] = {
        "auto_instrument": auto_instrument,
        "batch": batch,
        # With HTTP/protobuf, force the trace ingestion route to avoid 405 at "/".
        "endpoint": traces_endpoint if resolved_protocol == "http/protobuf" else resolved_endpoint,
        "protocol": resolved_protocol,
    }
    if project_name:
        register_kwargs["project_name"] = project_name
    if headers:
        register_kwargs["headers"] = headers

    _TRACER_PROVIDER = register(**register_kwargs)
    _TRACING_INITIALIZED = True
    _register_shutdown()
    return True


def shutdown_tracing() -> None:
    global _TRACING_INITIALIZED, _TRACER_PROVIDER
    if _TRACER_PROVIDER is None:
        return
    try:
        _TRACER_PROVIDER.shutdown()
    except Exception:
        pass
    finally:
        _TRACING_INITIALIZED = False
        _TRACER_PROVIDER = None


def get_tracer(name: str) -> Any:
    try:
        from opentelemetry import trace as otel_trace
    except Exception:
        return _NoOpTracer()
    return otel_trace.get_tracer(name)


@contextmanager
def using_tracing_attributes(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    prompt_template: str | None = None,
    prompt_template_version: str | None = None,
    prompt_template_variables: dict[str, Any] | None = None,
):
    kwargs: dict[str, Any] = {}
    if session_id:
        kwargs["session_id"] = session_id
    if user_id:
        kwargs["user_id"] = user_id
    if metadata:
        kwargs["metadata"] = metadata
    if tags:
        kwargs["tags"] = tags
    if prompt_template:
        kwargs["prompt_template"] = prompt_template
    if prompt_template_version:
        kwargs["prompt_template_version"] = prompt_template_version
    if prompt_template_variables:
        kwargs["prompt_template_variables"] = prompt_template_variables

    if not kwargs:
        with nullcontext():
            yield
        return

    try:
        from openinference.instrumentation import using_attributes
    except Exception:
        with nullcontext():
            yield
        return

    with using_attributes(**kwargs):
        yield
