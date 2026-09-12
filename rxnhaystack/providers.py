from __future__ import annotations

import os
from typing import Any

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openrouter import OpenRouter

from rxnhaystack.manifest import ManifestError

PROVIDER_ENV = "RXNHAYSTACK_PROVIDER"
METHOD_ENV = "RXNHAYSTACK_METHOD"
CODEACT_MAX_OUTPUT_TOKENS_ENV = "RXNHAYSTACK_CODEACT_OUTPUT_LIMIT"
CODEACT_MAX_OUTPUT_TOKENS = 2048
SWISSAI_API_KEY_ENV = "SWISSAI_RESEARCH_API_KEY"
SWISSAI_BASE_URL = "https://api.swissai.svc.cscs.ch/v1"
SWISSAI_REQUEST_TIMEOUT_ENV = "RXNHAYSTACK_SWISSAI_REQUEST_TIMEOUT_SECONDS"
SWISSAI_REQUEST_TIMEOUT_SECONDS = 300.0


def benchmark_provider(environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    provider = env.get(PROVIDER_ENV, "openrouter").strip().lower()
    if provider not in {"openrouter", "swissai"}:
        raise ManifestError(f"Unsupported {PROVIDER_ENV} value: {provider!r}")
    return provider


def provider_reports_cost(environ: dict[str, str] | None = None) -> bool:
    """Whether missing per-request price data must fail a benchmark job."""

    return benchmark_provider(environ) == "openrouter"


def _bounded_chat_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Apply the recorded per-turn CodeAct output guardrail."""

    configured = dict(kwargs)
    if os.environ.get(METHOD_ENV) != "codeact":
        return configured
    raw_limit = os.environ.get(
        CODEACT_MAX_OUTPUT_TOKENS_ENV, str(CODEACT_MAX_OUTPUT_TOKENS)
    )
    try:
        limit = int(raw_limit)
    except ValueError as error:
        raise ManifestError(
            f"{CODEACT_MAX_OUTPUT_TOKENS_ENV} must be a positive integer"
        ) from error
    if limit <= 0:
        raise ManifestError(
            f"{CODEACT_MAX_OUTPUT_TOKENS_ENV} must be a positive integer"
        )

    requested = int(configured.get("max_tokens", limit))
    configured["max_tokens"] = min(requested, limit)
    additional = dict(configured.get("additional_kwargs", {}) or {})
    if "max_completion_tokens" in additional:
        additional["max_completion_tokens"] = min(
            int(additional["max_completion_tokens"]), limit
        )
    configured["additional_kwargs"] = additional
    return configured


def build_benchmark_llm(**kwargs: Any) -> OpenRouter | OpenAILike:
    """Build a LlamaIndex chat client without changing benchmark prompt semantics."""

    kwargs = _bounded_chat_kwargs(kwargs)
    provider = benchmark_provider()
    if provider == "openrouter":
        return OpenRouter(**kwargs)

    api_key = os.environ.get(SWISSAI_API_KEY_ENV)
    if not api_key:
        raise ManifestError(f"{SWISSAI_API_KEY_ENV} is required for SwissAI models")
    configured = dict(kwargs)
    configured["api_key"] = api_key
    configured["api_base"] = SWISSAI_BASE_URL
    configured["is_chat_model"] = True
    configured["context_window"] = int(os.environ.get("RXNHAYSTACK_MODEL_CONTEXT_WINDOW", "262144"))
    try:
        request_timeout = float(
            os.environ.get(
                SWISSAI_REQUEST_TIMEOUT_ENV,
                str(SWISSAI_REQUEST_TIMEOUT_SECONDS),
            )
        )
    except ValueError as error:
        raise ManifestError(
            f"{SWISSAI_REQUEST_TIMEOUT_ENV} must be a positive number of seconds"
        ) from error
    if request_timeout <= 0:
        raise ManifestError(
            f"{SWISSAI_REQUEST_TIMEOUT_ENV} must be a positive number of seconds"
        )
    # The endpoint's large models can legitimately take longer than the
    # OpenAILike default of 60 seconds. Keep one explicit request deadline and
    # avoid nested SDK retries multiplying it invisibly; CodeAct and the job
    # ledger already provide auditable retry boundaries.
    configured["timeout"] = request_timeout
    configured["max_retries"] = 0
    # This OpenAI-compatible endpoint does not advertise OpenRouter's normalized
    # reasoning_effort or max_completion_tokens extensions.
    configured.pop("reasoning_effort", None)
    additional = dict(configured.pop("additional_kwargs", {}) or {})
    additional.pop("max_completion_tokens", None)
    # SwissAI's vLLM endpoint returns Qwen's chain of thought separately as
    # reasoning_content, which LlamaIndex does not expose to the agent. In live
    # trials, thinking consumed 30k-token allowances and timed out before any
    # final content. Disable that hidden channel for every SwissAI chat call;
    # CodeAct still performs its visible multi-turn reasoning and tool loop.
    extra_body = dict(additional.get("extra_body", {}) or {})
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})
    chat_template_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    additional["extra_body"] = extra_body
    configured["additional_kwargs"] = additional
    return OpenAILike(**configured)


def configure_rlm_for_provider(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Configure RLM's native client for the selected campaign transport."""

    configured = dict(kwargs)
    backend_kwargs = dict(configured.get("backend_kwargs", {}))
    model = os.environ.get("RXNHAYSTACK_MODEL")
    if model:
        backend_kwargs["model_name"] = model
    if benchmark_provider() == "swissai":
        api_key = os.environ.get(SWISSAI_API_KEY_ENV)
        if not api_key:
            raise ManifestError(f"{SWISSAI_API_KEY_ENV} is required for SwissAI models")
        configured["backend"] = "openai"
        backend_kwargs.update(
            {
                "api_key": api_key,
                "base_url": SWISSAI_BASE_URL,
                "timeout": SWISSAI_REQUEST_TIMEOUT_SECONDS,
                "max_retries": 0,
                "chat_completion_extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            }
        )
    configured["backend_kwargs"] = backend_kwargs
    return configured
