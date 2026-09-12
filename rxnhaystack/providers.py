from __future__ import annotations

import os
from typing import Any

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openrouter import OpenRouter

from rxnhaystack.manifest import ManifestError

PROVIDER_ENV = "RXNHAYSTACK_PROVIDER"
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


def build_benchmark_llm(**kwargs: Any) -> OpenRouter | OpenAILike:
    """Build a LlamaIndex chat client without changing benchmark prompt semantics."""

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
    reasoning_effort = configured.pop("reasoning_effort", None)
    additional = dict(configured.pop("additional_kwargs", {}) or {})
    additional.pop("max_completion_tokens", None)
    if reasoning_effort in {"none", "minimal", "low"}:
        # SwissAI's vLLM endpoint returns Qwen's chain of thought separately as
        # reasoning_content. Without this native template option, even a trivial
        # low-reasoning request can exhaust its output allowance before emitting
        # final content. Translate the intent without changing the user prompt.
        extra_body = dict(additional.get("extra_body", {}) or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})
        chat_template_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        additional["extra_body"] = extra_body
    if additional:
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
        backend_kwargs.update({"api_key": api_key, "base_url": SWISSAI_BASE_URL})
    configured["backend_kwargs"] = backend_kwargs
    return configured
