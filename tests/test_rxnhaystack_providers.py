from __future__ import annotations

import pytest

from rxnhaystack.manifest import ManifestError
from rxnhaystack.providers import (
    CODEACT_MAX_OUTPUT_TOKENS,
    CODEACT_MAX_OUTPUT_TOKENS_ENV,
    SWISSAI_BASE_URL,
    SWISSAI_REQUEST_TIMEOUT_ENV,
    SWISSAI_REQUEST_TIMEOUT_SECONDS,
    benchmark_provider,
    build_benchmark_llm,
    configure_rlm_for_provider,
    provider_reports_cost,
)


def test_provider_defaults_to_openrouter(monkeypatch) -> None:
    monkeypatch.delenv("RXNHAYSTACK_PROVIDER", raising=False)
    assert benchmark_provider() == "openrouter"
    assert provider_reports_cost()


def test_swissai_records_zero_cost_without_provider_price(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_PROVIDER", "swissai")

    assert not provider_reports_cost()


def test_swissai_rlm_uses_openai_compatible_transport(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_PROVIDER", "swissai")
    monkeypatch.setenv("SWISSAI_RESEARCH_API_KEY", "private")
    monkeypatch.setenv("RXNHAYSTACK_MODEL", "RCP-AIaaS/Qwen/Qwen3.5-397B-A17B")

    configured = configure_rlm_for_provider(
        {"backend": "openrouter", "backend_kwargs": {"model_name": "old"}}
    )

    assert configured["backend"] == "openai"
    assert configured["backend_kwargs"] == {
        "model_name": "RCP-AIaaS/Qwen/Qwen3.5-397B-A17B",
        "api_key": "private",
        "base_url": SWISSAI_BASE_URL,
        "timeout": SWISSAI_REQUEST_TIMEOUT_SECONDS,
        "max_retries": 0,
        "chat_completion_extra_body": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
    }


def test_swissai_llamaindex_client_preserves_chat_interface(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_PROVIDER", "swissai")
    monkeypatch.setenv("SWISSAI_RESEARCH_API_KEY", "private")
    client = build_benchmark_llm(
        model="RCP-AIaaS/deepseek-ai/DeepSeek-V4-Flash-0731",
        api_key="must-be-replaced",
        max_tokens=100,
        reasoning_effort="high",
        additional_kwargs={"max_completion_tokens": 100},
    )
    assert client.api_base == SWISSAI_BASE_URL
    assert client.api_key == "private"
    assert client.metadata.is_chat_model
    assert client.timeout == SWISSAI_REQUEST_TIMEOUT_SECONDS
    assert client.max_retries == 0
    assert client.additional_kwargs == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }


def test_swissai_uses_native_no_thinking_for_high_reasoning_chat(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_PROVIDER", "swissai")
    monkeypatch.setenv("SWISSAI_RESEARCH_API_KEY", "private")

    client = build_benchmark_llm(
        model="model",
        api_key="replaced",
        reasoning_effort="high",
        additional_kwargs={"max_completion_tokens": 100},
    )

    assert client.additional_kwargs == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }


def test_swissai_llamaindex_timeout_is_configurable_and_validated(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_PROVIDER", "swissai")
    monkeypatch.setenv("SWISSAI_RESEARCH_API_KEY", "private")
    monkeypatch.setenv(SWISSAI_REQUEST_TIMEOUT_ENV, "450")

    client = build_benchmark_llm(model="model", api_key="replaced")

    assert client.timeout == 450

    monkeypatch.setenv(SWISSAI_REQUEST_TIMEOUT_ENV, "invalid")
    with pytest.raises(ManifestError, match=SWISSAI_REQUEST_TIMEOUT_ENV):
        build_benchmark_llm(model="model", api_key="replaced")


def test_swissai_requires_its_own_key(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_PROVIDER", "swissai")
    monkeypatch.delenv("SWISSAI_RESEARCH_API_KEY", raising=False)
    with pytest.raises(ManifestError, match="SWISSAI_RESEARCH_API_KEY"):
        configure_rlm_for_provider({"backend_kwargs": {}})


def test_codeact_caps_each_provider_completion(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_PROVIDER", "openrouter")
    monkeypatch.setenv("RXNHAYSTACK_METHOD", "codeact")
    monkeypatch.setenv(CODEACT_MAX_OUTPUT_TOKENS_ENV, str(CODEACT_MAX_OUTPUT_TOKENS))

    client = build_benchmark_llm(
        model="openai/gpt-5-mini",
        api_key="private",
        max_tokens=30_000,
        additional_kwargs={"max_completion_tokens": 30_000},
    )

    assert client.max_tokens == CODEACT_MAX_OUTPUT_TOKENS
    assert client.additional_kwargs["max_completion_tokens"] == CODEACT_MAX_OUTPUT_TOKENS

    monkeypatch.setenv("RXNHAYSTACK_PROVIDER", "swissai")
    monkeypatch.setenv("SWISSAI_RESEARCH_API_KEY", "private")
    swiss_client = build_benchmark_llm(
        model="RCP-AIaaS/Qwen/Qwen3.5-397B-A17B",
        api_key="replaced",
        max_tokens=30_000,
        additional_kwargs={"max_completion_tokens": 30_000},
    )

    assert swiss_client.max_tokens == CODEACT_MAX_OUTPUT_TOKENS


def test_codeact_rejects_invalid_output_guardrail(monkeypatch) -> None:
    monkeypatch.setenv("RXNHAYSTACK_METHOD", "codeact")
    monkeypatch.setenv(CODEACT_MAX_OUTPUT_TOKENS_ENV, "0")

    with pytest.raises(ManifestError, match=CODEACT_MAX_OUTPUT_TOKENS_ENV):
        build_benchmark_llm(model="openai/gpt-5-mini", api_key="private")
