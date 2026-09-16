from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rlm.clients.openai import OpenAIClient
from rxnhaystack.accounting import RESPONSE_EVENTS_ENV


def response(
    *,
    generation_id: str = "gen-test",
    content: str | None = "valid answer",
    usage: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=generation_id,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=None),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


def usage(*, cost: float | None = 0.25) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
        cost=cost,
        model_extra=None,
    )


def client_with_response(value: object) -> OpenAIClient:
    transport = MagicMock()
    transport.chat.completions.create.return_value = value
    with (
        patch("rlm.clients.openai.openai.OpenAI", return_value=transport),
        patch("rlm.clients.openai.openai.AsyncOpenAI"),
    ):
        return OpenAIClient(
            api_key="not-a-real-key",
            model_name="openai/gpt-5-mini",
            base_url="https://openrouter.ai/api/v1",
        )


def read_events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_normal_usage_is_available_and_response_is_saved_first(monkeypatch, tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    client = client_with_response(response(usage=usage()))

    assert client.completion("prompt") == "valid answer"

    summary = client.get_usage_summary()
    assert summary.accounting_status == "available"
    assert summary.total_cost == 0.25
    assert [event["event"] for event in read_events(path)] == [
        "provider_response_saved",
        "usage_available",
    ]


def test_missing_usage_keeps_valid_response_and_marks_cost_unknown(monkeypatch, tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    client = client_with_response(response(usage=None))

    assert client.completion("prompt") == "valid answer"

    summary = client.get_usage_summary()
    assert summary.accounting_status == "unavailable"
    assert summary.total_cost is None
    assert summary.usage_unavailable_calls == 1
    assert summary.generation_ids == ("gen-test",)
    assert [event["event"] for event in read_events(path)] == [
        "provider_response_saved",
        "usage_unavailable",
    ]


def test_delayed_usage_recovery_is_idempotent_and_audited(monkeypatch, tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    client = client_with_response(response(usage=None))
    client.completion("prompt")

    assert client.recover_usage(
        generation_id="gen-test",
        prompt_tokens=20,
        completion_tokens=10,
        cost_usd=0.25,
    )
    assert not client.recover_usage(
        generation_id="gen-test",
        prompt_tokens=20,
        completion_tokens=10,
        cost_usd=0.25,
    )

    summary = client.get_usage_summary()
    assert summary.accounting_status == "recovered"
    assert summary.total_input_tokens == 20
    assert summary.total_output_tokens == 10
    assert summary.total_cost == 0.25
    assert [event["event"] for event in read_events(path)].count("usage_recovered") == 1


def test_malformed_response_is_saved_then_rejected(monkeypatch, tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    client = client_with_response(response(content=None, usage=usage()))

    with pytest.raises(ValueError, match="non-empty completion text"):
        client.completion("prompt")

    assert read_events(path)[0]["event"] == "provider_response_saved"
