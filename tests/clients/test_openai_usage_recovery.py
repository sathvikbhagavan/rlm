from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rlm.clients.openai import OpenAIClient
from rxnhaystack.accounting import RESPONSE_EVENTS_ENV


def response(
    *,
    generation_id: str = "gen-test",
    content: str | None = "valid answer",
    usage: object | None = None,
    choices: bool = True,
    model_extra: dict | None = None,
    request_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=generation_id,
        request_id=request_id,
        choices=(
            [
                SimpleNamespace(
                    message=SimpleNamespace(content=content, refusal=None),
                    finish_reason="stop",
                )
            ]
            if choices
            else []
        ),
        usage=usage,
        model="openai/gpt-5-mini",
        model_extra=model_extra,
    )


def usage(*, cost: float | None = 0.25) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
        cost=cost,
        model_extra=None,
    )


def client_with_responses(*values: object) -> tuple[OpenAIClient, MagicMock]:
    transport = MagicMock()
    transport.chat.completions.create.side_effect = values
    with (
        patch("rlm.clients.openai.openai.OpenAI", return_value=transport),
        patch("rlm.clients.openai.openai.AsyncOpenAI"),
    ):
        client = OpenAIClient(
            api_key="not-a-real-key",
            model_name="openai/gpt-5-mini",
            base_url="https://openrouter.ai/api/v1",
        )
    return client, transport


def async_client_with_responses(*values: object) -> tuple[OpenAIClient, MagicMock]:
    transport = MagicMock()
    async_transport = MagicMock()
    async_transport.chat.completions.create = AsyncMock(side_effect=values)
    with (
        patch("rlm.clients.openai.openai.OpenAI", return_value=transport),
        patch("rlm.clients.openai.openai.AsyncOpenAI", return_value=async_transport),
    ):
        client = OpenAIClient(
            api_key="not-a-real-key",
            model_name="openai/gpt-5-mini",
            base_url="https://openrouter.ai/api/v1",
        )
    return client, async_transport


def read_events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_normal_usage_is_available_and_response_is_saved_first(monkeypatch, tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    client, _ = client_with_responses(response(usage=usage()))

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
    client, _ = client_with_responses(response(usage=None))

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
    client, _ = client_with_responses(response(usage=None))
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
    client, transport = client_with_responses(response(content=None, usage=usage()))

    with pytest.raises(ValueError, match="non-empty completion text"):
        client.completion("prompt")

    assert read_events(path)[0]["event"] == "provider_response_saved"
    assert transport.chat.completions.create.call_count == 1


def test_empty_choices_retry_then_succeed_sync(monkeypatch, tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    monkeypatch.setenv("RXNHAYSTACK_ATTEMPT", "4")
    client, transport = client_with_responses(
        response(
            generation_id="gen-empty",
            choices=False,
            model_extra={"error": {"message": "upstream returned an empty envelope"}},
            request_id="req-empty",
        ),
        response(generation_id="gen-success", usage=usage()),
    )

    with patch("rlm.clients.openai.time.sleep") as sleep:
        assert client.completion("private prompt") == "valid answer"

    assert transport.chat.completions.create.call_count == 2
    sleep.assert_called_once_with(1.0)
    saved = [event for event in read_events(path) if event["event"] == "provider_response_saved"]
    assert [event["transport_attempt"] for event in saved] == [1, 2]
    assert saved[0]["launcher_attempt"] == 4
    assert saved[0]["request_id"] == "req-empty"
    assert saved[0]["response"]["model_extra"]["error"]["message"].startswith("upstream")
    assert client.get_usage_summary().total_cost is None
    assert client.get_last_usage().total_cost is None
    assert client.get_last_usage().total_calls == 2


def test_empty_choices_retry_then_succeed_async(monkeypatch, tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    client, transport = async_client_with_responses(
        response(generation_id="gen-empty", choices=False),
        response(generation_id="gen-success", usage=usage()),
    )

    with patch("rlm.clients.openai.asyncio.sleep", new=AsyncMock()) as sleep:
        assert asyncio.run(client.acompletion("private prompt")) == "valid answer"

    assert transport.chat.completions.create.await_count == 2
    sleep.assert_awaited_once_with(1.0)
    saved = [event for event in read_events(path) if event["event"] == "provider_response_saved"]
    assert [event["transport_attempt"] for event in saved] == [1, 2]
    assert client.get_usage_summary().total_cost is None
    assert client.get_last_usage().total_cost is None
    assert client.get_last_usage().total_calls == 2


def test_repeated_empty_choices_fail_with_unknown_cost_sync(monkeypatch, tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    client, transport = client_with_responses(
        *(
            response(
                generation_id=f"gen-empty-{attempt}",
                choices=False,
                usage=usage(cost=0.0),
            )
            for attempt in range(3)
        )
    )

    with (
        patch("rlm.clients.openai.time.sleep") as sleep,
        pytest.raises(ValueError, match="after 3 attempts"),
    ):
        client.completion("private prompt")

    assert transport.chat.completions.create.call_count == 3
    assert sleep.call_count == 2
    events = read_events(path)
    assert sum(event["event"] == "provider_response_saved" for event in events) == 3
    assert sum(event["event"] == "empty_choices_retry_scheduled" for event in events) == 2
    assert events[-1]["event"] == "empty_choices_exhausted"
    assert client.get_usage_summary().total_cost is None
    assert client.get_last_usage().total_cost is None


def test_repeated_empty_choices_fail_with_unknown_cost_async(monkeypatch, tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    client, transport = async_client_with_responses(
        *(response(generation_id=f"gen-empty-{attempt}", choices=False) for attempt in range(3))
    )

    with (
        patch("rlm.clients.openai.asyncio.sleep", new=AsyncMock()) as sleep,
        pytest.raises(ValueError, match="after 3 attempts"),
    ):
        asyncio.run(client.acompletion("private prompt"))

    assert transport.chat.completions.create.await_count == 3
    assert sleep.await_count == 2
    events = read_events(path)
    assert sum(event["event"] == "provider_response_saved" for event in events) == 3
    assert events[-1]["accounting_status"] == "unavailable"
    assert client.get_usage_summary().total_cost is None


def test_malformed_content_is_saved_then_rejected_without_async_retry(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    client, transport = async_client_with_responses(response(content=None, usage=usage()))

    with pytest.raises(ValueError, match="non-empty completion text"):
        asyncio.run(client.acompletion("private prompt"))

    assert transport.chat.completions.create.await_count == 1
    assert read_events(path)[0]["event"] == "provider_response_saved"


def test_sync_audit_serialization_redacts_prompts_and_credentials(monkeypatch, tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    client, _ = client_with_responses(
        response(
            usage=usage(),
            model_extra={
                "error": {"code": "upstream_error", "detail": "safe diagnostic"},
                "authorization": "Bearer not-a-real-key",
                "prompt": "private prompt",
                "echo": "private prompt / not-a-real-key / Bearer another-token",
            },
        )
    )

    client.completion("private prompt")

    serialized = path.read_text()
    assert "private prompt" not in serialized
    assert "not-a-real-key" not in serialized
    assert "another-token" not in serialized
    event = read_events(path)[0]
    extra = event["response"]["model_extra"]
    assert extra["error"] == {"code": "upstream_error", "detail": "safe diagnostic"}
    assert extra["authorization"] == "[REDACTED]"
    assert extra["prompt"] == "[REDACTED]"
    assert extra["echo"].count("[REDACTED]") == 3


def test_async_audit_serialization_redacts_prompts_and_credentials(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "responses.jsonl"
    monkeypatch.setenv(RESPONSE_EVENTS_ENV, str(path))
    client, _ = async_client_with_responses(
        response(
            usage=usage(),
            model_extra={
                "error": {"message": "safe async diagnostic"},
                "headers": {"authorization": "Bearer not-a-real-key"},
                "messages": [{"content": "private prompt"}],
            },
        )
    )

    asyncio.run(client.acompletion([{"role": "user", "content": "private prompt"}]))

    serialized = path.read_text()
    assert "private prompt" not in serialized
    assert "not-a-real-key" not in serialized
    extra = read_events(path)[0]["response"]["model_extra"]
    assert extra["error"]["message"] == "safe async diagnostic"
    assert extra["headers"] == "[REDACTED]"
    assert extra["messages"] == "[REDACTED]"
