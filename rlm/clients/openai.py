import asyncio
import os
import re
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import openai
from dotenv import load_dotenv

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary
from rxnhaystack.accounting import RESPONSE_EVENTS_ENV, append_audit_event
from rxnhaystack.rate_limit import call_swissai_async, call_swissai_sync, is_swissai_url

load_dotenv()

# Load API keys from environment variables
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_VERCEL_API_KEY = os.getenv("AI_GATEWAY_API_KEY")
DEFAULT_PRIME_API_KEY = os.getenv("PRIME_API_KEY")
DEFAULT_PRIME_INTELLECT_BASE_URL = "https://api.pinference.ai/api/v1/"
EMPTY_CHOICE_MAX_RETRIES = 2
EMPTY_CHOICE_BACKOFF_SECONDS = (1.0, 2.0)
MAX_RETRY_AFTER_SECONDS = 60.0
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 524, 529})
_REDACTED = "[REDACTED]"
_SENSITIVE_RESPONSE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "headers",
        "input",
        "messages",
        "password",
        "prompt",
        "proxy_authorization",
        "request",
        "request_body",
        "secret",
    }
)
_BEARER_CREDENTIAL = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")


class OpenAIClient(BaseLM):
    """
    LM Client for running models with the OpenAI API. Works with vLLM as well.

    Any additional keyword arguments (e.g. default_headers, default_query, max_retries)
    are passed through to the underlying openai.OpenAI and openai.AsyncOpenAI constructors.
    Only model_name is excluded, since it is not a client constructor argument.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        chat_completion_extra_body: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.chat_completion_extra_body = dict(chat_completion_extra_body or {})
        self.max_output_tokens = max_output_tokens

        if api_key is None:
            if base_url == "https://api.openai.com/v1" or base_url is None:
                api_key = DEFAULT_OPENAI_API_KEY
            elif base_url == "https://openrouter.ai/api/v1":
                api_key = DEFAULT_OPENROUTER_API_KEY
            elif base_url == "https://ai-gateway.vercel.sh/v1":
                api_key = DEFAULT_VERCEL_API_KEY
            elif base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
                api_key = DEFAULT_PRIME_API_KEY

        # Pass through arbitrary kwargs to the OpenAI client (e.g. default_headers, default_query, max_retries).
        # Exclude model_name since it is not an OpenAI client constructor argument.
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            **{k: v for k, v in self.kwargs.items() if k != "model_name"},
        }
        self._is_openrouter = str(base_url or "").rstrip("/") == "https://openrouter.ai/api/v1"
        # Bound OpenRouter retries here so every transport attempt is audited.
        if self._is_openrouter:
            client_kwargs["max_retries"] = 0
        self.client = openai.OpenAI(**client_kwargs)
        self.async_client = openai.AsyncOpenAI(**client_kwargs)
        self.model_name = model_name
        self.base_url = base_url  # Track for cost extraction
        self._api_key = api_key or ""
        self._is_swissai = is_swissai_url(base_url)
        self._accounting_lock = threading.Lock()

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.model_costs: dict[str, float] = defaultdict(float)  # Cost in USD
        self.model_usage_available_calls: dict[str, int] = defaultdict(int)
        self.model_usage_unavailable_calls: dict[str, int] = defaultdict(int)
        self.model_recovered_calls: dict[str, int] = defaultdict(int)
        self.model_generation_ids: dict[str, list[str]] = defaultdict(list)
        self._unavailable_generations: dict[str, tuple[str, int | None, int | None]] = {}
        self._recovered_generations: set[str] = set()
        self._last_usage = ModelUsageSummary(0, 0, 0)

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        extra_body = dict(self.chat_completion_extra_body)
        if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra_body["usage"] = {"include": True}

        request_kwargs = {"model": model, "messages": messages, "extra_body": extra_body}
        if self._is_openrouter:
            request_kwargs["extra_headers"] = {"X-OpenRouter-Metadata": "enabled"}
        if self.max_output_tokens is not None:
            request_kwargs["max_tokens"] = self.max_output_tokens

        def request():
            return self.client.chat.completions.create(**request_kwargs)

        self._begin_request_sequence()
        for transport_attempt in range(1, EMPTY_CHOICE_MAX_RETRIES + 2):
            try:
                response = (
                    call_swissai_sync(request, api_key=self._api_key)
                    if self._is_swissai
                    else request()
                )
            except Exception as error:
                if not self._handle_transport_error(
                    error,
                    model=model,
                    prompt=prompt,
                    transport_attempt=transport_attempt,
                ):
                    raise
                if transport_attempt > EMPTY_CHOICE_MAX_RETRIES:
                    raise
                delay = self._retry_delay(error, transport_attempt)
                self._audit_retry(error, model, transport_attempt, delay)
                time.sleep(delay)
                continue
            self._save_response(
                response,
                model,
                transport_attempt=transport_attempt,
                prompt=prompt,
            )
            has_choices = self._has_choices(response)
            self._track_cost(response, model, scientifically_usable=has_choices)
            if has_choices:
                return self._response_text(response)
            status = self._status_code(response)
            if status is not None and status not in RETRYABLE_HTTP_STATUS_CODES:
                self._raise_non_retryable_response(
                    response,
                    model=model,
                    prompt=prompt,
                    transport_attempt=transport_attempt,
                )
            if transport_attempt > EMPTY_CHOICE_MAX_RETRIES:
                self._audit(
                    "empty_choices_exhausted",
                    model=model,
                    launcher_attempt=self._launcher_attempt(),
                    transport_attempt=transport_attempt,
                    generation_id=self._generation_id(response),
                    request_id=self._request_id(response),
                    accounting_status="unavailable",
                )
                raise ValueError(
                    f"Provider response has no completion choices after {transport_attempt} attempts"
                )
            backoff = self._retry_delay(response, transport_attempt)
            self._audit(
                "empty_choices_retry_scheduled",
                model=model,
                launcher_attempt=self._launcher_attempt(),
                transport_attempt=transport_attempt,
                generation_id=self._generation_id(response),
                request_id=self._request_id(response),
                backoff_seconds=backoff,
            )
            time.sleep(backoff)
        raise AssertionError("unreachable")

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        extra_body = dict(self.chat_completion_extra_body)
        if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra_body["usage"] = {"include": True}

        request_kwargs = {"model": model, "messages": messages, "extra_body": extra_body}
        if self._is_openrouter:
            request_kwargs["extra_headers"] = {"X-OpenRouter-Metadata": "enabled"}
        if self.max_output_tokens is not None:
            request_kwargs["max_tokens"] = self.max_output_tokens

        async def request():
            return await self.async_client.chat.completions.create(**request_kwargs)

        self._begin_request_sequence()
        for transport_attempt in range(1, EMPTY_CHOICE_MAX_RETRIES + 2):
            try:
                response = (
                    await call_swissai_async(request, api_key=self._api_key)
                    if self._is_swissai
                    else await request()
                )
            except Exception as error:
                if not self._handle_transport_error(
                    error,
                    model=model,
                    prompt=prompt,
                    transport_attempt=transport_attempt,
                ):
                    raise
                if transport_attempt > EMPTY_CHOICE_MAX_RETRIES:
                    raise
                delay = self._retry_delay(error, transport_attempt)
                self._audit_retry(error, model, transport_attempt, delay)
                await asyncio.sleep(delay)
                continue
            self._save_response(
                response,
                model,
                transport_attempt=transport_attempt,
                prompt=prompt,
            )
            has_choices = self._has_choices(response)
            self._track_cost(response, model, scientifically_usable=has_choices)
            if has_choices:
                return self._response_text(response)
            status = self._status_code(response)
            if status is not None and status not in RETRYABLE_HTTP_STATUS_CODES:
                self._raise_non_retryable_response(
                    response,
                    model=model,
                    prompt=prompt,
                    transport_attempt=transport_attempt,
                )
            if transport_attempt > EMPTY_CHOICE_MAX_RETRIES:
                self._audit(
                    "empty_choices_exhausted",
                    model=model,
                    launcher_attempt=self._launcher_attempt(),
                    transport_attempt=transport_attempt,
                    generation_id=self._generation_id(response),
                    request_id=self._request_id(response),
                    accounting_status="unavailable",
                )
                raise ValueError(
                    f"Provider response has no completion choices after {transport_attempt} attempts"
                )
            backoff = self._retry_delay(response, transport_attempt)
            self._audit(
                "empty_choices_retry_scheduled",
                model=model,
                launcher_attempt=self._launcher_attempt(),
                transport_attempt=transport_attempt,
                generation_id=self._generation_id(response),
                request_id=self._request_id(response),
                backoff_seconds=backoff,
            )
            await asyncio.sleep(backoff)
        raise AssertionError("unreachable")

    @staticmethod
    def _has_choices(response: openai.ChatCompletion) -> bool:
        return bool(getattr(response, "choices", None))

    @staticmethod
    def _response_text(response: openai.ChatCompletion) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("Provider response has no completion choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Provider response has no non-empty completion text")
        return content

    @staticmethod
    def _generation_id(response: openai.ChatCompletion) -> str | None:
        value = getattr(response, "id", None)
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _launcher_attempt() -> int | None:
        raw = os.environ.get("RXNHAYSTACK_ATTEMPT")
        try:
            return int(raw) if raw is not None else None
        except ValueError:
            return None

    @staticmethod
    def _request_id(response: openai.ChatCompletion) -> str | None:
        for name in ("request_id", "_request_id"):
            value = getattr(response, name, None)
            if isinstance(value, str) and value:
                return value
        extra = getattr(response, "model_extra", None)
        if isinstance(extra, Mapping):
            for name in ("request_id", "request-id", "x-request-id"):
                value = extra.get(name)
                if isinstance(value, str) and value:
                    return value
        return None

    @classmethod
    def _error_details(cls, value: Any) -> Mapping[str, Any]:
        payload = cls._response_payload(value)
        if not isinstance(payload, Mapping):
            return {}
        error = payload.get("error")
        if not isinstance(error, Mapping):
            model_extra = payload.get("model_extra")
            if isinstance(model_extra, Mapping):
                error = model_extra.get("error")
        return error if isinstance(error, Mapping) else {}

    @classmethod
    def _status_code(cls, value: Any) -> int | None:
        direct = getattr(value, "status_code", None)
        if isinstance(direct, int):
            return direct
        code = cls._error_details(value).get("code")
        try:
            return int(code) if code is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _error_message(cls, value: Any) -> str | None:
        message = cls._error_details(value).get("message")
        if isinstance(message, str) and message:
            return message
        direct = getattr(value, "message", None)
        return direct if isinstance(direct, str) and direct else None

    @classmethod
    def _response_headers(cls, value: Any) -> Mapping[str, Any]:
        headers: dict[str, Any] = {}
        for candidate in (
            getattr(value, "headers", None),
            getattr(getattr(value, "response", None), "headers", None),
            cls._error_details(value).get("headers"),
            cls._error_details(value).get("metadata"),
        ):
            if isinstance(candidate, Mapping):
                headers.update({str(key): item for key, item in candidate.items()})
        return headers

    @classmethod
    def _retry_after_seconds(cls, value: Any) -> float | None:
        raw = None
        for key, item in cls._response_headers(value).items():
            if str(key).casefold().replace("_", "-") == "retry-after":
                raw = item
                break
        if raw is None:
            return None
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(raw))
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            seconds = (retry_at - datetime.now(UTC)).total_seconds()
        return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)

    @classmethod
    def _retry_delay(cls, value: Any, transport_attempt: int) -> float:
        retry_after = cls._retry_after_seconds(value)
        if retry_after is not None:
            return retry_after
        return EMPTY_CHOICE_BACKOFF_SECONDS[transport_attempt - 1]

    def _audit_retry(
        self,
        value: Any,
        model: str,
        transport_attempt: int,
        delay: float,
    ) -> None:
        self._audit(
            "provider_retry_scheduled",
            model=model,
            launcher_attempt=self._launcher_attempt(),
            transport_attempt=transport_attempt,
            request_id=self._request_id(value),
            status_code=self._status_code(value),
            backoff_seconds=delay,
            retry_after_seconds=self._retry_after_seconds(value),
        )

    def _save_transport_error(
        self,
        error: Exception,
        model: str,
        *,
        transport_attempt: int,
        prompt: str | list[dict[str, Any]],
    ) -> None:
        secrets = (self._api_key, *self._prompt_values(prompt))
        message = self._error_message(error) or str(error)
        self._audit(
            "provider_error_saved",
            model=model,
            launcher_attempt=self._launcher_attempt(),
            transport_attempt=transport_attempt,
            request_id=self._request_id(error),
            status_code=self._status_code(error),
            error_message=self._sanitize_response(message, secrets=secrets),
            error=self._sanitize_response(
                self._response_payload(error),
                secrets=secrets,
            ),
        )

    def _handle_transport_error(
        self,
        error: Exception,
        *,
        model: str,
        prompt: str | list[dict[str, Any]],
        transport_attempt: int,
    ) -> bool:
        status = self._status_code(error)
        is_connection_error = isinstance(
            error,
            (openai.APIConnectionError, openai.APITimeoutError),
        )
        if status is None and not is_connection_error:
            return False
        self._save_transport_error(
            error,
            model,
            transport_attempt=transport_attempt,
            prompt=prompt,
        )
        self._mark_usage_unavailable(
            model,
            None,
            reason=(
                f"provider transport returned HTTP {status}"
                if status is not None
                else "provider connection failed before returning usage metadata"
            ),
        )
        retryable = status in RETRYABLE_HTTP_STATUS_CODES if status is not None else True
        if not retryable:
            self._audit(
                "provider_error_non_retryable",
                model=model,
                launcher_attempt=self._launcher_attempt(),
                transport_attempt=transport_attempt,
                request_id=self._request_id(error),
                status_code=status,
                accounting_status="unavailable",
            )
        elif transport_attempt > EMPTY_CHOICE_MAX_RETRIES:
            self._audit(
                "provider_retries_exhausted",
                model=model,
                launcher_attempt=self._launcher_attempt(),
                transport_attempt=transport_attempt,
                request_id=self._request_id(error),
                status_code=status,
                accounting_status="unavailable",
            )
        return retryable

    def _raise_non_retryable_response(
        self,
        response: openai.ChatCompletion,
        *,
        model: str,
        prompt: str | list[dict[str, Any]],
        transport_attempt: int,
    ) -> None:
        status = self._status_code(response)
        secrets = (self._api_key, *self._prompt_values(prompt))
        message = self._sanitize_response(self._error_message(response), secrets=secrets)
        self._audit(
            "provider_error_non_retryable",
            model=model,
            launcher_attempt=self._launcher_attempt(),
            transport_attempt=transport_attempt,
            generation_id=self._generation_id(response),
            request_id=self._request_id(response),
            status_code=status,
            error_message=message,
            accounting_status="unavailable",
        )
        suffix = f": {message}" if message else ""
        raise ValueError(f"Provider response failed with non-retryable HTTP {status}{suffix}")

    @staticmethod
    def _prompt_values(prompt: str | list[dict[str, Any]]) -> tuple[str, ...]:
        values: list[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, str):
                if value:
                    values.append(value)
            elif isinstance(value, Mapping):
                for item in value.values():
                    visit(item)
            elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
                for item in value:
                    visit(item)

        visit(prompt)
        return tuple(values)

    @classmethod
    def _response_payload(
        cls,
        response: Any,
        *,
        seen: set[int] | None = None,
        depth: int = 0,
    ) -> Any:
        if response is None or isinstance(response, (bool, int, float, str)):
            return response
        if depth >= 32:
            return "<maximum serialization depth>"
        seen = set() if seen is None else seen
        identity = id(response)
        if identity in seen:
            return "<recursive reference>"
        seen.add(identity)

        def serialize(value: Any) -> Any:
            return cls._response_payload(value, seen=seen, depth=depth + 1)

        if isinstance(response, Mapping):
            return {str(key): serialize(value) for key, value in response.items()}
        if isinstance(response, (list, tuple)):
            return [serialize(value) for value in response]
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            try:
                payload = model_dump(mode="json", warnings=False)
            except TypeError:
                payload = model_dump()
            if isinstance(payload, Mapping):
                serialized = {str(key): serialize(value) for key, value in payload.items()}
                extra = getattr(response, "model_extra", None)
                if extra is not None and "model_extra" not in serialized:
                    serialized["model_extra"] = serialize(extra)
                return serialized
        values = getattr(response, "__dict__", None)
        if isinstance(values, dict):
            return {str(key): serialize(value) for key, value in values.items()}
        return str(response)

    @classmethod
    def _sanitize_response(cls, value: Any, *, secrets: tuple[str, ...]) -> Any:
        if isinstance(value, Mapping):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                normalized = str(key).casefold().replace("-", "_")
                if normalized in _SENSITIVE_RESPONSE_KEYS or normalized.endswith("_api_key"):
                    sanitized[str(key)] = _REDACTED
                else:
                    sanitized[str(key)] = cls._sanitize_response(item, secrets=secrets)
            return sanitized
        if isinstance(value, list):
            return [cls._sanitize_response(item, secrets=secrets) for item in value]
        if isinstance(value, str):
            sanitized_text = _BEARER_CREDENTIAL.sub(_REDACTED, value)
            for secret in secrets:
                if secret:
                    sanitized_text = sanitized_text.replace(secret, _REDACTED)
            return sanitized_text
        return value

    def _audit(self, event: str, **fields: Any) -> None:
        path = os.environ.get(RESPONSE_EVENTS_ENV)
        if path:
            append_audit_event(path, event, **fields)

    def _begin_request_sequence(self) -> None:
        with self._accounting_lock:
            self._last_usage = ModelUsageSummary(0, 0, 0)

    def _accumulate_last_usage(self, current: ModelUsageSummary) -> None:
        previous = self._last_usage
        if previous.total_calls == 0:
            total_cost = current.total_cost
            accounting_status = current.accounting_status
        else:
            total_cost = (
                previous.total_cost + current.total_cost
                if previous.total_cost is not None and current.total_cost is not None
                else None
            )
            accounting_status = (
                "unavailable"
                if "unavailable" in {previous.accounting_status, current.accounting_status}
                else (
                    "recovered"
                    if "recovered" in {previous.accounting_status, current.accounting_status}
                    else "available"
                )
            )
        self._last_usage = ModelUsageSummary(
            total_calls=previous.total_calls + current.total_calls,
            total_input_tokens=previous.total_input_tokens + current.total_input_tokens,
            total_output_tokens=previous.total_output_tokens + current.total_output_tokens,
            total_cost=total_cost,
            accounting_status=accounting_status,
            usage_available_calls=(previous.usage_available_calls + current.usage_available_calls),
            usage_unavailable_calls=(
                previous.usage_unavailable_calls + current.usage_unavailable_calls
            ),
            generation_ids=previous.generation_ids + current.generation_ids,
        )

    def _save_response(
        self,
        response: openai.ChatCompletion,
        model: str,
        *,
        transport_attempt: int,
        prompt: str | list[dict[str, Any]],
    ) -> None:
        """Persist provider output before any accounting or shape validation."""

        if not os.environ.get(RESPONSE_EVENTS_ENV):
            return

        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        usage = getattr(response, "usage", None)
        usage_payload = None
        if usage is not None:
            usage_payload = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "cost_usd": self._extract_cost(usage),
            }
        secrets = (self._api_key, *self._prompt_values(prompt))
        self._audit(
            "provider_response_saved",
            model=model,
            launcher_attempt=self._launcher_attempt(),
            transport_attempt=transport_attempt,
            generation_id=self._generation_id(response),
            request_id=self._request_id(response),
            content=self._sanitize_response(getattr(message, "content", None), secrets=secrets),
            refusal=self._sanitize_response(getattr(message, "refusal", None), secrets=secrets),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage_payload,
            response=self._sanitize_response(
                self._response_payload(response),
                secrets=secrets,
            ),
        )

    def _track_cost(
        self,
        response: openai.ChatCompletion,
        model: str,
        *,
        scientifically_usable: bool,
    ) -> None:
        usage = getattr(response, "usage", None)
        generation_id = self._generation_id(response)
        with self._accounting_lock:
            self.model_call_counts[model] += 1
            if generation_id:
                self.model_generation_ids[model].append(generation_id)
        if usage is None:
            self._mark_usage_unavailable(
                model, generation_id, reason="provider response omitted usage metadata"
            )
            return

        try:
            prompt_tokens = int(usage.prompt_tokens)
            completion_tokens = int(usage.completion_tokens)
            total_tokens = int(usage.total_tokens)
            if min(prompt_tokens, completion_tokens, total_tokens) < 0:
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            self._mark_usage_unavailable(
                model, generation_id, reason="provider response contained incomplete usage metadata"
            )
            return
        reported_cost = self._extract_cost(usage)
        cost_missing = reported_cost is None or not scientifically_usable
        cost = None if cost_missing else reported_cost

        with self._accounting_lock:
            self.model_input_tokens[model] += prompt_tokens
            self.model_output_tokens[model] += completion_tokens
            self.model_total_tokens[model] += total_tokens
            if cost_missing:
                self.model_usage_unavailable_calls[model] += 1
                if generation_id:
                    self._unavailable_generations[generation_id] = (
                        model,
                        prompt_tokens,
                        completion_tokens,
                    )
            else:
                self.model_usage_available_calls[model] += 1
            if cost is not None:
                self.model_costs[model] += cost
            self._accumulate_last_usage(
                ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=prompt_tokens,
                    total_output_tokens=completion_tokens,
                    total_cost=cost,
                    accounting_status="unavailable" if cost_missing else "available",
                    usage_available_calls=int(not cost_missing),
                    usage_unavailable_calls=int(cost_missing),
                    generation_ids=(generation_id,) if generation_id else (),
                )
            )
        self._audit(
            "usage_unavailable" if cost_missing else "usage_available",
            model=model,
            generation_id=generation_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            reason=(
                "provider response had no completion choices"
                if not scientifically_usable
                else ("provider response omitted cost metadata" if cost_missing else None)
            ),
        )

    def _mark_usage_unavailable(
        self, model: str, generation_id: str | None, *, reason: str
    ) -> None:
        with self._accounting_lock:
            self.model_usage_unavailable_calls[model] += 1
            if generation_id:
                self._unavailable_generations[generation_id] = (model, None, None)
            self._accumulate_last_usage(
                ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    total_cost=None,
                    accounting_status="unavailable",
                    usage_unavailable_calls=1,
                    generation_ids=(generation_id,) if generation_id else (),
                )
            )
        self._audit(
            "usage_unavailable",
            model=model,
            generation_id=generation_id,
            reason=reason,
        )

    @staticmethod
    def _extract_cost(usage: Any) -> float | None:
        cost = None
        direct_cost = getattr(usage, "cost", None)
        if direct_cost is not None:
            cost = direct_cost
        else:
            extra = getattr(usage, "model_extra", None)
        if cost is None and isinstance(extra, dict):
            if extra.get("cost") is not None:
                cost = extra["cost"]
            elif extra.get("cost_details", {}).get("upstream_inference_cost") is not None:
                cost = extra["cost_details"]["upstream_inference_cost"]
        try:
            return float(cost) if cost is not None else None
        except (TypeError, ValueError):
            return None

    def recover_usage(
        self,
        *,
        generation_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> bool:
        """Idempotently apply delayed provider metadata to an unavailable call."""

        with self._accounting_lock:
            if generation_id in self._recovered_generations:
                return False
            unavailable = self._unavailable_generations.get(generation_id)
            if unavailable is None:
                raise ValueError(f"No unavailable usage event for generation {generation_id!r}")
            model, known_prompt_tokens, known_completion_tokens = unavailable
            if min(prompt_tokens, completion_tokens, cost_usd) < 0:
                raise ValueError("Recovered usage and cost must be non-negative")
            if known_prompt_tokens is None or known_completion_tokens is None:
                self.model_input_tokens[model] += prompt_tokens
                self.model_output_tokens[model] += completion_tokens
                self.model_total_tokens[model] += prompt_tokens + completion_tokens
            elif (known_prompt_tokens, known_completion_tokens) != (
                prompt_tokens,
                completion_tokens,
            ):
                raise ValueError("Recovered token counts conflict with recorded provider usage")
            self.model_costs[model] += cost_usd
            self.model_usage_available_calls[model] += 1
            self.model_usage_unavailable_calls[model] -= 1
            self.model_recovered_calls[model] += 1
            self._recovered_generations.add(generation_id)
            del self._unavailable_generations[generation_id]
        self._audit(
            "usage_recovered",
            model=model,
            generation_id=generation_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
        )
        return True

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            cost = self.model_costs.get(model)
            unavailable = self.model_usage_unavailable_calls[model]
            recovered = self.model_recovered_calls[model]
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
                total_cost=None if unavailable else float(cost or 0.0),
                accounting_status=(
                    "unavailable" if unavailable else ("recovered" if recovered else "available")
                ),
                usage_available_calls=self.model_usage_available_calls[model],
                usage_unavailable_calls=unavailable,
                generation_ids=tuple(self.model_generation_ids[model]),
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return self._last_usage
