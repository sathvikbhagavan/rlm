from __future__ import annotations

from types import SimpleNamespace

from rlm.codeact_helpers import extract_usage_metrics


def test_extract_usage_metrics_includes_openrouter_cache_accounting() -> None:
    usage = SimpleNamespace(
        prompt_tokens=12_000,
        completion_tokens=50,
        total_tokens=12_050,
        model_extra={
            "cost": 0.012,
            "cache_discount": 0.018,
            "prompt_tokens_details": {
                "cached_tokens": 10_000,
                "cache_write_tokens": 1_000,
            },
        },
    )
    response = SimpleNamespace(raw=SimpleNamespace(usage=usage))

    assert extract_usage_metrics(response) == {
        "prompt_tokens": 12_000,
        "completion_tokens": 50,
        "total_tokens": 12_050,
        "cost_usd": 0.012,
        "cached_tokens": 10_000,
        "cache_write_tokens": 1_000,
        "cache_discount_usd": 0.018,
    }


def test_extract_usage_metrics_preserves_uncached_usage_shape() -> None:
    response = SimpleNamespace(
        raw={
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 3,
                "total_tokens": 23,
                "cost": 0.001,
            }
        }
    )

    assert extract_usage_metrics(response) == {
        "prompt_tokens": 20,
        "completion_tokens": 3,
        "total_tokens": 23,
        "cost_usd": 0.001,
    }
