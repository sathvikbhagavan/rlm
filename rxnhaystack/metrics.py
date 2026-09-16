from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rxnhaystack.ledger import validate_metrics
from rxnhaystack.manifest import ManifestError
from rxnhaystack.runtime import atomic_write_json

METRICS_PATH_ENV = "RXNHAYSTACK_METRICS_PATH"
USD_TO_CHF_ENV = "RXNHAYSTACK_USD_TO_CHF"


@dataclass(frozen=True)
class RunMetrics:
    calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    tool_time_seconds: float
    cost_chf: float | None
    cost_usd: float | None = None
    accounting_status: str = "available"
    estimated_cost_chf: float | None = None
    wandb_url: str | None = None
    results: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_metrics(asdict(self), require_complete=True)
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ManifestError("total_tokens must equal input_tokens + output_tokens")


def cost_chf_from_usd(cost_usd: float, *, environ: dict[str, str] | None = None) -> float:
    env = os.environ if environ is None else environ
    raw_rate = env.get(USD_TO_CHF_ENV)
    if raw_rate is None:
        raise ManifestError(f"{USD_TO_CHF_ENV} is required to convert API cost")
    try:
        rate = float(raw_rate)
    except ValueError as error:
        raise ManifestError(f"{USD_TO_CHF_ENV} must be numeric") from error
    if rate <= 0 or cost_usd < 0:
        raise ManifestError("Costs and conversion rates must be non-negative, with a positive rate")
    return cost_usd * rate


def write_run_metrics(
    metrics: RunMetrics,
    *,
    path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> Path:
    env = os.environ if environ is None else environ
    raw_path = path or env.get(METRICS_PATH_ENV)
    if raw_path is None:
        raise ManifestError(f"Metrics path not provided and {METRICS_PATH_ENV} is unset")
    destination = Path(raw_path).expanduser().resolve()
    atomic_write_json(destination, asdict(metrics))
    return destination
