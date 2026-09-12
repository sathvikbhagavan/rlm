from __future__ import annotations

import hashlib

from rxnhaystack.dataset import DatasetError
from rxnhaystack.metrics import RunMetrics, write_run_metrics
from rxnhaystack.runtime import WorkerConfig


def main() -> None:
    config = WorkerConfig.from_environment()
    dataset_path = config.require_dataset().cleaned
    with dataset_path.open("rb") as handle:
        first_record = handle.readline().rstrip(b"\n")
    if not first_record:
        raise DatasetError(f"Cleaned dataset is empty: {dataset_path}")
    write_run_metrics(
        RunMetrics(
            calls=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_seconds=0,
            tool_time_seconds=0,
            cost_usd=0,
            cost_chf=0,
            results={
                "check": "dataset-readable",
                "first_record_sha256": hashlib.sha256(first_record).hexdigest(),
            },
        )
    )
    print(f"Verified launcher access to {dataset_path}")


if __name__ == "__main__":
    main()
