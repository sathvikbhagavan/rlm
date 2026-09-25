#!/usr/bin/env python3
"""Fetch selected W&B output logs into a private local review cache."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse


def parse_wandb_url(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) != 4 or parts[2] != "runs":
        raise ValueError(f"Unexpected W&B run URL: {url}")
    return f"{parts[0]}/{parts[1]}/{parts[3]}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def fetch(args: argparse.Namespace) -> None:
    import wandb

    os.environ["WANDB_API_KEY"] = args.key_file.read_text().strip()
    api = wandb.Api(timeout=args.timeout)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, str]] = []

    for row in read_rows(args.review_sample):
        run_id = row["run_id"]
        url = row.get("wandb_url", "")
        target = args.cache_dir / "runs" / run_id / "attempt-wandb" / "stdout.log"
        if row.get("local_trace"):
            report.append(
                {"run_id": run_id, "status": "local trace already available", "detail": ""}
            )
            continue
        if not url:
            report.append({"run_id": run_id, "status": "no W&B URL", "detail": ""})
            continue
        try:
            run = api.run(parse_wandb_url(url))
            output = next((item for item in run.files() if item.name == "output.log"), None)
            if output is None:
                report.append({"run_id": run_id, "status": "output.log absent", "detail": url})
                continue
            download_dir = target.parent / "download"
            downloaded = Path(output.download(root=download_dir, replace=True).name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), target)
            if download_dir.exists():
                download_dir.rmdir()
            target.chmod(0o600)
            report.append({"run_id": run_id, "status": "downloaded", "detail": url})
        except Exception as error:  # W&B exposes permission and transport failures through one API.
            report.append(
                {
                    "run_id": run_id,
                    "status": "unavailable",
                    "detail": f"{type(error).__name__}: {str(error)[:240]}",
                }
            )

    with args.report.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("run_id", "status", "detail"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(report)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--review-sample", type=Path, required=True)
    result.add_argument("--key-file", type=Path, required=True)
    result.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/home/amin/.cache/rxnhaystack-failure-traces"),
    )
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--timeout", type=int, default=30)
    return result


if __name__ == "__main__":
    fetch(parser().parse_args())
