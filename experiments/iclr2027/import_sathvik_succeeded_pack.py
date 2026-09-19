from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from rxnhaystack.ledger import RunLedger
from rxnhaystack.manifest import load_manifest


ARCHIVE_ROOT = "qwen_gemini_succeeded_pack"
ALLOWED_MODELS = {"qwen3.5-397b", "gemini-3.7-flash"}


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            parts = PurePosixPath(member.name).parts
            if member.name.startswith("/") or ".." in parts or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination, filter="data")
    return destination / ARCHIVE_ROOT


def nested_metrics(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if not key.startswith("metrics.") or value is None:
            continue
        cursor = result
        parts = key.removeprefix("metrics.").split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    results = result.setdefault("results", {})
    if "macro_f1" not in results and "macro_reaction_f1" in results:
        results["macro_f1"] = results["macro_reaction_f1"]
    if row.get("wandb_url") and "wandb_url" not in result:
        result["wandb_url"] = row["wandb_url"]
    return result


def import_pack(*, archive: Path, manifest_path: Path, ledger_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    expected = {run.run_id: run for run in manifest.runs}
    ledger = RunLedger(ledger_path)
    ledger.sync_runs(manifest.runs, manifest_sha256=manifest.sha256)
    with tempfile.TemporaryDirectory(prefix="rxnhaystack-sathvik-pack-") as temporary:
        root = safe_extract(archive, Path(temporary))
        pack_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        rows = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or pack_manifest.get("n_runs") != len(rows):
        raise ValueError("Pack row count does not match its manifest")
    imported = 0
    counts: dict[str, int] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        model = str(row.get("model", ""))
        if model not in ALLOWED_MODELS or row.get("status") != "succeeded":
            raise ValueError(f"Unexpected packed result: {run_id}")
        if run_id not in expected:
            raise ValueError(f"Packed run is absent from full v34: {run_id}")
        record = ledger.get(run_id)
        if record is not None and record.status == "succeeded":
            continue
        attempt = ledger.claim(run_id)
        if attempt is None:
            raise ValueError(f"Cannot import packed run in state {record.status}: {run_id}")
        artifact = ledger_path.parent / "archive-records" / run_id / f"attempt-{attempt:03d}"
        ledger.finish(
            run_id,
            return_code=0,
            artifact_dir=artifact,
            metrics=nested_metrics(row),
        )
        imported += 1
        key = f"{model} {row['method']}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "archive_sha256": archive_sha256(archive),
        "packed_at": pack_manifest.get("packed_at"),
        "pack_rows": len(rows),
        "newly_imported": imported,
        "counts": counts or pack_manifest.get("counts", {}),
        "task15_score_alias": "macro_reaction_f1 is copied to macro_f1 when macro_f1 is absent",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Sathvik's successful v34 cells safely.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = import_pack(
        archive=args.archive.expanduser().resolve(),
        manifest_path=args.manifest,
        ledger_path=args.ledger.expanduser().resolve(),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
