"""Extract a reviewer source archive without datasets, caches, or binary state."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reviewer-id", required=True)
    args = parser.parse_args()
    members: list[dict[str, object]] = []
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as source:
        for info in source.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir() or member.is_absolute() or ".." in member.parts:
                continue
            if member.suffix != ".py" and member.name not in {"results.txt", "check.txt"}:
                continue
            if "__pycache__" in member.parts:
                continue
            relative = Path(*member.parts[1:]) if len(member.parts) > 1 else Path(member.name)
            data = source.read(info)
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            members.append(
                {"path": relative.as_posix(), "bytes": len(data), "sha256": sha256(data)}
            )
    archive_bytes = args.archive.read_bytes()
    manifest = {
        "schema_version": 1,
        "reviewer_id": args.reviewer_id,
        "source_archive": {
            "filename": args.archive.name,
            "bytes": len(archive_bytes),
            "sha256": sha256(archive_bytes),
        },
        "excluded": [
            "reactionSmilesFigShareUSPTO2023_cleaned.txt",
            "*.pkl",
            "*.pyc",
            "__pycache__/",
        ],
        "members": sorted(members, key=lambda item: str(item["path"])),
    }
    (args.output / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output), "files": len(members)}, indent=2))


if __name__ == "__main__":
    main()
