from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

RAW_FILENAME = "reactionSmilesFigShareUSPTO2023.txt"
CLEANED_FILENAME = "reactionSmilesFigShareUSPTO2023_cleaned.txt"
RAW_URL = "https://ndownloader.figshare.com/files/43858050"
RAW_MD5 = "8ba755614c81f43f9a7f3a841a8ce4b3"
RAW_SHA256 = "8ba53c8aa5a513bdef651fb06f3ed1392ac3056c5f472087bdc52e820440b791"
CLEANED_SHA256 = "9f9b2e71676e3e8f132b495b3fc62e2fdec01bc5b43f7be3dc09ef279c351b14"
RAW_LINE_COUNT = 137_261
CLEANED_LINE_COUNT = 122_456
RDKIT_VERSION = "2022.09.5"
DATASET_DOI = "10.6084/m9.figshare.24921555.v1"

DATA_DIR_ENV = "RXNHAYSTACK_DATA_DIR"
RAW_DATASET_ENV = "RXNHAYSTACK_RAW_DATASET"
CLEANED_DATASET_ENV = "RXNHAYSTACK_CLEANED_DATASET"


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be resolved or fails provenance checks."""


@dataclass(frozen=True)
class FileProvenance:
    path: Path
    sha256: str
    line_count: int


@dataclass(frozen=True)
class DatasetPaths:
    data_dir: Path
    raw: Path
    cleaned: Path

    def require(self, *, raw: bool = False, cleaned: bool = True) -> DatasetPaths:
        required = []
        if raw:
            required.append(self.raw)
        if cleaned:
            required.append(self.cleaned)
        missing = [path for path in required if not path.is_file()]
        if missing:
            rendered = ", ".join(str(path) for path in missing)
            raise DatasetError(
                f"Required RxnHaystack dataset file(s) not found: {rendered}. "
                f"Set {DATA_DIR_ENV}, pass an explicit path, or run dataset/prepare.py."
            )
        return self


def default_data_dir() -> Path:
    return Path.home() / "datasets" / "rxnhaystack"


def resolve_dataset_paths(
    *,
    data_dir: str | Path | None = None,
    raw_path: str | Path | None = None,
    cleaned_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> DatasetPaths:
    env = os.environ if environ is None else environ
    resolved_data_dir = Path(data_dir or env.get(DATA_DIR_ENV) or default_data_dir()).expanduser()
    raw = Path(raw_path or env.get(RAW_DATASET_ENV) or resolved_data_dir / RAW_FILENAME)
    cleaned = Path(
        cleaned_path or env.get(CLEANED_DATASET_ENV) or resolved_data_dir / CLEANED_FILENAME
    )
    return DatasetPaths(
        data_dir=resolved_data_dir.resolve(),
        raw=raw.expanduser().resolve(),
        cleaned=cleaned.expanduser().resolve(),
    )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path, *, chunk_size: int = 1024 * 1024) -> int:
    line_count = 0
    last_byte = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        line_count += 1
    return line_count


def inspect_file(path: Path) -> FileProvenance:
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    return FileProvenance(path=path, sha256=sha256_file(path), line_count=count_lines(path))


def verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_line_count: int,
) -> FileProvenance:
    observed = inspect_file(path)
    failures = []
    if observed.sha256 != expected_sha256:
        failures.append(f"SHA-256 {observed.sha256} != {expected_sha256}")
    if observed.line_count != expected_line_count:
        failures.append(f"lines {observed.line_count} != {expected_line_count}")
    if failures:
        raise DatasetError(f"Dataset verification failed for {path}: {'; '.join(failures)}")
    return observed


def verify_raw(path: Path) -> FileProvenance:
    return verify_file(path, expected_sha256=RAW_SHA256, expected_line_count=RAW_LINE_COUNT)


def verify_cleaned(path: Path) -> FileProvenance:
    return verify_file(
        path,
        expected_sha256=CLEANED_SHA256,
        expected_line_count=CLEANED_LINE_COUNT,
    )


def download_raw(
    destination: Path,
    *,
    url: str = RAW_URL,
    force: bool = False,
) -> FileProvenance:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return verify_raw(destination)

    partial = destination.with_name(f".{destination.name}.part")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            copy_stream(response, output)
        provenance = verify_raw(partial)
        os.replace(partial, destination)
        return FileProvenance(
            path=destination,
            sha256=provenance.sha256,
            line_count=provenance.line_count,
        )
    finally:
        partial.unlink(missing_ok=True)


def copy_stream(source: BinaryIO, destination: BinaryIO, *, chunk_size: int = 1024 * 1024) -> None:
    while chunk := source.read(chunk_size):
        destination.write(chunk)
