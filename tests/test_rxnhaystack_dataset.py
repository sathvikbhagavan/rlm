from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rxnhaystack.dataset import (
    CLEANED_DATASET_ENV,
    RAW_DATASET_ENV,
    DatasetError,
    count_lines,
    download_raw,
    resolve_dataset_paths,
    sha256_file,
    verify_file,
)


def test_resolve_dataset_paths_uses_explicit_paths_before_environment(tmp_path: Path) -> None:
    env_dir = tmp_path / "environment"
    explicit_dir = tmp_path / "explicit"
    paths = resolve_dataset_paths(
        data_dir=explicit_dir,
        raw_path=explicit_dir / "raw.txt",
        cleaned_path=explicit_dir / "cleaned.txt",
        environ={
            "RXNHAYSTACK_DATA_DIR": str(env_dir),
            RAW_DATASET_ENV: str(env_dir / "raw.txt"),
            CLEANED_DATASET_ENV: str(env_dir / "cleaned.txt"),
        },
    )

    assert paths.data_dir == explicit_dir.resolve()
    assert paths.raw == (explicit_dir / "raw.txt").resolve()
    assert paths.cleaned == (explicit_dir / "cleaned.txt").resolve()


def test_resolve_dataset_paths_supports_file_specific_environment(tmp_path: Path) -> None:
    raw = tmp_path / "inputs" / "raw.txt"
    cleaned = tmp_path / "outputs" / "cleaned.txt"
    paths = resolve_dataset_paths(
        environ={RAW_DATASET_ENV: str(raw), CLEANED_DATASET_ENV: str(cleaned)}
    )

    assert paths.raw == raw.resolve()
    assert paths.cleaned == cleaned.resolve()


def test_require_reports_all_missing_files(tmp_path: Path) -> None:
    paths = resolve_dataset_paths(data_dir=tmp_path, environ={})

    with pytest.raises(DatasetError, match="Required RxnHaystack dataset") as error:
        paths.require(raw=True, cleaned=True)

    assert str(paths.raw) in str(error.value)
    assert str(paths.cleaned) in str(error.value)


@pytest.mark.parametrize(
    ("payload", "expected_lines"),
    [(b"", 0), (b"one", 1), (b"one\n", 1), (b"one\ntwo", 2), (b"one\ntwo\n", 2)],
)
def test_file_inspection_handles_final_newline(
    tmp_path: Path, payload: bytes, expected_lines: int
) -> None:
    path = tmp_path / "data.txt"
    path.write_bytes(payload)

    assert count_lines(path, chunk_size=3) == expected_lines
    assert sha256_file(path, chunk_size=3) == hashlib.sha256(payload).hexdigest()


def test_verify_file_reports_checksum_and_line_failures(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("one\ntwo\n")

    with pytest.raises(DatasetError, match="SHA-256.*lines 2 != 3"):
        verify_file(path, expected_sha256="0" * 64, expected_line_count=3)


def test_download_raw_is_atomic_and_removes_invalid_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "raw.txt"
    source = tmp_path / "source.txt"
    source.write_text("not the expected corpus\n")
    monkeypatch.setattr("rxnhaystack.dataset.RAW_SHA256", "0" * 64)
    monkeypatch.setattr("rxnhaystack.dataset.RAW_LINE_COUNT", 1)

    with pytest.raises(DatasetError, match="verification failed"):
        download_raw(destination, url=source.as_uri())

    assert not destination.exists()
    assert not (tmp_path / ".raw.txt.part").exists()


def test_download_raw_promotes_verified_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "raw.txt"
    source = tmp_path / "source.txt"
    payload = b"reaction-1\nreaction-2\n"
    source.write_bytes(payload)
    monkeypatch.setattr("rxnhaystack.dataset.RAW_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr("rxnhaystack.dataset.RAW_LINE_COUNT", 2)

    provenance = download_raw(destination, url=source.as_uri())

    assert destination.read_bytes() == payload
    assert provenance.path == destination.resolve()
    assert provenance.line_count == 2
