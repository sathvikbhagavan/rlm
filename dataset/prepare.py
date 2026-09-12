from __future__ import annotations

import argparse
import os
from pathlib import Path

from rdkit import __version__ as rdkit_version

from dataset.clean import clean_dataset
from rxnhaystack.dataset import (
    CLEANED_LINE_COUNT,
    CLEANED_SHA256,
    DATASET_DOI,
    RDKIT_VERSION,
    DatasetError,
    download_raw,
    resolve_dataset_paths,
    verify_cleaned,
    verify_raw,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, clean, and verify the exact RxnHaystack USPTO-2023 corpus."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--cleaned-path", type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing raw and cleaned files without downloading or cleaning.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing files after constructing and verifying replacements.",
    )
    return parser.parse_args()


def require_rdkit_version() -> None:
    if rdkit_version != RDKIT_VERSION:
        raise DatasetError(f"RDKit {RDKIT_VERSION} is required; found {rdkit_version}")


def prepare_cleaned(raw_path: Path, cleaned_path: Path, *, force: bool) -> None:
    if cleaned_path.exists() and not force:
        verify_cleaned(cleaned_path)
        return

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    partial = cleaned_path.with_name(f".{cleaned_path.name}.part")
    partial.unlink(missing_ok=True)
    try:
        stats = clean_dataset(raw_path, partial, quiet_rdkit=True)
        if stats.written != CLEANED_LINE_COUNT:
            raise DatasetError(
                f"Cleaning wrote {stats.written} reactions; expected {CLEANED_LINE_COUNT}"
            )
        verify_cleaned(partial)
        os.replace(partial, cleaned_path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    require_rdkit_version()
    paths = resolve_dataset_paths(
        data_dir=args.data_dir,
        raw_path=args.raw_path,
        cleaned_path=args.cleaned_path,
    )
    if args.verify_only:
        raw = verify_raw(paths.raw)
        cleaned = verify_cleaned(paths.cleaned)
    else:
        raw = download_raw(paths.raw, force=args.force)
        prepare_cleaned(paths.raw, paths.cleaned, force=args.force)
        cleaned = verify_cleaned(paths.cleaned)

    print(f"Dataset DOI: {DATASET_DOI}")
    print(f"RDKit: {rdkit_version}")
    print(f"Raw: {raw.path} ({raw.line_count} lines, sha256={raw.sha256})")
    print(f"Cleaned: {cleaned.path} ({cleaned.line_count} lines, sha256={CLEANED_SHA256})")


if __name__ == "__main__":
    main()
