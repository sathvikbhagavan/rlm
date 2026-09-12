from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rxnhaystack.dataset import (
    CLEANED_DATASET_ENV,
    RAW_DATASET_ENV,
    DatasetPaths,
    resolve_dataset_paths,
    verify_cleaned,
    verify_raw,
)
from rxnhaystack.manifest import ENVIRONMENT_NAME_PATTERN, ExperimentManifest, ManifestError

RESOURCE_TRACE_PATH_ENV = "RXNHAYSTACK_RESOURCE_TRACE_PATH"


@dataclass(frozen=True)
class GitProvenance:
    commit: str
    tracked_dirty: bool


@dataclass(frozen=True)
class DatasetProvenance:
    raw_path: str
    raw_sha256: str
    raw_lines: int
    cleaned_path: str
    cleaned_sha256: str
    cleaned_lines: int


@dataclass(frozen=True)
class Preflight:
    git: GitProvenance
    dataset: DatasetProvenance | None


@dataclass(frozen=True)
class WorkerConfig:
    run_id: str
    run_dir: Path
    metrics_path: Path
    task: str
    condition: str
    method: str
    model: str
    corpus_size: int | str
    positive_cardinality: int | None
    seed: int
    repetition: int
    usd_to_chf: float
    raw_dataset: Path | None
    cleaned_dataset: Path | None
    question_parallelism: int
    resource_trace_path: Path

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> WorkerConfig:
        env = os.environ if environ is None else environ
        corpus_value = require_environment(env, "RXNHAYSTACK_CORPUS_SIZE")
        corpus_size: int | str = int(corpus_value) if corpus_value.isdecimal() else corpus_value
        positive_value = env.get("RXNHAYSTACK_POSITIVE_CARDINALITY")
        positive_cardinality = (
            parse_environment_integer(positive_value, "RXNHAYSTACK_POSITIVE_CARDINALITY", minimum=0)
            if positive_value is not None
            else None
        )
        raw_dataset = optional_environment_path(env, RAW_DATASET_ENV)
        cleaned_dataset = optional_environment_path(env, CLEANED_DATASET_ENV)
        return cls(
            run_id=require_environment(env, "RXNHAYSTACK_RUN_ID"),
            run_dir=Path(require_environment(env, "RXNHAYSTACK_RUN_DIR")).resolve(),
            metrics_path=Path(require_environment(env, "RXNHAYSTACK_METRICS_PATH")).resolve(),
            task=require_environment(env, "RXNHAYSTACK_TASK"),
            condition=require_environment(env, "RXNHAYSTACK_CONDITION"),
            method=require_environment(env, "RXNHAYSTACK_METHOD"),
            model=require_environment(env, "RXNHAYSTACK_MODEL"),
            corpus_size=corpus_size,
            positive_cardinality=positive_cardinality,
            seed=parse_environment_integer(
                require_environment(env, "RXNHAYSTACK_SEED"), "RXNHAYSTACK_SEED"
            ),
            repetition=parse_environment_integer(
                require_environment(env, "RXNHAYSTACK_REPETITION"),
                "RXNHAYSTACK_REPETITION",
                minimum=1,
            ),
            usd_to_chf=parse_environment_float(
                require_environment(env, "RXNHAYSTACK_USD_TO_CHF"),
                "RXNHAYSTACK_USD_TO_CHF",
                minimum_exclusive=0,
            ),
            raw_dataset=raw_dataset,
            cleaned_dataset=cleaned_dataset,
            question_parallelism=parse_environment_integer(
                require_environment(env, "RXNHAYSTACK_QUESTION_PARALLELISM"),
                "RXNHAYSTACK_QUESTION_PARALLELISM",
                minimum=1,
            ),
            resource_trace_path=Path(require_environment(env, RESOURCE_TRACE_PATH_ENV)).resolve(),
        )

    def require_dataset(self) -> DatasetPaths:
        if self.raw_dataset is None or self.cleaned_dataset is None:
            raise ManifestError("Worker requires raw and cleaned dataset paths from preflight")
        return DatasetPaths(
            data_dir=self.cleaned_dataset.parent,
            raw=self.raw_dataset,
            cleaned=self.cleaned_dataset,
        ).require(raw=True, cleaned=True)


def inspect_git(project_root: Path) -> GitProvenance:
    commit = run_git(project_root, "rev-parse", "HEAD").strip()
    status = run_git(project_root, "status", "--porcelain", "--untracked-files=no")
    return GitProvenance(commit=commit, tracked_dirty=bool(status.strip()))


def run_git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ManifestError(
            f"Git preflight failed in {project_root}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


def perform_preflight(
    manifest: ExperimentManifest,
    *,
    data_dir: str | Path | None = None,
    raw_path: str | Path | None = None,
    cleaned_path: str | Path | None = None,
) -> Preflight:
    campaign = manifest.campaign
    if not campaign.project_root.is_dir():
        raise ManifestError(f"Project root does not exist: {campaign.project_root}")
    git = inspect_git(campaign.project_root)
    if campaign.require_clean_git and git.tracked_dirty:
        raise ManifestError(
            "Tracked files have uncommitted changes. Commit them or explicitly set "
            "campaign.require_clean_git = false for a non-archival run."
        )
    for run in manifest.runs:
        require_executable(run.command[0], campaign.project_root)

    dataset = None
    if campaign.require_dataset:
        paths = resolve_dataset_paths(
            data_dir=data_dir,
            raw_path=raw_path,
            cleaned_path=cleaned_path,
        ).require(raw=True, cleaned=True)
        dataset = inspect_dataset(paths)
    return Preflight(git=git, dataset=dataset)


def inspect_dataset(paths: DatasetPaths) -> DatasetProvenance:
    raw = verify_raw(paths.raw)
    cleaned = verify_cleaned(paths.cleaned)
    return DatasetProvenance(
        raw_path=str(raw.path),
        raw_sha256=raw.sha256,
        raw_lines=raw.line_count,
        cleaned_path=str(cleaned.path),
        cleaned_sha256=cleaned.sha256,
        cleaned_lines=cleaned.line_count,
    )


def require_executable(command: str, project_root: Path) -> None:
    if "/" in command:
        path = Path(command)
        candidate = path if path.is_absolute() else project_root / path
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise ManifestError(f"Command is not executable: {candidate}")
    elif shutil.which(command) is None:
        raise ManifestError(f"Command is not available on PATH: {command}")


def load_secret_specs(specifications: list[str]) -> dict[str, str]:
    secrets: dict[str, str] = {}
    for specification in specifications:
        name, separator, raw_path = specification.partition("=")
        if not separator or not ENVIRONMENT_NAME_PATTERN.fullmatch(name) or not raw_path:
            raise ManifestError(
                "Secret files must use an uppercase environment assignment: NAME=/path/to/file"
            )
        if name in secrets:
            raise ManifestError(f"Duplicate secret environment name: {name}")
        path = Path(raw_path).expanduser().resolve()
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
            value = path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ManifestError(f"Cannot read secret file for {name}: {error}") from error
        if mode & 0o077:
            raise ManifestError(
                f"Secret file for {name} is accessible by group or others; require mode 0600"
            )
        if not value:
            raise ManifestError(f"Secret file for {name} is empty")
        secrets[name] = value
    return secrets


def resolve_required_secrets(
    manifest: ExperimentManifest,
    supplied: dict[str, str],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ if environ is None else environ
    resolved = dict(supplied)
    for name in manifest.campaign.required_secrets:
        if name not in resolved and env.get(name):
            resolved[name] = env[name]
    missing = [name for name in manifest.campaign.required_secrets if name not in resolved]
    if missing:
        rendered = ", ".join(missing)
        examples = " ".join(f"--secret-file {name}=~/.{name.lower()}" for name in missing)
        raise ManifestError(
            f"Missing required campaign secret(s): {rendered}. Export each variable or provide "
            f"a private mode-0600 file, for example: {examples}"
        )
    return resolved


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def preflight_as_dict(preflight: Preflight) -> dict[str, Any]:
    return {
        "git": asdict(preflight.git),
        "dataset": asdict(preflight.dataset) if preflight.dataset is not None else None,
    }


def require_environment(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if value is None or not value:
        raise ManifestError(f"Required worker environment variable is missing: {name}")
    return value


def parse_environment_integer(value: str, name: str, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ManifestError(f"Worker environment variable {name} must be an integer") from error
    if minimum is not None and parsed < minimum:
        raise ManifestError(f"Worker environment variable {name} must be >= {minimum}")
    return parsed


def parse_environment_float(value: str, name: str, *, minimum_exclusive: float) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ManifestError(f"Worker environment variable {name} must be numeric") from error
    if parsed <= minimum_exclusive:
        raise ManifestError(f"Worker environment variable {name} must be > {minimum_exclusive}")
    return parsed


def optional_environment_path(environ: Mapping[str, str], name: str) -> Path | None:
    value = environ.get(name)
    return Path(value).resolve() if value else None
