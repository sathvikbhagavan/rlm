from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SECRET_MARKERS = ("API_KEY", "PASSWORD", "SECRET", "TOKEN", "CREDENTIAL")
ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class ManifestError(ValueError):
    """Raised when an experiment manifest is incomplete or inconsistent."""


@dataclass(frozen=True)
class Campaign:
    name: str
    project_root: Path
    artifact_dir: Path
    budget_chf: float
    usd_to_chf: float
    require_dataset: bool
    require_clean_git: bool
    require_metrics: bool
    max_parallel_memory_mib: int | None = None
    required_secrets: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    base_id: str
    campaign: str
    task: str
    condition: str
    method: str
    model: str
    corpus_size: int | str
    positive_cardinality: int | None
    seed: int
    repetition: int
    command: tuple[str, ...]
    env: dict[str, str]
    estimated_cost_chf: float
    memory_reservation_mib: int | None = None
    memory_limit_mib: int | None = None
    question_parallelism: int = 1
    source_run_id: str | None = None

    @property
    def spec_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(self.spec_json.encode()).hexdigest()


@dataclass(frozen=True)
class ExperimentManifest:
    path: Path
    sha256: str
    campaign: Campaign
    runs: tuple[PlannedRun, ...]

    @property
    def estimated_cost_chf(self) -> float:
        return sum(run.estimated_cost_chf for run in self.runs)


def load_manifest(path: str | Path) -> ExperimentManifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = manifest_path.read_bytes()
    except OSError as error:
        raise ManifestError(f"Cannot read experiment manifest {manifest_path}: {error}") from error
    try:
        data = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ManifestError(f"Invalid TOML in {manifest_path}: {error}") from error

    if data.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"schema_version must be {SCHEMA_VERSION}; found {data.get('schema_version')!r}"
        )
    campaign = parse_campaign(data.get("campaign"), manifest_path.parent)
    raw_runs = data.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ManifestError("The manifest must contain at least one [[runs]] entry")

    runs: list[PlannedRun] = []
    base_ids: set[str] = set()
    for index, raw_run in enumerate(raw_runs, start=1):
        if not isinstance(raw_run, dict):
            raise ManifestError(f"runs[{index}] must be a table")
        enabled = raw_run.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ManifestError(f"runs[{index}].enabled must be a boolean")
        if not enabled:
            continue
        base_id = require_string(raw_run, "id", f"runs[{index}]")
        if not RUN_ID_PATTERN.fullmatch(base_id):
            raise ManifestError(
                f"runs[{index}].id must match {RUN_ID_PATTERN.pattern!r}; found {base_id!r}"
            )
        if base_id in base_ids:
            raise ManifestError(f"Duplicate run id: {base_id}")
        base_ids.add(base_id)
        runs.extend(parse_run(raw_run, index=index, campaign=campaign, base_id=base_id))

    if not runs:
        raise ManifestError("The manifest has no enabled runs")
    expanded_ids = [run.run_id for run in runs]
    if len(expanded_ids) != len(set(expanded_ids)):
        raise ManifestError("Expanded run IDs are not unique")
    estimated_cost = sum(run.estimated_cost_chf for run in runs)
    if estimated_cost > campaign.budget_chf:
        raise ManifestError(
            f"Estimated campaign cost CHF {estimated_cost:.2f} exceeds budget "
            f"CHF {campaign.budget_chf:.2f}"
        )
    if campaign.max_parallel_memory_mib is not None:
        for run in runs:
            if run.memory_reservation_mib is None:
                raise ManifestError(
                    f"Run {run.base_id!r} needs memory_reservation_mib because "
                    "campaign.max_parallel_memory_mib is set"
                )
            if run.memory_reservation_mib > campaign.max_parallel_memory_mib:
                raise ManifestError(
                    f"Run {run.base_id!r} memory reservation {run.memory_reservation_mib} MiB "
                    f"exceeds campaign limit {campaign.max_parallel_memory_mib} MiB"
                )
    return ExperimentManifest(
        path=manifest_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        campaign=campaign,
        runs=tuple(runs),
    )


def parse_campaign(raw: Any, manifest_dir: Path) -> Campaign:
    if not isinstance(raw, dict):
        raise ManifestError("[campaign] table is required")
    name = require_string(raw, "name", "campaign")
    project_root_value = raw.get("project_root", ".")
    artifact_dir_value = raw.get("artifact_dir", "artifacts")
    if not isinstance(project_root_value, str) or not project_root_value:
        raise ManifestError("campaign.project_root must be a non-empty string")
    if not isinstance(artifact_dir_value, str) or not artifact_dir_value:
        raise ManifestError("campaign.artifact_dir must be a non-empty string")
    project_root = resolve_relative(manifest_dir, project_root_value)
    artifact_path = Path(artifact_dir_value).expanduser()
    artifact_dir = (
        artifact_path.resolve()
        if artifact_path.is_absolute()
        else (project_root / artifact_path).resolve()
    )
    budget_chf = require_number(raw, "budget_chf", "campaign", minimum=0)
    usd_to_chf = require_number(raw, "usd_to_chf", "campaign", minimum=0)
    if usd_to_chf == 0:
        raise ManifestError("campaign.usd_to_chf must be greater than zero")
    require_dataset = require_bool(raw, "require_dataset", "campaign", default=True)
    require_clean_git = require_bool(raw, "require_clean_git", "campaign", default=True)
    require_metrics = require_bool(raw, "require_metrics", "campaign", default=True)
    max_parallel_memory_mib = optional_positive_integer(raw, "max_parallel_memory_mib", "campaign")
    required_secrets = parse_required_secrets(raw.get("required_secrets", []))
    return Campaign(
        name=name,
        project_root=project_root,
        artifact_dir=artifact_dir,
        budget_chf=budget_chf,
        usd_to_chf=usd_to_chf,
        require_dataset=require_dataset,
        require_clean_git=require_clean_git,
        require_metrics=require_metrics,
        max_parallel_memory_mib=max_parallel_memory_mib,
        required_secrets=required_secrets,
    )


def parse_run(
    raw: dict[str, Any],
    *,
    index: int,
    campaign: Campaign,
    base_id: str,
) -> list[PlannedRun]:
    location = f"runs[{index}]"
    command = raw.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(argument, str) or not argument for argument in command)
    ):
        raise ManifestError(f"{location}.command must be a non-empty array of strings")
    env = raw.get("env", {})
    if not isinstance(env, dict) or any(
        not isinstance(key, str) or not isinstance(value, (str, int, float, bool))
        for key, value in env.items()
    ):
        raise ManifestError(f"{location}.env must map strings to scalar values")
    for key in env:
        upper_key = key.upper()
        if any(marker in upper_key for marker in SECRET_MARKERS):
            raise ManifestError(
                f"{location}.env contains secret-like key {key!r}; inject secrets at launch time"
            )

    repetitions = require_integer(raw, "repetitions", location, minimum=1, default=1)
    repetition_start = require_integer(raw, "repetition_start", location, minimum=1, default=1)
    source_run_id_template = raw.get("source_run_id")
    if source_run_id_template is not None:
        if not isinstance(source_run_id_template, str) or not source_run_id_template:
            raise ManifestError(f"{location}.source_run_id must be a non-empty string")
        try:
            source_run_id_template.format(repetition=repetition_start)
        except (KeyError, IndexError, ValueError) as error:
            raise ManifestError(
                f"{location}.source_run_id has an invalid repetition template: {error}"
            ) from error
    estimated_cost = require_number(raw, "estimated_cost_chf", location, minimum=0)
    memory_reservation_mib = optional_positive_integer(raw, "memory_reservation_mib", location)
    memory_limit_mib = optional_positive_integer(raw, "memory_limit_mib", location)
    question_parallelism = require_integer(
        raw, "question_parallelism", location, minimum=1, default=1
    )
    if (
        memory_reservation_mib is not None
        and memory_limit_mib is not None
        and memory_reservation_mib > memory_limit_mib
    ):
        raise ManifestError(f"{location}.memory_reservation_mib cannot exceed memory_limit_mib")
    positive_cardinality = raw.get("positive_cardinality")
    if positive_cardinality is not None and (
        not isinstance(positive_cardinality, int)
        or isinstance(positive_cardinality, bool)
        or positive_cardinality < 0
    ):
        raise ManifestError(f"{location}.positive_cardinality must be a non-negative integer")
    corpus_size = raw.get("corpus_size")
    if not isinstance(corpus_size, (int, str)) or isinstance(corpus_size, bool):
        raise ManifestError(f"{location}.corpus_size must be an integer or string")
    if isinstance(corpus_size, int) and corpus_size <= 0:
        raise ManifestError(f"{location}.corpus_size must be positive")
    if isinstance(corpus_size, str) and not corpus_size:
        raise ManifestError(f"{location}.corpus_size cannot be empty")

    task = require_string(raw, "task", location)
    condition = require_string(raw, "condition", location)
    method = require_string(raw, "method", location)
    model = require_string(raw, "model", location)
    seed = require_integer(raw, "seed", location)
    normalized_env = {key: str(value) for key, value in sorted(env.items())}
    repetition_values = range(repetition_start, repetition_start + repetitions)
    return [
        PlannedRun(
            run_id=base_id if repetitions == 1 else f"{base_id}-r{repetition:02d}",
            base_id=base_id,
            campaign=campaign.name,
            task=task,
            condition=condition,
            method=method,
            model=model,
            corpus_size=corpus_size,
            positive_cardinality=positive_cardinality,
            seed=seed,
            repetition=repetition,
            command=tuple(command),
            env=normalized_env,
            estimated_cost_chf=estimated_cost,
            memory_reservation_mib=memory_reservation_mib,
            memory_limit_mib=memory_limit_mib,
            question_parallelism=question_parallelism,
            source_run_id=(
                source_run_id_template.format(repetition=repetition)
                if source_run_id_template is not None
                else None
            ),
        )
        for repetition in repetition_values
    ]


def require_string(raw: dict[str, Any], key: str, location: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def require_bool(raw: dict[str, Any], key: str, location: str, *, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ManifestError(f"{location}.{key} must be a boolean")
    return value


def require_integer(
    raw: dict[str, Any],
    key: str,
    location: str,
    *,
    minimum: int | None = None,
    default: int | None = None,
) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ManifestError(f"{location}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ManifestError(f"{location}.{key} must be >= {minimum}")
    return value


def require_number(raw: dict[str, Any], key: str, location: str, *, minimum: float) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ManifestError(f"{location}.{key} must be numeric")
    numeric = float(value)
    if numeric < minimum:
        raise ManifestError(f"{location}.{key} must be >= {minimum}")
    return numeric


def optional_positive_integer(raw: dict[str, Any], key: str, location: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ManifestError(f"{location}.{key} must be a positive integer")
    return value


def parse_required_secrets(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or any(
        not isinstance(name, str) or not ENVIRONMENT_NAME_PATTERN.fullmatch(name) for name in raw
    ):
        raise ManifestError(
            "campaign.required_secrets must be an array of uppercase environment-variable names"
        )
    if len(raw) != len(set(raw)):
        raise ManifestError("campaign.required_secrets contains duplicate names")
    return tuple(sorted(raw))


def resolve_relative(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
