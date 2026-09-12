from __future__ import annotations

from pathlib import Path

import pytest

from rxnhaystack.manifest import ManifestError, load_manifest


def write_manifest(
    tmp_path: Path,
    runs: str,
    *,
    budget: float = 10.0,
    campaign_extra: str = "",
) -> Path:
    path = tmp_path / "campaign.toml"
    path.write_text(
        f"""
schema_version = 1

[campaign]
name = "test-campaign"
project_root = "."
artifact_dir = "artifacts"
budget_chf = {budget}
usd_to_chf = 0.80
require_dataset = false
require_clean_git = false
{campaign_extra}

{runs}
""".strip()
        + "\n"
    )
    return path


def valid_run(**overrides: str) -> str:
    values = {
        "id": '"oracle-task10"',
        "task": '"tier3/task10"',
        "condition": '"oracle-predicate"',
        "method": '"rlm"',
        "model": '"openai/gpt-5-mini"',
        "corpus_size": '"full"',
        "positive_cardinality": "5",
        "seed": "42",
        "repetitions": "2",
        "estimated_cost_chf": "1.25",
        "command": '["python", "tier3/oracle_task10.py"]',
    }
    values.update(overrides)
    body = "\n".join(f"{key} = {value}" for key, value in values.items())
    return f"[[runs]]\n{body}"


def test_manifest_expands_repetitions_and_resolves_paths(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, valid_run()))

    assert [run.run_id for run in manifest.runs] == ["oracle-task10-r01", "oracle-task10-r02"]
    assert [run.repetition for run in manifest.runs] == [1, 2]
    assert manifest.estimated_cost_chf == 2.5
    assert manifest.campaign.project_root == tmp_path.resolve()
    assert manifest.campaign.artifact_dir == (tmp_path / "artifacts").resolve()
    assert len(manifest.sha256) == 64
    assert manifest.runs[0].spec_hash == manifest.runs[0].spec_hash


def test_manifest_parses_memory_admission_and_watchdog_limits(tmp_path: Path) -> None:
    run = valid_run(
        memory_reservation_mib="1024",
        memory_limit_mib="1536",
        question_parallelism="4",
    )
    manifest = load_manifest(
        write_manifest(
            tmp_path,
            run,
            campaign_extra="max_parallel_memory_mib = 4096",
        )
    )

    assert manifest.campaign.max_parallel_memory_mib == 4096
    assert manifest.runs[0].memory_reservation_mib == 1024
    assert manifest.runs[0].memory_limit_mib == 1536
    assert manifest.runs[0].question_parallelism == 4


def test_manifest_parses_sorted_required_secret_names(tmp_path: Path) -> None:
    manifest = load_manifest(
        write_manifest(
            tmp_path,
            valid_run(),
            campaign_extra=('required_secrets = ["WANDB_API_KEY", "OPENROUTER_API_KEY"]'),
        )
    )

    assert manifest.campaign.required_secrets == ("OPENROUTER_API_KEY", "WANDB_API_KEY")


@pytest.mark.parametrize(
    "value",
    ['["lowercase"]', '["OPENROUTER_API_KEY", "OPENROUTER_API_KEY"]', '"not-an-array"'],
)
def test_manifest_rejects_invalid_required_secrets(tmp_path: Path, value: str) -> None:
    with pytest.raises(ManifestError, match="required_secrets"):
        load_manifest(
            write_manifest(
                tmp_path,
                valid_run(),
                campaign_extra=f"required_secrets = {value}",
            )
        )


def test_manifest_requires_reservations_when_campaign_memory_budget_is_set(
    tmp_path: Path,
) -> None:
    with pytest.raises(ManifestError, match="needs memory_reservation_mib"):
        load_manifest(
            write_manifest(
                tmp_path,
                valid_run(),
                campaign_extra="max_parallel_memory_mib = 4096",
            )
        )


def test_manifest_rejects_reservation_above_watchdog_limit(tmp_path: Path) -> None:
    run = valid_run(memory_reservation_mib="2048", memory_limit_mib="1024")
    with pytest.raises(ManifestError, match="cannot exceed memory_limit_mib"):
        load_manifest(write_manifest(tmp_path, run))


def test_manifest_rejects_reservation_above_campaign_budget(tmp_path: Path) -> None:
    run = valid_run(memory_reservation_mib="4097")
    with pytest.raises(ManifestError, match="exceeds campaign limit"):
        load_manifest(
            write_manifest(
                tmp_path,
                run,
                campaign_extra="max_parallel_memory_mib = 4096",
            )
        )


def test_manifest_excludes_disabled_runs(tmp_path: Path) -> None:
    enabled = valid_run(repetitions="1")
    disabled = valid_run(id='"disabled"', enabled="false")
    manifest = load_manifest(write_manifest(tmp_path, f"{enabled}\n\n{disabled}"))

    assert [run.run_id for run in manifest.runs] == ["oracle-task10"]


def test_manifest_rejects_campaign_over_budget(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="exceeds budget"):
        load_manifest(write_manifest(tmp_path, valid_run(), budget=2.0))


def test_manifest_rejects_duplicate_base_ids(tmp_path: Path) -> None:
    run = valid_run(repetitions="1")
    with pytest.raises(ManifestError, match="Duplicate run id"):
        load_manifest(write_manifest(tmp_path, f"{run}\n\n{run}"))


@pytest.mark.parametrize("identifier", ["UPPER", "spaces are bad", "../escape", "-prefix"])
def test_manifest_rejects_unsafe_run_ids(tmp_path: Path, identifier: str) -> None:
    with pytest.raises(ManifestError, match="must match"):
        load_manifest(write_manifest(tmp_path, valid_run(id=f'"{identifier}"')))


@pytest.mark.parametrize("key", ["OPENROUTER_API_KEY", "access_token", "my_secret"])
def test_manifest_rejects_secret_like_environment_keys(tmp_path: Path, key: str) -> None:
    run = valid_run(env=f'{{ {key} = "must-not-be-committed" }}')
    with pytest.raises(ManifestError, match="secret-like key"):
        load_manifest(write_manifest(tmp_path, run))


def test_manifest_rejects_zero_repetitions(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="repetitions must be >= 1"):
        load_manifest(write_manifest(tmp_path, valid_run(repetitions="0")))


def test_manifest_rejects_zero_question_parallelism(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="question_parallelism must be >= 1"):
        load_manifest(write_manifest(tmp_path, valid_run(question_parallelism="0")))


def test_run_spec_is_stable_across_manifest_locations(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = load_manifest(write_manifest(first_dir, valid_run()))
    second = load_manifest(write_manifest(second_dir, valid_run()))

    assert first.runs[0].spec_hash == second.runs[0].spec_hash
