from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rxnhaystack import cli


def test_plan_prints_dataset_rows_and_fingerprints(monkeypatch, capsys) -> None:
    run = SimpleNamespace(
        run_id="example-r01",
        task="tier1/task1",
        condition="full-x100",
        method="llm",
        model="model",
        corpus_size=100,
        estimated_cost_chf=0.1,
        memory_reservation_mib=256,
        memory_limit_mib=512,
        question_parallelism=1,
    )
    manifest = SimpleNamespace(
        campaign=SimpleNamespace(
            name="example",
            max_parallel_memory_mib=1024,
        ),
        runs=(run,),
    )
    prepared = SimpleNamespace(
        git=SimpleNamespace(commit="abcdef1234567890"),
        dataset=SimpleNamespace(
            raw_lines=137_261,
            raw_sha256="raw-fingerprint",
            cleaned_lines=122_456,
            cleaned_sha256="cleaned-fingerprint",
        ),
    )
    monkeypatch.setattr(cli, "load_manifest", lambda _path: manifest)
    monkeypatch.setattr(cli, "select_runs", lambda runs, _patterns: list(runs))
    monkeypatch.setattr(cli, "perform_preflight", lambda *_args, **_kwargs: prepared)
    args = SimpleNamespace(
        manifest=Path("experiment.toml"),
        select=[],
        data_dir=None,
        raw_path=None,
        cleaned_path=None,
    )

    assert cli.command_plan(args) == 0

    output = capsys.readouterr().out
    assert "git abcdef123456" in output
    assert "Dataset raw: rows=137261; sha256=raw-fingerprint" in output
    assert "Dataset cleaned: rows=122456; sha256=cleaned-fingerprint" in output
