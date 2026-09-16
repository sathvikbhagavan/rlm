from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rxnhaystack.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
ICLR = ROOT / "experiments" / "iclr2027"
DATASET = Path(
    os.environ.get(
        "RXNHAYSTACK_CLEANED_DATASET",
        Path.home() / "datasets/rxnhaystack/reactionSmilesFigShareUSPTO2023_cleaned.txt",
    )
)

NORMAL_PROMPT_HASHES = {
    "tier3-task6": "e8b831426029229be8577e6596a2f7d33ebd171b68ec11ccc7809a40dd0e2d99",
    "tier3-task10": "92a9f5003cead2d9507a11657a40188f647fac0c2598d660f5c9ed40108e0239",
    "tier3-task23": "5d22be9c55f36fd74a7688707b005df3a21daee57b86cb19034dfdeb1ced9e3a",
    "tier4-task13": "07fca8d584631a28f477160b241ec13beaa8ffb1ec812244a320496d1df6b63d",
    "tier4-task14": "f40748cad5d22d0f8143c191d379d14206ce2623bf0c10041f5ef6996c5f5495",
}


def purge_task_modules() -> None:
    prefixes = ("oracle_predicates", "rlm_task", "task6_", "task10_", "task13_", "task14_")
    for name in tuple(sys.modules):
        if name.startswith(prefixes):
            sys.modules.pop(name, None)


def import_from_tier(monkeypatch: pytest.MonkeyPatch, tier: str, module: str):
    purge_task_modules()
    monkeypatch.syspath_prepend(str(ROOT / tier))
    return importlib.import_module(module)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_generated_oracle_experiments_are_current_and_have_exact_scope() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ICLR / "generate_oracle_predicate_campaign.py"),
            "--check",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    oracle = load_manifest(ICLR / "oracle-predicate-campaign.toml")
    executor = load_manifest(ICLR / "oracle-executor-campaign.toml")
    assert len(oracle.runs) == 150
    assert len(executor.runs) == 15
    assert oracle.estimated_cost_chf == pytest.approx(17.8962, abs=1e-6)
    assert executor.estimated_cost_chf == 0
    assert {run.model for run in oracle.runs} == {
        "RCP-AIaaS/Qwen/Qwen3.5-397B-A17B",
        "anthropic/claude-haiku-4.5",
    }
    assert {run.task for run in oracle.runs} == {
        "tier3/task6",
        "tier3/task10",
        "tier3/task23",
        "tier4/task13",
        "tier4/task14",
    }
    assert all(run.env["RXNHAYSTACK_ORACLE_PREDICATE"] == "1" for run in oracle.runs)
    question_counts = {
        "tier3/task6": 4,
        "tier3/task10": 5,
        "tier3/task23": 1,
        "tier4/task13": 4,
        "tier4/task14": 2,
    }
    assert sum(question_counts[run.task] for run in oracle.runs) == 480
    assert sum(question_counts[run.task] for run in executor.runs) == 48


def test_answer_free_helpers_contain_no_ground_truth_payloads() -> None:
    forbidden = (
        "HARDCODED_GROUND_TRUTH",
        "HARDCODED_CHAINS",
        "HARDCODED_PAIRS",
        "POSITIVE_REACTIONS",
        "support_indices",
        ".json",
    )
    for path in (ROOT / "tier3/oracle_predicates.py", ROOT / "tier4/oracle_predicates.py"):
        source = path.read_text(encoding="utf-8")
        assert all(token not in source for token in forbidden), path


def test_tier3_oracle_wrappers_match_existing_predicates(monkeypatch: pytest.MonkeyPatch) -> None:
    oracle = import_from_tier(monkeypatch, "tier3", "oracle_predicates")
    mechanism = importlib.import_module("task10_mechanism_evaluator")
    stereo = importlib.import_module("task23_stereocenter_evaluator")

    click = "0 [N-]=[N+]=NCC.C#C>Cu>c1nn[nH]c1"
    achiral = "1 CC=O>>C[C@H](O)C"
    assert oracle.task10_reaction_matches(click, "azide_alkyne_huisgen_cycloaddition") == (
        mechanism.reaction_line_matches_mechanism(
            click.split(" ", 1)[1], "azide_alkyne_huisgen_cycloaddition"
        )
    )
    assert oracle.task23_reaction_matches(
        achiral
    ) == stereo.reaction_creates_stereocenter_from_achiral(achiral.split(" ", 1)[1])


def test_tier4_oracle_chemistry_matches_existing_chemistry(monkeypatch: pytest.MonkeyPatch) -> None:
    oracle = import_from_tier(monkeypatch, "tier4", "oracle_predicates")
    task13 = importlib.import_module("task13_fg_chain_graph")
    task14 = importlib.import_module("task14_protecting_group_graph")
    for smiles in ("CCO", "CC(=O)O", "CC#N", "CCN(C)C"):
        assert oracle.task13_detect_functional_groups(smiles) == task13.detect_functional_groups(
            smiles
        )
    assert oracle.TASK14_PROTECTED_SMARTS == {
        spec.label: spec.protected_smarts for spec in task14.PROTECTING_GROUPS
    }


def test_normal_prompts_remain_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RXNHAYSTACK_ORACLE_PREDICATE", raising=False)
    task6 = import_from_tier(monkeypatch, "tier3", "rlm_task6")
    assert sha256(task6.build_question("L", "D")) == NORMAL_PROMPT_HASHES["tier3-task6"]
    task10 = import_from_tier(monkeypatch, "tier3", "rlm_task10")
    assert (
        sha256(task10.build_question("wittig_olefination")) == NORMAL_PROMPT_HASHES["tier3-task10"]
    )
    task23 = import_from_tier(monkeypatch, "tier3", "rlm_task23")
    assert sha256(task23.build_question()) == NORMAL_PROMPT_HASHES["tier3-task23"]

    task13 = import_from_tier(monkeypatch, "tier4", "task13_fg_chain_graph")
    prompt13 = task13.build_rlm_question(
        "primary_alcohol",
        "carboxylic_acid",
        context_reaction_count=100,
        molecule_freq_cap=3,
    )
    assert sha256(prompt13) == NORMAL_PROMPT_HASHES["tier4-task13"]
    task14 = import_from_tier(monkeypatch, "tier4", "task14_protecting_group_graph")
    assert (
        sha256(task14.build_rlm_question(task14.PROTECTING_GROUPS[0], 0))
        == (NORMAL_PROMPT_HASHES["tier4-task14"])
    )


def test_oracle_prompts_name_only_answer_free_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    task6 = import_from_tier(monkeypatch, "tier3", "rlm_task6")
    monkeypatch.setattr(task6, "ORACLE_PREDICATE", True)
    prompt = task6.build_run_question("ester_with_primary_amine", "label", "description")
    assert "task6_reaction_matches" in prompt
    assert "ground-truth" not in prompt.lower()
    assert "answer indices" in prompt

    oracle4 = import_from_tier(monkeypatch, "tier4", "oracle_predicates")
    prompt13 = oracle4.task13_oracle_guidance()
    prompt14 = oracle4.task14_oracle_guidance("Boc_N")
    assert "graph-search code" in prompt13
    assert "pairing code" in prompt14
    assert "hardcoded" not in (prompt13 + prompt14).lower()


@pytest.mark.skipif(
    os.environ.get("RXNHAYSTACK_RUN_FULL_ORACLE_PARITY") != "1" or not DATASET.is_file(),
    reason="Explicit full-dataset oracle parity audit",
)
def test_full_dataset_oracle_predicate_parity(tmp_path: Path) -> None:
    report = tmp_path / "oracle-parity.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ICLR / "audit_oracle_predicate_parity.py"),
            "--dataset",
            str(DATASET),
            "--task",
            "tier3-task6",
            "--task",
            "tier3-task10",
            "--task",
            "tier3-task23",
            "--task",
            "tier4-task13",
            "--task",
            "tier4-task14",
            "--output",
            str(report),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"match": false' not in report.read_text(encoding="utf-8").lower()
