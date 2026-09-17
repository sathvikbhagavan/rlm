from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/iclr2027/run_local_docker_catchup.sh"


def test_local_docker_catchup_has_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_local_docker_catchup_dry_run_covers_exact_claude_docker_scope(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "RXNHAYSTACK_DOCKER_QUEUE_DRY_RUN": "1",
            "RXNHAYSTACK_DOCKER_QUEUE_LOG": str(tmp_path / "queue.log"),
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    run_ids = [
        line.split("DRY-RUN ", 1)[1] for line in completed.stdout.splitlines() if "DRY-RUN " in line
    ]

    assert len(run_ids) == 45
    assert len(set(run_ids)) == 45
    assert all(run_id.startswith("full-claude-haiku-4.5-tier4-") for run_id in run_ids)
    assert {run_id.split("-rlm-", 1)[0].rsplit("-", 1)[1] for run_id in run_ids} == {
        "task16",
        "task17",
        "task17b",
    }
    assert {run_id.split("-rlm-", 1)[1].rsplit("-r", 1)[0] for run_id in run_ids} == {
        "x100",
        "x500",
        "xfull",
    }
    assert not any("gpt" in run_id or "deepseek" in run_id for run_id in run_ids)
