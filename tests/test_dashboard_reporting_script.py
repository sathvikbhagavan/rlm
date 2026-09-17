from __future__ import annotations

import subprocess
from pathlib import Path


def test_dashboard_reporting_helper_has_valid_shell_syntax() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "iclr2027" / "start_dashboard_reporting.sh"

    subprocess.run(["bash", "-n", str(script)], check=True)
    content = script.read_text(encoding="utf-8")
    assert "iclr2027-six-model-full-v34/ledger.sqlite3" in content
    assert "iclr2027-matched-cardinality-v7/ledger.sqlite3" in content
    assert "--local-only" in content
    assert "tmux has-session" in content
