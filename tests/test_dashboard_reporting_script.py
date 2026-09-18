from __future__ import annotations

import subprocess
from pathlib import Path


def test_dashboard_reporting_helper_has_valid_shell_syntax() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "iclr2027" / "start_dashboard_reporting.sh"

    subprocess.run(["bash", "-n", str(script)], check=True)
    content = script.read_text(encoding="utf-8")
    assert "iclr2027-six-model-full-v34" in content
    assert "iclr2027-matched-cardinality-v7" in content
    assert "iclr2027-oracle-predicate-v1" in content
    assert "iclr2027-oracle-executor-v1" in content
    assert "iclr2027-task16-prospective-decomposition-v1" in content
    assert "iclr2027-gpt5mini-direct-openai-docker-v1" in content
    assert 'dashboard_ledger="$dashboard_campaign_dir/ledger.sqlite3"' in content
    assert "--local-only" in content
    assert "tmux has-session" in content
    assert "RXNHAYSTACK_DASHBOARD_ENTITY" in content
    assert "-prune" in content
    assert 'echo "Scanning $dashboard_search_root' in content
    assert 'dashboard_mode="${4:-ensure}"' in content
    assert "dashboard_machine:$dashboard_kind" in content


def test_dashboard_viewer_helper_has_valid_shell_syntax() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "iclr2027" / "start_dashboard_viewer.sh"

    subprocess.run(["bash", "-n", str(script)], check=True)
    content = script.read_text(encoding="utf-8")
    assert "rxnhaystack-dashboard" in content
    assert "sathvikbhagavan-epfl" in content
    assert "127.0.0.1:8765" in content
