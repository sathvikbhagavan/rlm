# ruff: noqa: E402, I001

from __future__ import annotations

from pathlib import Path
import sys


TIER4 = Path(__file__).resolve().parents[1] / "tier4"
sys.path.insert(0, str(TIER4))

import task15_ring_chain_graph as graph  # noqa: E402


REACTION_LINES = [
    "10 CC.O>>CCO",
    "20 CCO>>CC=O",
    "30 CC=O>>CC(=O)O",
]


def test_empty_prediction_does_not_parse_or_canonicalize(monkeypatch) -> None:
    def unexpected(_: str) -> list[str]:
        raise AssertionError("empty predictions must not canonicalize corpus rows")

    monkeypatch.setattr(graph, "canonicalize_components", unexpected)

    assert graph.parse_selected_records_from_lines(REACTION_LINES, ()) == {}


def test_only_predicted_records_are_parsed(monkeypatch) -> None:
    calls: list[str] = []
    original = graph.canonicalize_components

    def recording(value: str) -> list[str]:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(graph, "canonicalize_components", recording)
    records = graph.parse_selected_records_from_lines(REACTION_LINES, (20,))

    assert set(records) == {20}
    assert calls == ["CCO", "CC=O"]


def test_selected_records_equal_full_parser_subset() -> None:
    full = graph.parse_records_from_lines(REACTION_LINES)
    selected = graph.parse_selected_records_from_lines(REACTION_LINES, (30, 10))

    assert selected == {10: full[10], 30: full[30]}


def test_unknown_prediction_remains_absent() -> None:
    assert graph.parse_selected_records_from_lines(REACTION_LINES, (999,)) == {}
