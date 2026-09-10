from __future__ import annotations

import json
from pathlib import Path

from human_eval.dataset_browser import DatasetIndex


def test_pagination_search_index_and_exports(tmp_path: Path, tiny_dataset: Path):
    index = DatasetIndex(tmp_path / "index.sqlite", tiny_dataset)
    assert index.build() == 3
    assert index.query(page=1, page_size=2)["rows"][1]["idx"] == 1
    assert index.query(index=2)["rows"][0]["products"] == "N=N"
    assert index.query(search="CCO", exact=False)["total"] == 1
    assert index.query(search="CCO", exact=True)["total"] == 1
    assert b"reactants" in index.export_rows([0, 2], "csv")
    values = [json.loads(x) for x in index.export_rows([1], "jsonl").splitlines()]
    assert values[0]["idx"] == 1
