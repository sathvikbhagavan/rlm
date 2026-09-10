from __future__ import annotations

import csv
import io
import sqlite3
from pathlib import Path
from typing import Any


def parse_reaction(raw: str) -> tuple[str, str, str]:
    parts = raw.rstrip("\n").split(">")
    if len(parts) != 3:
        return raw.rstrip("\n"), "", ""
    return parts[0], parts[1], parts[2]


class DatasetIndex:
    def __init__(self, database: str | Path, dataset: str | Path):
        self.database = Path(database).resolve()
        self.dataset = Path(dataset).resolve()

    def build(self) -> int:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.database)
        try:
            db.executescript("""PRAGMA journal_mode=WAL; DROP TABLE IF EXISTS reactions;
            CREATE TABLE reactions(idx INTEGER PRIMARY KEY, raw TEXT NOT NULL, reactants TEXT NOT NULL, reagents TEXT NOT NULL, products TEXT NOT NULL);
            CREATE INDEX idx_reactants ON reactions(reactants); CREATE INDEX idx_products ON reactions(products);""")
            batch = []
            with self.dataset.open(encoding="utf-8") as handle:
                for idx, raw in enumerate(handle):
                    reactants, reagents, products = parse_reaction(raw)
                    batch.append((idx, raw.rstrip("\n"), reactants, reagents, products))
                    if len(batch) == 1000:
                        db.executemany("INSERT INTO reactions VALUES (?,?,?,?,?)", batch)
                        batch.clear()
            if batch:
                db.executemany("INSERT INTO reactions VALUES (?,?,?,?,?)", batch)
            db.commit()
            return int(db.execute("SELECT COUNT(*) FROM reactions").fetchone()[0])
        finally:
            db.close()

    def query(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        search: str = "",
        exact: bool = False,
        index: int | None = None,
    ) -> dict[str, Any]:
        page = max(1, page)
        page_size = min(100, max(1, page_size))
        db = sqlite3.connect(self.database)
        db.row_factory = sqlite3.Row
        try:
            params: list[Any] = []
            where = ""
            if index is not None:
                where, params = " WHERE idx=?", [index]
            elif search:
                if exact:
                    where = " WHERE reactants=? OR reagents=? OR products=? OR instr('.'||reactants||'.','.'||?||'.')>0 OR instr('.'||reagents||'.','.'||?||'.')>0 OR instr('.'||products||'.','.'||?||'.')>0"
                    params = [search] * 6
                else:
                    escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                    where, params = " WHERE raw LIKE ? ESCAPE '\\'", [f"%{escaped}%"]
            total = int(db.execute("SELECT COUNT(*) FROM reactions" + where, params).fetchone()[0])
            rows = [
                dict(x)
                for x in db.execute(
                    "SELECT * FROM reactions" + where + " ORDER BY idx LIMIT ? OFFSET ?",
                    [*params, page_size, (page - 1) * page_size],
                )
            ]
            return {"rows": rows, "total": total, "page": page, "page_size": page_size}
        finally:
            db.close()

    def export_rows(self, indices: list[int], format_: str) -> bytes:
        clean = sorted(set(indices))[:10000]
        if not clean:
            return b""
        db = sqlite3.connect(self.database)
        db.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in clean)
        rows = [
            dict(x)
            for x in db.execute(
                f"SELECT * FROM reactions WHERE idx IN ({placeholders}) ORDER BY idx", clean
            )
        ]
        db.close()
        if format_ == "jsonl":
            import json

            return ("\n".join(json.dumps(x, sort_keys=True) for x in rows) + "\n").encode()
        output = io.StringIO()
        writer = csv.DictWriter(
            output, fieldnames=["idx", "reactants", "reagents", "products", "raw"]
        )
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode()
