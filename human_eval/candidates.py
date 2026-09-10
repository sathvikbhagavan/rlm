from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .assignments import blinded_order
from .db import Store, utcnow

FORBIDDEN_PUBLIC_KEYS = {
    "model",
    "model_id",
    "run",
    "run_id",
    "prompting_method",
    "evaluator_outcome",
    "source_model",
    "original_outcome",
    "control_type",
    "duplicate_group",
}


def forbidden_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return (
        normalized in FORBIDDEN_PUBLIC_KEYS
        or "model" in normalized
        or "prompt" in normalized
        or "evaluator" in normalized
        or normalized.startswith("run_")
        or normalized.endswith("_run")
    )


def import_candidate_pack(store: Store, pack_path: Path, admin_dir: Path) -> dict[str, Any]:
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    required = {"schema_version", "pack_id", "version", "seed", "candidates"}
    missing = required - pack.keys()
    if missing:
        raise ValueError(f"Candidate pack missing: {sorted(missing)}")
    ids = [str(x["candidate_id"]) for x in pack["candidates"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique")
    blind_ids = {
        source_id: "cand-"
        + hashlib.sha256(f"{pack['pack_id']}:{pack['version']}:{source_id}".encode()).hexdigest()[
            :16
        ]
        for source_id in ids
    }
    blind_pack_id = (
        "pack-" + hashlib.sha256(f"{pack['pack_id']}:{pack['version']}".encode()).hexdigest()[:16]
    )
    order = blinded_order(ids, seed=int(pack["seed"]), annotator_id=str(pack["pack_id"]))
    rank = {blind_ids[item_id]: position for position, item_id in enumerate(order)}
    unblinding: dict[str, Any] = {
        "pack_id": pack["pack_id"],
        "version": pack["version"],
        "created_at": utcnow(),
        "map": {},
    }
    with store.connect() as db:
        for candidate in pack["candidates"]:
            source_id = str(candidate["candidate_id"])
            cid = blind_ids[source_id]
            secret = {"original_candidate_id": source_id}
            secret["hidden_fields"] = collect_hidden(candidate)
            public = sanitize_public(candidate)
            public.pop("candidate_id", None)
            unblinding["map"][cid] = secret
            db.execute(
                "INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?,?)",
                (
                    cid,
                    blind_pack_id,
                    str(candidate["question_id"]),
                    json.dumps(public, sort_keys=True),
                    candidate.get("control_type"),
                    candidate.get("duplicate_group"),
                    rank[cid],
                ),
            )
    admin_dir.mkdir(parents=True, exist_ok=True)
    target = (admin_dir / f"unblinding_{blind_pack_id}.json").resolve()
    if not target.is_relative_to(admin_dir.resolve()):
        raise ValueError("Unsafe unblinding-map path")
    target.write_text(json.dumps(unblinding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target.chmod(0o600)
    return {
        "pack_id": pack["pack_id"],
        "candidate_count": len(ids),
        "public_order_sha256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
        "unblinding_path": str(target),
    }


def sanitize_public(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_public(item) for key, item in value.items() if not forbidden_key(key)}
    if isinstance(value, list):
        return [sanitize_public(item) for item in value]
    return value


def collect_hidden(value: Any, prefix: str = "") -> dict[str, Any]:
    hidden: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if forbidden_key(key):
                hidden[path] = item
            else:
                hidden.update(collect_hidden(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hidden.update(collect_hidden(item, f"{prefix}[{index}]"))
    return hidden


def public_candidates(store: Store) -> list[dict[str, Any]]:
    with store.connect() as db:
        rows = db.execute(
            "SELECT candidate_id,pack_id,question_id,candidate_json,control_type,duplicate_group,public_order FROM candidates ORDER BY pack_id,public_order"
        ).fetchall()
    values = []
    for row in rows:
        item = {
            "candidate_id": row["candidate_id"],
            "pack_id": row["pack_id"],
            "question_id": row["question_id"],
            "public_order": row["public_order"],
        }
        item.update(json.loads(row["candidate_json"]))
        values.append(item)
    return values
