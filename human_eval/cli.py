from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rxnhaystack.dataset import verify_cleaned

from .analysis import analyze
from .app import create_app
from .assignments import assign_questions
from .candidate_export import build_candidate_pack, write_candidate_pack
from .candidates import import_candidate_pack
from .canonical import build_bundle, validate_taxonomy
from .dataset_browser import DatasetIndex
from .db import Store
from .exporting import export_filename, export_zip
from .restoring import restore_export
from .schema import Question

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path.home() / "datasets/rxnhaystack/reactionSmilesFigShareUSPTO2023_cleaned.txt"
DEFAULT_BUNDLE = ROOT / "human_eval/generated/canonical-v4"
DEFAULT_STATE = ROOT / "human_eval/local_state"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="RxnHaystack human-validation application")
    commands = value.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-bundle")
    build.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    build.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    validate = commands.add_parser("validate-bundle")
    validate.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    index = commands.add_parser("index-dataset")
    index.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    index.add_argument("--state", type=Path, default=DEFAULT_STATE)
    serve = commands.add_parser("serve")
    serve.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    serve.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    serve.add_argument("--state", type=Path, default=DEFAULT_STATE)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--allow-network", action="store_true")
    candidates = commands.add_parser("import-candidates")
    candidates.add_argument("pack", type=Path)

    export_candidates = commands.add_parser("export-task16-candidates")
    export_candidates.add_argument("output", type=Path)
    export_candidates.add_argument("predictions", nargs="+", type=Path)
    export_candidates.add_argument("--pack-id", required=True)
    export_candidates.add_argument("--version", default="1.0.0")
    export_candidates.add_argument("--seed", type=int, default=20270910)
    export_candidates.add_argument("--max-false-positives", type=int)
    export_candidates.add_argument("--positive-controls-per-question", type=int, default=1)
    candidates.add_argument("--state", type=Path, default=DEFAULT_STATE)
    study = commands.add_parser("install-study")
    study.add_argument("manifest", type=Path)
    study.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    study.add_argument("--state", type=Path, default=DEFAULT_STATE)
    export = commands.add_parser("export")
    export.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    export.add_argument("--state", type=Path, default=DEFAULT_STATE)
    export.add_argument("--output", type=Path)
    restore = commands.add_parser("restore")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--state", type=Path, default=DEFAULT_STATE)
    report = commands.add_parser("analyze")
    report.add_argument("exports", nargs="+", type=Path)
    report.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--unblinding", action="append", type=Path, default=[])
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "build-bundle":
        provenance = verify_cleaned(args.dataset)
        manifest = build_bundle(ROOT, args.output, provenance.sha256)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "questions": manifest["question_count"],
                    "taxonomy": manifest["taxonomy"],
                    "questions_sha256": manifest["questions_sha256"],
                    "dataset_lines": provenance.line_count,
                    "dataset_sha256": provenance.sha256,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "validate-bundle":
        questions = [Question.from_dict(x) for x in read_jsonl(args.bundle / "questions.jsonl")]
        validate_taxonomy(questions)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "questions": len(questions),
                    "taxonomy": json.loads((args.bundle / "manifest.json").read_text())["taxonomy"],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "index-dataset":
        provenance = verify_cleaned(args.dataset)
        count = DatasetIndex(args.state / "dataset_index.sqlite3", args.dataset).build()
        if count != provenance.line_count:
            raise ValueError(f"Index count {count} != dataset count {provenance.line_count}")
        print(
            json.dumps(
                {
                    "status": "indexed",
                    "records": count,
                    "database": str((args.state / "dataset_index.sqlite3").resolve()),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "serve":
        if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_network:
            raise SystemExit("Refusing network exposure without --allow-network")
        import uvicorn

        uvicorn.run(
            create_app(
                bundle_dir=args.bundle,
                state_dir=args.state,
                dataset_path=args.dataset,
                allow_network=args.allow_network,
            ),
            host=args.host,
            port=args.port,
        )
        return 0
    if args.command == "analyze":
        summary = analyze(
            args.exports,
            args.bundle / "questions.jsonl",
            args.bundle / "admin/ground_truth.jsonl",
            args.output,
            args.unblinding,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "export-task16-candidates":
        payload = build_candidate_pack(
            args.predictions,
            pack_id=args.pack_id,
            version=args.version,
            seed=args.seed,
            max_false_positives=args.max_false_positives,
            positive_controls_per_question=args.positive_controls_per_question,
        )
        output = write_candidate_pack(payload, args.output)
        print(
            json.dumps(
                {"output": str(output), "candidate_count": len(payload["candidates"])}, indent=2
            )
        )
        return 0
    store = Store(args.state / "annotations.sqlite3")
    if args.command == "import-candidates":
        print(json.dumps(import_candidate_pack(store, args.pack, args.state / "admin"), indent=2))
        return 0
    if args.command == "install-study":
        manifest = json.loads(args.manifest.read_text())
        questions = read_jsonl(args.bundle / "questions.jsonl")
        assignment_items = questions
        if manifest.get("mode") == "prospective":
            from .candidates import public_candidates

            categories = {q["question_id"]: q["category"] for q in questions}
            assignment_items = [
                {
                    "question_id": candidate["candidate_id"],
                    "category": categories.get(candidate["question_id"], "prospective"),
                }
                for candidate in public_candidates(store)
            ]
        annotators = manifest.get("annotators") or [
            {"annotator_id": store.profile()["annotator_id"], "quotas": manifest.get("quotas")}
        ]
        assignments = {
            str(a["annotator_id"]): assign_questions(
                assignment_items,
                seed=int(manifest["seed"]),
                quotas=a.get("quotas"),
                annotator_id=str(a["annotator_id"]),
            )
            for a in annotators
        }
        store.install_study(manifest, assignments)
        print(
            json.dumps(
                {
                    "study_id": manifest["study_id"],
                    "assignment_counts": {k: len(v) for k, v in assignments.items()},
                },
                indent=2,
            )
        )
        return 0
    if args.command == "export":
        manifest = json.loads((args.bundle / "manifest.json").read_text())
        annotator = store.profile()["annotator_id"]
        destination = args.output or Path(export_filename(annotator))
        destination.write_bytes(export_zip(store, annotator, manifest))
        print(destination.resolve())
        return 0
    if args.command == "restore":
        result = restore_export(store, args.archive.read_bytes())
        print(json.dumps(result, indent=2))
        return 0
    return 2


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    sys.exit(main())
