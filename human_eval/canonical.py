from __future__ import annotations

import ast
import hashlib
import importlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .schema import GroundTruth, Question, content_sha256

EXPECTED = {
    "tiers": {1: 10, 2: 20, 3: 35, 4: 35},
    "tier2": {
        "molecular-weight-change": 6,
        "ring-count-change": 5,
        "aromatic-ring-formation": 5,
        "combined-mw-ring": 4,
    },
    "tier3": {
        "bond-level": 3,
        "scaffold": 3,
        "reagent": 2,
        "stereochemistry": 2,
        "functional-groups": 15,
        "mechanism-templates": 10,
    },
    "tier4": {
        "mechanical-graph": 5,
        "chemically-constrained-graph": 10,
        "prospective-multiconstraint": 20,
    },
}

SUGGESTED_MINUTES_BY_TIER = {1: 5, 2: 10, 3: 10, 4: 15}


def literal_constants(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        target = None
        value = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        if target and value is not None:
            try:
                values[target] = ast.literal_eval(value)
            except (ValueError, TypeError):
                pass
    return values


def source_function(path: Path, name: str, globals_: dict[str, Any] | None = None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    namespace.update(literal_constants(path))
    namespace.update(globals_ or {})
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def clean_prompt(value: str) -> str:
    lines = value.strip().splitlines()
    nonempty = [line for line in lines if line.strip()]
    margin = min((len(line) - len(line.lstrip()) for line in nonempty), default=0)
    return "\n".join(line[margin:].rstrip() for line in lines).strip()


@contextmanager
def import_area(root: Path, area: str) -> Iterator[None]:
    path = str(root / area)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)


def imported(area: str, module: str, root: Path):
    with import_area(root, area):
        sys.modules.pop(module, None)
        return importlib.import_module(module)


class BundleBuilder:
    def __init__(self, root: Path, dataset_sha256: str):
        self.root = root.resolve()
        self.dataset_sha256 = dataset_sha256
        self.questions: list[Question] = []
        self.answers: list[GroundTruth] = []

    def add(
        self,
        *,
        tier: int,
        category: str,
        subcategory: str,
        key: str,
        prompt: str,
        answer_type: str,
        answer: Any,
        sources: list[str],
        evaluator: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        minutes: int | None = None,
    ) -> None:
        slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")
        question_id = f"rxh-t{tier}-{slug}"
        gt_ref = f"gt:{question_id}"
        indices = sorted({int(i) for i in flatten_indices(answer)})
        scoring = evaluator or {
            "method": "exact set equality plus precision/recall/F1",
            "normalization": "trim whitespace; parse complete comma-delimited entries",
        }
        question = Question(
            question_id=question_id,
            tier=tier,
            category=category,
            subcategory=subcategory,
            canonical_prompt=clean_prompt(prompt),
            answer_type=answer_type,
            source_files=tuple(sources),
            dataset_sha256=self.dataset_sha256,
            ground_truth_ref=gt_ref,
            suggested_time_minutes=(
                minutes if minutes is not None else SUGGESTED_MINUTES_BY_TIER[tier]
            ),
            scoring=scoring,
            metadata=metadata or {},
        )
        question.validate()
        self.questions.append(question)
        self.answers.append(
            GroundTruth(gt_ref, question_id, answer, tuple(indices), scoring, tuple(sources))
        )

    def build(self) -> tuple[list[Question], list[GroundTruth]]:
        self.tier1()
        self.tier2()
        self.tier3()
        self.tier4()
        validate_taxonomy(self.questions)
        return self.questions, self.answers

    def tier1(self) -> None:
        area = self.root / "tier1"
        constants = literal_constants(area / "task1_hardcoded_cases.py")
        prompt_fn = source_function(area / "llm_task1.py", "build_question")
        for number, (product, answer) in enumerate(
            zip(
                constants["TASK1_HARDCODED_PRODUCTS"],
                constants["TASK1_HARDCODED_GROUND_TRUTH_INDICES"],
                strict=True,
            ),
            1,
        ):
            self.add(
                tier=1,
                category="structural-lookup",
                subcategory="product-lookup",
                key=f"task1-{number:02d}",
                prompt=prompt_fn(product),
                answer_type="index_set",
                answer=answer,
                sources=["tier1/llm_task1.py", "tier1/task1_hardcoded_cases.py"],
                metadata={"target_product": product},
            )

    def tier2(self) -> None:
        specs = [
            (
                2,
                "molecular-weight-change",
                "TASK2_THRESHOLDS",
                "TASK2_HARDCODED_GROUND_TRUTH_INDICES",
            ),
            (3, "ring-count-change", "TASK3_THRESHOLDS", "TASK3_HARDCODED_GROUND_TRUTH_INDICES"),
            (
                4,
                "aromatic-ring-formation",
                "TASK4_THRESHOLDS",
                "TASK4_HARDCODED_GROUND_TRUTH_INDICES",
            ),
        ]
        for task, category, thresholds_name, gt_name in specs:
            gt_file = self.root / "tier2" / f"task{task}_hardcoded_ground_truth.py"
            values = literal_constants(gt_file)
            fn = source_function(self.root / "tier2" / f"llm_task{task}.py", "build_question")
            for threshold in values[thresholds_name]:
                self.add(
                    tier=2,
                    category=category,
                    subcategory=category,
                    key=f"task{task}-{threshold}",
                    prompt=fn(threshold),
                    answer_type="index_set",
                    answer=values[gt_name][threshold],
                    sources=[f"tier2/llm_task{task}.py", f"tier2/{gt_file.name}"],
                )
        values = literal_constants(self.root / "tier2/task5_hardcoded_ground_truth.py")
        fn = source_function(self.root / "tier2/llm_task5.py", "build_question")
        for weight in values["TASK5_WEIGHT_THRESHOLDS_DA"]:
            for rings in values["TASK5_RING_X_VALUES"]:
                answer = values["TASK5_HARDCODED_GROUND_TRUTH_INDICES"][(weight, rings)]
                self.add(
                    tier=2,
                    category="combined-mw-ring",
                    subcategory="combined-mw-ring",
                    key=f"task5-{weight}-{rings}",
                    prompt=fn(weight, rings),
                    answer_type="index_set",
                    answer=answer,
                    sources=["tier2/llm_task5.py", "tier2/task5_hardcoded_ground_truth.py"],
                )

    def tier3(self) -> None:
        for task in (6, 7, 8):
            source = self.root / f"tier3/llm_task{task}.py"
            values = literal_constants(source)
            gt_values = literal_constants(self.root / f"tier3/task{task}_hardcoded_ground_truth.py")
            labels = next(v for k, v in values.items() if k.endswith("LABELS"))
            descriptions = next(v for k, v in values.items() if k.endswith("DESCRIPTIONS"))
            gt = gt_values[f"TASK{task}_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION"]
            fn = source_function(source, "build_question")
            smirks = next(v for k, v in gt_values.items() if k.endswith("SMIRKS"))
            for key in gt:
                configured = key in labels and key in descriptions
                label = labels.get(key, key.replace("_", " ").title())
                description = descriptions.get(
                    key, f"Stored reaction template (SMIRKS): {smirks[key]}"
                )
                metadata = (
                    {}
                    if configured
                    else {
                        "extraction_warning": "Prompt label/description is commented out in the current runner; mechanically derived from the stored ground-truth key and SMIRKS."
                    }
                )
                self.add(
                    tier=3,
                    category="functional-groups",
                    subcategory=f"task{task}",
                    key=f"task{task}-{key}",
                    prompt=fn(label, description),
                    answer_type="index_set",
                    answer=gt[key],
                    sources=[
                        f"tier3/llm_task{task}.py",
                        f"tier3/task{task}_hardcoded_ground_truth.py",
                    ],
                    metadata=metadata,
                )
        for task_name in ("10", "10b"):
            config = imported("tier3", f"task{task_name}_prompt_config", self.root)
            gt_values = literal_constants(
                self.root / f"tier3/task{task_name}_hardcoded_ground_truth.py"
            )
            gt = gt_values[f"TASK{task_name.upper()}_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION"]
            keys = list(config.REACTION_KEYS)
            for key in keys:
                self.add(
                    tier=3,
                    category="mechanism-templates",
                    subcategory=f"task{task_name}",
                    key=f"task{task_name}-{key}",
                    prompt=config.build_task10_question(key, allow_code=False)
                    if task_name == "10"
                    else config.build_task10b_question(key, allow_code=False),
                    answer_type="index_set",
                    answer=gt[key],
                    sources=[
                        f"tier3/task{task_name}_prompt_config.py",
                        f"tier3/task{task_name}_hardcoded_ground_truth.py",
                    ],
                )
        singleton_groups = {
            11: "bond-level",
            12: "bond-level",
            15: "bond-level",
            18: "scaffold",
            19: "scaffold",
            20: "scaffold",
            21: "reagent",
            22: "reagent",
            23: "stereochemistry",
            24: "stereochemistry",
        }
        for task, category in singleton_groups.items():
            prompt_file = self.root / f"tier3/llm_task{task}.py"
            gt_file = self.root / f"tier3/task{task}_hardcoded_ground_truth.py"
            prompt = source_function(prompt_file, "build_question")()
            answer = literal_constants(gt_file)[f"TASK{task}_HARDCODED_GROUND_TRUTH_INDICES"]
            self.add(
                tier=3,
                category=category,
                subcategory=f"task{task}",
                key=f"task{task}",
                prompt=prompt,
                answer_type="index_set",
                answer=answer,
                sources=[f"tier3/llm_task{task}.py", f"tier3/{gt_file.name}"],
            )

    def tier4(self) -> None:
        # Imports are deliberate: these are the benchmark's own question builders and dataclasses.
        with import_area(self.root, "tier4"):
            t11 = imported("tier4", "task11_synthetic_chain_ground_truth", self.root)
            prompt11 = source_function(self.root / "tier4/rlm_task11.py", "build_question")
            for start, length in t11.FIXED_QUESTIONS:
                answer = [list(x) for x in t11.HARDCODED_GT_CHAINS[(start, length)]]
                self.add(
                    tier=4,
                    category="mechanical-graph",
                    subcategory="synthetic-chain",
                    key=f"task11-{start}-{length}",
                    prompt=prompt11(start, length),
                    answer_type="reaction_chains",
                    answer=answer,
                    sources=["tier4/rlm_task11.py", "tier4/task11_synthetic_chain_ground_truth.py"],
                )
            t12 = imported("tier4", "task12_longest_chain_ground_truth", self.root)
            prompt12 = source_function(self.root / "tier4/rlm_task12.py", "build_question")
            for number, product in enumerate(t12.FIXED_TARGET_PRODUCTS, 1):
                self.add(
                    tier=4,
                    category="mechanical-graph",
                    subcategory="longest-chain",
                    key=f"task12-{number}",
                    prompt=prompt12(product),
                    answer_type="single_chain",
                    answer=list(t12.HARDCODED_GT_LONGEST_CHAIN[product]),
                    sources=["tier4/rlm_task12.py", "tier4/task12_longest_chain_ground_truth.py"],
                )
            t12b = imported("tier4", "task12b_hub_molecule_ground_truth", self.root)
            prompt12b = source_function(
                self.root / "tier4/rlm_task12b.py",
                "build_question",
                {"TASK12B_MIN_DOWNSTREAM": t12b.TASK12B_MIN_DOWNSTREAM},
            )
            self.add(
                tier=4,
                category="mechanical-graph",
                subcategory="hub-molecule",
                key="task12b",
                prompt=prompt12b(),
                answer_type="smiles_set",
                answer=list(t12b.HARDCODED_GT_HUB_MOLECULES),
                sources=["tier4/rlm_task12b.py", "tier4/task12b_hub_molecule_ground_truth.py"],
            )

            t13g = imported("tier4", "task13_fg_chain_graph", self.root)
            t13 = imported("tier4", "task13_fg_chain_ground_truth", self.root)
            for q in t13.FIXED_QUESTIONS:
                answer = [list(x) for x in t13.hardcoded_chains_for_pair(q.source_fg, q.target_fg)]
                prompt = t13g.build_question(
                    q.source_fg, q.target_fg, context_reaction_count=122456
                )
                self.add(
                    tier=4,
                    category="chemically-constrained-graph",
                    subcategory="functional-group-chain",
                    key=f"task13-{q.source_fg}-{q.target_fg}",
                    prompt=prompt,
                    answer_type="reaction_chains",
                    answer=answer,
                    sources=[
                        "tier4/task13_fg_chain_graph.py",
                        "tier4/task13_fg_chain_ground_truth.py",
                        "tier4/task13_fg_hardcoded_chains.json",
                    ],
                )
            t14g = imported("tier4", "task14_protecting_group_graph", self.root)
            t14 = imported("tier4", "task14_protecting_group_ground_truth", self.root)
            for q in t14.FIXED_QUESTIONS:
                pairs = t14.hardcoded_pairs_for_label(q.label)
                answer = [[x.install_index, x.remove_index] for x in pairs]
                self.add(
                    tier=4,
                    category="chemically-constrained-graph",
                    subcategory="protecting-group-pairs",
                    key=f"task14-{q.label}",
                    prompt=t14g.build_question(q, 0),
                    answer_type="reaction_pair_set",
                    answer=answer,
                    sources=[
                        "tier4/task14_protecting_group_graph.py",
                        "tier4/task14_protecting_group_ground_truth.py",
                        "tier4/task14_pg_hardcoded_pairs.json",
                    ],
                )
            t15g = imported("tier4", "task15_ring_chain_graph", self.root)
            t15 = imported("tier4", "task15_ring_chain_ground_truth", self.root)
            for q in t15.FIXED_QUESTIONS:
                spec = t15g.RING_SYSTEM_BY_LABEL[q.ring_system]
                answer = [list(x) for x in t15.hardcoded_chains_for_question(q.ring_system)]
                prompt = t15g.build_question(
                    spec,
                    context_reaction_count=122456,
                    molecule_freq_cap=t15g.MAX_MOLECULE_FREQ_REFERENCE,
                )
                self.add(
                    tier=4,
                    category="chemically-constrained-graph",
                    subcategory="ring-chain",
                    key=f"task15-{q.ring_system}",
                    prompt=prompt,
                    answer_type="reaction_chains",
                    answer=answer,
                    sources=[
                        "tier4/task15_ring_chain_graph.py",
                        "tier4/task15_ring_chain_ground_truth.py",
                        "tier4/task15_ring_hardcoded_chains.json",
                    ],
                )
            self.add_prospective()

    def add_prospective(self) -> None:
        t16g = imported("tier4", "task16_truncated_synthesis_graph", self.root)
        t16 = imported("tier4", "task16_truncated_synthesis_ground_truth", self.root)
        for q in t16.FIXED_QUESTIONS:
            answer = [list(x) for x in t16.hardcoded_prefixes_for_question(q.question_id)]
            self.add(
                tier=4,
                category="prospective-multiconstraint",
                subcategory="truncated-synthesis",
                key=f"task16-{q.question_id}",
                prompt=t16g.build_question(t16.target_spec_for_question(q)),
                answer_type="reaction_chains",
                answer=answer,
                sources=[
                    "tier4/task16_truncated_synthesis_graph.py",
                    "tier4/task16_truncated_synthesis_ground_truth.py",
                    "tier4/task16_truncated_hardcoded_chains.json",
                ],
                metadata={"conceptual_family": "prospective-truncated-synthesis"},
            )
        for suffix in ("17", "17b"):
            graph = imported("tier4", f"task{suffix}_smirks_sequential_graph", self.root)
            gt = imported("tier4", f"task{suffix}_ground_truth", self.root)
            for q in gt.FIXED_QUESTIONS:
                answer = [list(x) for x in gt.hardcoded_chains_for_question(q.question_id)]
                prompt_builder = getattr(graph, "build_question", graph.question_prompt)
                self.add(
                    tier=4,
                    category="prospective-multiconstraint",
                    subcategory=f"sequential-template-{suffix}",
                    key=f"task{suffix}-{q.question_id}",
                    prompt=prompt_builder(q),
                    answer_type="reaction_chains",
                    answer=answer,
                    sources=[
                        f"tier4/task{suffix}_smirks_sequential_graph.py",
                        f"tier4/task{suffix}_ground_truth.py",
                        f"tier4/task{suffix}_hardcoded_chains.json",
                    ],
                    metadata={"conceptual_family": "multi-constraint-sequential-template"},
                )


def flatten_indices(value: Any) -> Iterator[int]:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_indices(item)


def validate_taxonomy(questions: list[Question]) -> None:
    ids = [q.question_id for q in questions]
    if len(questions) != 100 or len(set(ids)) != 100:
        raise ValueError(
            f"Expected 100 unique questions, got {len(questions)} total/{len(set(ids))} unique"
        )
    tiers = Counter(q.tier for q in questions)
    if dict(tiers) != EXPECTED["tiers"]:
        raise ValueError(f"Tier mismatch: {dict(tiers)} != {EXPECTED['tiers']}")
    for tier, key in ((2, "tier2"), (3, "tier3"), (4, "tier4")):
        counts = Counter(q.category for q in questions if q.tier == tier)
        if dict(counts) != EXPECTED[key]:
            raise ValueError(f"{key} category mismatch: {dict(counts)} != {EXPECTED[key]}")


def build_bundle(root: Path, output: Path, dataset_sha256: str) -> dict[str, Any]:
    questions, answers = BundleBuilder(root, dataset_sha256).build()
    output.mkdir(parents=True, exist_ok=True)
    public_path = output / "questions.jsonl"
    admin_dir = output / "admin"
    admin_dir.mkdir(exist_ok=True)
    admin_path = admin_dir / "ground_truth.jsonl"
    write_jsonl(public_path, (q.to_dict() for q in questions))
    write_jsonl(admin_path, (a.to_dict() for a in answers))
    public_hash = hashlib.sha256(public_path.read_bytes()).hexdigest()
    admin_hash = hashlib.sha256(admin_path.read_bytes()).hexdigest()
    source_files = sorted({source for question in questions for source in question.source_files})
    source_checksums = {
        source: hashlib.sha256((root / source).read_bytes()).hexdigest() for source in source_files
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle_version": f"rxnhaystack-human-{SCHEMA_VERSION}",
        "question_count": len(questions),
        "dataset_sha256": dataset_sha256,
        "dataset": {
            "name": "Reaction SMILES USPTO year 2023, cleaned RxnHaystack derivative",
            "doi": "10.6084/m9.figshare.24921555.v1",
            "expected_records": 122456,
            "license": "CC BY 4.0",
        },
        "questions_sha256": public_hash,
        "admin_ground_truth_sha256": admin_hash,
        "taxonomy": EXPECTED,
        "source_root": ".",
        "excluded_historical_tier3_modules": ["task9", "task13", "task14", "task16", "task17"],
        "source_inconsistencies": [
            "Four Tier-3 functional-group definitions have stored ground truth but are commented out in current runner label/description maps; prompts are mechanically derived from key and stored SMIRKS."
        ],
        "audit_sampling": {
            "method": "SHA-256 rank without replacement over canonical JSON entries",
            "seed": 20270910,
            "default_sample_size": 25,
        },
        "ground_truth_location": "admin/ground_truth.jsonl",
        "manifest_content_sha256": content_sha256([q.to_dict() for q in questions]),
        "source_file_sha256": source_checksums,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def write_jsonl(path: Path, values: Iterator[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
