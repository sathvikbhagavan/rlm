import argparse
import json
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rlm import RLM
from rlm.tracing import init_tracing, using_tracing_attributes

try:
    import wandb
except ImportError:
    wandb = None

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
SEED = 42

MAX_PAIRS_PER_GROUP = 0
MIN_HEAVY_ATOMS = 3
MAX_HEAVY_ATOMS = 90

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class ReactionRecord:
    index: int
    raw: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]


@dataclass(frozen=True)
class ProtectingGroupSpec:
    label: str
    functional_group: str
    protected_smarts: tuple[str, ...]
    aliases: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ProtectionEvent:
    reaction_index: int
    pg_label: str
    direction: str
    free_smiles: str
    protected_smiles: str
    scaffold_key: str


@dataclass(frozen=True)
class GroundTruthPair:
    install_index: int
    remove_index: int
    pg_label: str
    functional_group: str
    scaffold_key: str
    install_free_smiles: str
    install_protected_smiles: str
    remove_protected_smiles: str
    remove_free_smiles: str


PROTECTING_GROUPS: tuple[ProtectingGroupSpec, ...] = (
    ProtectingGroupSpec(
        label="Boc_N",
        functional_group="amine",
        protected_smarts=("[NX3][CX3](=O)[OX2][C;X4]([CH3])([CH3])[CH3]",),
        aliases=("Boc", "tert-butyloxycarbonyl", "BOC"),
        description="Boc-protected amines: N-C(=O)-O-tert-butyl carbamates.",
    ),
    ProtectingGroupSpec(
        label="Cbz_N",
        functional_group="amine",
        protected_smarts=("[NX3][CX3](=O)[OX2]Cc1ccccc1",),
        aliases=("Cbz", "Z", "benzyloxycarbonyl"),
        description="Cbz-protected amines: N-C(=O)-O-benzyl carbamates.",
    ),
    # ProtectingGroupSpec(
    #     label="Fmoc_N",
    #     functional_group="amine",
    #     protected_smarts=("[NX3][CX3](=O)[OX2]CC1c2ccccc2-c2ccccc21",),
    #     aliases=("Fmoc", "fluorenylmethoxycarbonyl"),
    #     description="Fmoc-protected amines: N-C(=O)-O-CH2-fluorenyl carbamates.",
    # ),
    ProtectingGroupSpec(
        label="benzyl_O_N",
        functional_group="alcohol_or_amine",
        protected_smarts=("[O,N]Cc1ccccc1",),
        aliases=("Bn", "benzyl"),
        description="Benzyl-protected alcohols or amines: heteroatom-CH2-phenyl.",
    ),
    # ProtectingGroupSpec(
    #     label="silyl_ether",
    #     functional_group="alcohol",
    #     protected_smarts=("[OX2][Si]",),
    #     aliases=("TBS", "TBDMS", "TMS", "TES", "silyl ether"),
    #     description="Silyl-protected alcohols: O-Si ether protecting groups.",
    # ),
)


def compile_pg_patterns() -> dict[str, list[Chem.Mol]]:
    patterns: dict[str, list[Chem.Mol]] = {}
    for spec in PROTECTING_GROUPS:
        compiled = []
        for smarts in spec.protected_smarts:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                raise ValueError(f"Invalid SMARTS for {spec.label}: {smarts}")
            compiled.append(patt)
        patterns[spec.label] = compiled
    return patterns


PG_PATTERNS = compile_pg_patterns()


def canonicalize_components(smiles_str: str) -> list[str]:
    canonical: list[str] = []
    for smi in smiles_str.split("."):
        smi = smi.strip()
        if not smi:
            continue
        try:
            csmi = Chem.CanonSmiles(smi)
            if csmi:
                canonical.append(csmi)
        except Exception:
            continue
    return canonical


def split_reaction_line(indexed_line: str) -> tuple[int, str, str]:
    idx_str, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    if len(parts) < 2:
        raise ValueError(f"Malformed reaction line: {indexed_line[:80]}")
    return int(idx_str), parts[0].strip(), parts[-1].strip()


def heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def molecule_in_size_window(smiles: str) -> bool:
    heavy = heavy_atom_count(smiles)
    return MIN_HEAVY_ATOMS <= heavy <= MAX_HEAVY_ATOMS


def parse_dataset(dataset_path: str) -> dict[int, ReactionRecord]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]

    records: dict[int, ReactionRecord] = {}
    for i, raw in enumerate(raw_lines):
        indexed_line = f"{i} {raw}"
        try:
            idx, reactants_raw, products_raw = split_reaction_line(indexed_line)
        except Exception:
            continue
        reactants = tuple(canonicalize_components(reactants_raw))
        products = tuple(canonicalize_components(products_raw))
        if not reactants or not products:
            continue
        records[idx] = ReactionRecord(
            index=idx,
            raw=raw,
            reactants=reactants,
            products=products,
        )
    return records


def has_pg(smiles: str, pg_label: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(mol.HasSubstructMatch(pattern) for pattern in PG_PATTERNS[pg_label])


def stripped_scaffold_keys(smiles: str, pg_label: str) -> tuple[str, ...]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tuple()

    keys: set[str] = set()
    for pattern in PG_PATTERNS[pg_label]:
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            # SMARTS patterns put the protected heteroatom first. Keep that atom
            # and delete the protecting-group atoms to normalize one protected site.
            atoms_to_remove = sorted(match[1:], reverse=True)
            editable = Chem.RWMol(mol)
            for atom_idx in atoms_to_remove:
                editable.RemoveAtom(atom_idx)
            try:
                stripped = editable.GetMol()
                Chem.SanitizeMol(stripped)
                keys.add(Chem.MolToSmiles(stripped, canonical=True))
            except Exception:
                continue
    return tuple(sorted(keys))


def stripped_scaffold_key(smiles: str, pg_label: str) -> str | None:
    keys = stripped_scaffold_keys(smiles, pg_label)
    return keys[0] if keys else None


def free_scaffold_key(smiles: str) -> str | None:
    try:
        return Chem.CanonSmiles(smiles)
    except Exception:
        return None


def mine_protection_events(records: dict[int, ReactionRecord]) -> list[ProtectionEvent]:
    events: list[ProtectionEvent] = []
    heavy_cache: dict[str, bool] = {}
    has_pg_cache: dict[tuple[str, str], bool] = {}
    free_key_cache: dict[str, str | None] = {}
    stripped_key_cache: dict[tuple[str, str], tuple[str, ...]] = {}

    def cached_in_size_window(smiles: str) -> bool:
        if smiles not in heavy_cache:
            heavy_cache[smiles] = molecule_in_size_window(smiles)
        return heavy_cache[smiles]

    def cached_has_pg(smiles: str, pg_label: str) -> bool:
        key = (smiles, pg_label)
        if key not in has_pg_cache:
            has_pg_cache[key] = has_pg(smiles, pg_label)
        return has_pg_cache[key]

    def cached_free_key(smiles: str) -> str | None:
        if smiles not in free_key_cache:
            free_key_cache[smiles] = free_scaffold_key(smiles)
        return free_key_cache[smiles]

    def cached_stripped_keys(smiles: str, pg_label: str) -> tuple[str, ...]:
        key = (smiles, pg_label)
        if key not in stripped_key_cache:
            stripped_key_cache[key] = stripped_scaffold_keys(smiles, pg_label)
        return stripped_key_cache[key]

    for rec in records.values():
        reactants = [smi for smi in rec.reactants if cached_in_size_window(smi)]
        products = [smi for smi in rec.products if cached_in_size_window(smi)]
        if not reactants or not products:
            continue

        for spec in PROTECTING_GROUPS:
            reactant_pg = [smi for smi in reactants if cached_has_pg(smi, spec.label)]
            product_pg = [smi for smi in products if cached_has_pg(smi, spec.label)]

            for free_smi in reactants:
                free_key = cached_free_key(free_smi)
                if free_key is None:
                    continue
                for protected_smi in product_pg:
                    if free_key in cached_stripped_keys(protected_smi, spec.label):
                        events.append(
                            ProtectionEvent(
                                reaction_index=rec.index,
                                pg_label=spec.label,
                                direction="install",
                                free_smiles=free_smi,
                                protected_smiles=protected_smi,
                                scaffold_key=free_key,
                            )
                        )

            for protected_smi in reactant_pg:
                protected_keys = cached_stripped_keys(protected_smi, spec.label)
                if not protected_keys:
                    continue
                for free_smi in products:
                    free_key = cached_free_key(free_smi)
                    if free_key is None:
                        continue
                    if free_key in protected_keys:
                        events.append(
                            ProtectionEvent(
                                reaction_index=rec.index,
                                pg_label=spec.label,
                                direction="remove",
                                free_smiles=free_smi,
                                protected_smiles=protected_smi,
                                scaffold_key=free_key,
                            )
                        )
    return sorted(
        set(events),
        key=lambda event: (
            event.pg_label,
            event.scaffold_key,
            event.direction,
            event.reaction_index,
            event.free_smiles,
            event.protected_smiles,
        ),
    )


def build_ground_truth_pairs(
    events: list[ProtectionEvent],
    max_pairs_per_group: int,
) -> list[GroundTruthPair]:
    by_key: dict[tuple[str, str], dict[str, list[ProtectionEvent]]] = defaultdict(
        lambda: {"install": [], "remove": []}
    )
    for event in events:
        by_key[(event.pg_label, event.scaffold_key)][event.direction].append(event)

    spec_by_label = {spec.label: spec for spec in PROTECTING_GROUPS}
    pairs: list[GroundTruthPair] = []

    for (pg_label, scaffold_key), grouped in sorted(by_key.items()):
        installs = sorted(grouped["install"], key=lambda event: event.reaction_index)
        removals = sorted(grouped["remove"], key=lambda event: event.reaction_index)
        selected_pair: GroundTruthPair | None = None
        for install in installs:
            for remove in removals:
                if install.reaction_index >= remove.reaction_index:
                    continue
                selected_pair = GroundTruthPair(
                    install_index=install.reaction_index,
                    remove_index=remove.reaction_index,
                    pg_label=pg_label,
                    functional_group=spec_by_label[pg_label].functional_group,
                    scaffold_key=scaffold_key,
                    install_free_smiles=install.free_smiles,
                    install_protected_smiles=install.protected_smiles,
                    remove_protected_smiles=remove.protected_smiles,
                    remove_free_smiles=remove.free_smiles,
                )
                break
            if selected_pair is not None:
                break
        if selected_pair is not None:
            pairs.append(selected_pair)

    pairs = sorted(pairs, key=lambda pair: (pair.pg_label, pair.install_index, pair.remove_index))
    if max_pairs_per_group <= 0:
        return pairs

    capped_pairs: list[GroundTruthPair] = []
    per_group_counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        if per_group_counts[pair.pg_label] >= max_pairs_per_group:
            continue
        capped_pairs.append(pair)
        per_group_counts[pair.pg_label] += 1
    return capped_pairs


def parse_response(response: str) -> set[tuple[int, int]]:
    text = response.strip()
    if not text or text.replace(" ", "") == "-1":
        return set()

    pairs: set[tuple[int, int]] = set()
    for line in text.splitlines():
        nums = re.findall(r"\d+", line)
        if len(nums) < 2:
            continue
        install_idx, remove_idx = int(nums[0]), int(nums[1])
        if install_idx != remove_idx:
            pairs.add((install_idx, remove_idx))
    return pairs


def precision_recall_f1(
    predicted: set[tuple[int, int]], ground_truth: set[tuple[int, int]]
) -> tuple[float, float, float]:
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def summarize_gt_pair(pair: GroundTruthPair) -> dict[str, object]:
    return {
        "install_index": pair.install_index,
        "remove_index": pair.remove_index,
        "pg_label": pair.pg_label,
        "functional_group": pair.functional_group,
        "scaffold_key": pair.scaffold_key,
        "install_free_smiles": pair.install_free_smiles,
        "install_protected_smiles": pair.install_protected_smiles,
        "remove_protected_smiles": pair.remove_protected_smiles,
        "remove_free_smiles": pair.remove_free_smiles,
    }


def count_by_label(items: list[ProtectionEvent] | list[GroundTruthPair]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item.pg_label] += 1
    return dict(sorted(counts.items()))


def build_question(spec: ProtectingGroupSpec, max_pairs: int) -> str:
    pair_limit_instruction = (
        "Return all valid pairs for this protecting group."
        if max_pairs <= 0
        else (
            f"Return at most {max_pairs} pairs for this protecting group, choosing the "
            "earliest pairs after sorting by install index and then removal index."
        )
    )
    return f"""
    Context: You are given a large set of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Each side may contain multiple molecules separated by dots (.).

    Task:
    Find protecting-group install/remove reaction pairs (A, B) in the dataset for this protecting group:
    - label: {spec.label}
    - aliases: {", ".join(spec.aliases)}
    - functional group protected: {spec.functional_group}
    - description: {spec.description}

    A valid pair has:
    - A installs {spec.label} onto a free {spec.functional_group} functional group.
    - B later removes {spec.label} from the same underlying scaffold.
    - The install reaction index must be smaller than the removal reaction index.

    Grounding rules:
    - Use RDKit for canonicalization and SMARTS/substructure checks.
    - Split multi-component sides on dots (.) and canonicalize each component independently.
    - Treat two molecules as the same scaffold if removing/ignoring the protecting-group atoms
      from the protected molecule gives the same canonical free-scaffold SMILES.
    - Do not rely only on graph connectivity through exact reaction products and reactants;
      the scaffold comparison is chemical/substructure based.
    - Ignore molecules with fewer than {MIN_HEAVY_ATOMS} or more than {MAX_HEAVY_ATOMS} heavy atoms.
    - {pair_limit_instruction}
    - If several valid pairs share the same scaffold and protecting group, prefer the earliest
      install reaction and the earliest later removal reaction.
    - Skip malformed reactions or molecules that RDKit cannot parse.
    - DO NOT assume/simulate output of code. Wait for code execution and only then return.
    - DO NOT USE `FINAL` for writing a thought/comment.

    Output format:
    - Return ONLY protecting-group pairs, one pair per line.
    - Each line must be "install_index,remove_index" (e.g., 60483,60620).
    - No labels, explanations, quotes, JSON, markdown, or other punctuation.

    If no pair exists for {spec.label}, return -1.
    """


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task15",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RLM task 15 — protecting-group install/remove pair mining."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for backend (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DATASET_PATH,
        help=f"Path to USPTO reaction dataset (default: {DATASET_PATH}).",
    )
    parser.add_argument(
        "--max-pairs-per-group",
        type=int,
        default=MAX_PAIRS_PER_GROUP,
        help=(
            "Maximum ground-truth pairs per protecting group; use 0 for all "
            f"pairs (default: {MAX_PAIRS_PER_GROUP})."
        ),
    )
    parser.add_argument(
        "--mine-only",
        action="store_true",
        help="Only compute and print RDKit ground truth; skip model evaluation.",
    )
    return parser.parse_args()


def main(
    model_name: str,
    dataset_path: str,
    max_pairs_per_group: int,
    mine_only: bool,
) -> None:
    if max_pairs_per_group < 0:
        raise ValueError("--max-pairs-per-group must be non-negative.")

    records = parse_dataset(dataset_path)
    print(f"Loaded {len(records)} parsable reactions from {dataset_path}")

    events = mine_protection_events(records)
    print(f"Mined {len(events)} protecting-group install/remove events.")
    print(f"Event counts by protecting group: {json.dumps(count_by_label(events), sort_keys=True)}")

    gt_pairs = build_ground_truth_pairs(
        events=events,
        max_pairs_per_group=max_pairs_per_group,
    )
    gt_pairs_by_label: dict[str, list[GroundTruthPair]] = {
        spec.label: [] for spec in PROTECTING_GROUPS
    }
    for pair in gt_pairs:
        gt_pairs_by_label[pair.pg_label].append(pair)
    evaluated_specs = [
        spec for spec in PROTECTING_GROUPS
        if gt_pairs_by_label[spec.label]
    ]
    skipped_specs = [
        spec.label for spec in PROTECTING_GROUPS
        if not gt_pairs_by_label[spec.label]
    ]

    print(f"Pair counts by protecting group: {json.dumps(count_by_label(gt_pairs), sort_keys=True)}")
    if skipped_specs:
        print(f"Skipping protecting groups with empty ground truth: {json.dumps(skipped_specs)}")
    print(
        "Ground truth protecting-group pairs: "
        + json.dumps([summarize_gt_pair(pair) for pair in gt_pairs], separators=(",", ":"))
    )

    if mine_only:
        return
    if not evaluated_specs:
        raise ValueError("No protecting-group questions have non-empty ground truth.")

    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
    context = "\n".join(f"{i} {line}" for i, line in enumerate(raw_lines))

    run = None
    if wandb is None:
        print("wandb not installed; continuing without experiment logging.")
    else:
        run = wandb.init(
            project="RLMs-Task15",
            config={
                "MODEL_NAME": model_name,
                "backend": BACKEND,
                "model_name": model_name,
                "dataset_path": dataset_path,
                "max_pairs_per_group": max_pairs_per_group,
                "min_heavy_atoms": MIN_HEAVY_ATOMS,
                "max_heavy_atoms": MAX_HEAVY_ATOMS,
                "num_questions": len(evaluated_specs),
                "num_ground_truth_pairs": len(gt_pairs),
                "ground_truth_pair_counts_by_group": count_by_label(gt_pairs),
                "protecting_groups": [spec.label for spec in evaluated_specs],
                "skipped_empty_ground_truth_groups": skipped_specs,
                "seed": SEED,
                "rlm_init_kwargs": rlm_init_kwargs,
                "task_description": "Per-protecting-group install/remove pair discovery via SMARTS-normalized scaffolds.",
            },
        )
        wandb.define_metric("sample_iteration")
        wandb.define_metric("sample/*", step_metric="sample_iteration")

    macro_exact_set_match = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, spec in enumerate(evaluated_specs):
        group_gt_pairs = gt_pairs_by_label[spec.label]
        gt_set = {(pair.install_index, pair.remove_index) for pair in group_gt_pairs}
        question = build_question(spec=spec, max_pairs=max_pairs_per_group)

        print(
            f"\nQuestion {i + 1}/{len(evaluated_specs)}: "
            f"pg_label={spec.label}, gt_pair_count={len(gt_set)}"
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(evaluated_specs),
                "task": "protecting_group_pairs_by_group",
                "pg_label": spec.label,
                "functional_group": spec.functional_group,
                "num_ground_truth_pairs": len(gt_set),
            },
            tags=["run_rlms", "sample", "task15_PROTECTING_GROUP_PAIRS"],
        ):
            completion = rlm.completion(prompt=context, root_prompt=question)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        predicted = parse_response(response)
        precision, recall, f1 = precision_recall_f1(predicted=predicted, ground_truth=gt_set)
        exact_set_match = float(predicted == gt_set)
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        macro_exact_set_match += exact_set_match
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        print(f"Response [sample={i}]: {response[:1000]}{'...' if len(response) > 1000 else ''}")
        print(f"Predicted pairs [sample={i}]: {sorted(predicted)}")
        print(f"Ground-truth pairs [sample={i}]: {sorted(gt_set)}")
        print(
            f"Metrics [sample={i}] -> exact_set_match={exact_set_match:.0f} "
            f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
        )

        if wandb is not None:
            for metric in iteration_metrics:
                wandb.log(
                    {
                        "sample_iteration": metric["iteration"],
                        f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                        f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                        f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                    }
                )

        if run is not None and wandb is not None:
            last_metric = iteration_metrics[-1] if iteration_metrics else {}
            wandb.log(
                {
                    "sample_idx": i,
                    f"sample/{i}/pg_label": spec.label,
                    f"sample/{i}/functional_group": spec.functional_group,
                    f"sample/{i}/ground_truth": json.dumps(
                        [summarize_gt_pair(pair) for pair in group_gt_pairs],
                        separators=(",", ":"),
                    ),
                    f"sample/{i}/pred_pairs": json.dumps(sorted(predicted), separators=(",", ":")),
                    f"sample/{i}/response_raw": response,
                    f"sample/{i}/exact_set_match": exact_set_match,
                    f"sample/{i}/precision": precision,
                    f"sample/{i}/recall": recall,
                    f"sample/{i}/f1": f1,
                    f"sample/{i}/completion_root_prompt": question,
                    f"sample/{i}/completion_prompt_char_count": len(context),
                    f"sample/{i}/final_total_input_tokens": last_metric.get("total_input_tokens", 0),
                    f"sample/{i}/final_total_output_tokens": last_metric.get("total_output_tokens", 0),
                    f"sample/{i}/final_total_tokens": last_metric.get("total_tokens", 0),
                    f"sample/{i}/iterations": len(iteration_metrics),
                    **(
                        {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                        if sample_cost_usd is not None
                        else {}
                    ),
                }
            )

    total = len(evaluated_specs)
    macro_exact_set_match = macro_exact_set_match / total if total else 0.0
    macro_precision = macro_precision / total if total else 0.0
    macro_recall = macro_recall / total if total else 0.0
    macro_f1 = macro_f1 / total if total else 0.0

    print(f"\n{'=' * 60}")
    print(f"Protecting-group questions evaluated: {total}")
    print(f"Macro exact-set match: {macro_exact_set_match:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    if run is not None and wandb is not None:
        run.summary["questions_evaluated"] = total
        run.summary["ground_truth_pair_count"] = len(gt_pairs)
        run.summary["macro_exact_set_match"] = macro_exact_set_match
        run.summary["macro_precision"] = macro_precision
        run.summary["macro_recall"] = macro_recall
        run.summary["macro_f1"] = macro_f1
        run.summary["samples_with_cost"] = samples_with_cost
        if samples_with_cost > 0:
            run.summary["total_cost_usd"] = total_cost_usd
            run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        max_pairs_per_group=args.max_pairs_per_group,
        mine_only=args.mine_only,
    )
