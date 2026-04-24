import argparse
import json
import os
import re
import uuid
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rlm import RLM
from rlm.tracing import init_tracing, using_tracing_attributes

try:
    import wandb
except ImportError:
    wandb = None

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
CURATED_PAIRS_PATH = "/home/bhagavan/rlms/rlm/tier4/task13_curated_pairs.json"
CANDIDATE_OUTPUT_PATH = "/home/bhagavan/rlms/rlm/tier4/task13_candidate_pairs.json"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
SEED = 42

MIN_SUBSTRATE_SIMILARITY = 0.45
MAX_REAGENT_JACCARD = 0.45
MAX_CANDIDATES_PER_SIGNATURE = 40
TOP_CANDIDATES_TO_EXPORT = 200

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}


SENSITIVE_GROUP_SMARTS: dict[str, list[str]] = {
    "silyl_ether": ["[OX2][Si]"],
    "acetal_or_ketal": ["[CX4H0](-[OX2])(-[OX2])"],
    "Boc_carbamate": ["CC(C)(C)OC(=O)N", "CC(C)(C)OC(=O)[N]"],
    "benzyl_protecting_group": ["[O,N]Cc1ccccc1"],
    "allyl_protecting_group": ["[O,N]CC=C"],
    "aldehyde_sensitive_site": ["[CX3H1](=O)[#6]"],
    "free_primary_alcohol": ["[CH2][OX2H]"],
    "free_secondary_alcohol": ["[CH]([#6])[OX2H]"],
    "free_amine": ["[NX3;H1,H2]"],
    "alkene": ["C=C"],
    "alkyne": ["C#C"],
}


COMMON_REAGENT_PATTERNS: dict[str, list[str]] = {
    "PCC": [r"\bPCC\b", r"pyridinium chlorochromate"],
    "Jones": [r"\bjones\b", r"cro3", r"h2so4", r"chromic acid"],
    "DessMartin": [r"dmp", r"dess[- ]?martin"],
    "Swern": [r"swern", r"oxalyl chloride", r"dmso"],
    "PDC": [r"\bPDC\b", r"pyridinium dichromate"],
    "MnO2": [r"\bMnO2\b", r"manganese dioxide"],
    "NaBH4": [r"\bNaBH4\b", r"sodium borohydride"],
    "LiAlH4": [r"\bLiAlH4\b", r"lialh4", r"lithium aluminium hydride"],
}


@dataclass
class ReactionRecord:
    index: int
    raw: str
    reactants: list[str]
    reagents: list[str]
    products: list[str]
    main_substrate: Optional[str]
    transformation_signature: tuple[int, ...]


@dataclass
class CandidatePair:
    idx_a: int
    idx_b: int
    substrate_similarity: float
    reagent_jaccard: float
    score: float
    signature: tuple[int, ...]


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


def split_reaction_line(indexed_line: str) -> tuple[int, str, str, str]:
    idx_str, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    if len(parts) == 2:
        reactants, products = parts
        reagents = ""
    elif len(parts) == 3:
        reactants, reagents, products = parts
    else:
        raise ValueError(f"Malformed reaction line: {indexed_line[:80]}")
    return int(idx_str), reactants.strip(), reagents.strip(), products.strip()


def pick_main_substrate(reactants: list[str]) -> Optional[str]:
    best_smi = None
    best_size = -1
    for smi in reactants:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        heavy = mol.GetNumHeavyAtoms()
        if heavy > best_size:
            best_size = heavy
            best_smi = smi
    return best_smi


def morgan_count_fingerprint(mol: Chem.Mol) -> dict[int, int]:
    fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=2)
    return dict(fp.GetNonzeroElements())


def reaction_signature(reactants: list[str], products: list[str], top_k: int = 24) -> tuple[int, ...]:
    diff: dict[int, int] = {}
    for smi in products:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for bit, count in morgan_count_fingerprint(mol).items():
            diff[bit] = diff.get(bit, 0) + count
    for smi in reactants:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for bit, count in morgan_count_fingerprint(mol).items():
            diff[bit] = diff.get(bit, 0) - count
    if not diff:
        return tuple()
    ranked = sorted(diff.items(), key=lambda kv: (abs(kv[1]), kv[0]), reverse=True)[:top_k]
    signature: list[int] = []
    for bit, count in ranked:
        if count > 0:
            signature.extend([bit] * min(count, 3))
        elif count < 0:
            signature.extend([-(bit + 1)] * min(-count, 3))
    return tuple(signature)


def bitvect_for_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def reagent_tokens(reagents: list[str]) -> set[str]:
    tokens: set[str] = set()
    for smi in reagents:
        token = smi.strip()
        if not token:
            continue
        tokens.add(token)
        lowered = token.lower()
        for label, patterns in COMMON_REAGENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, lowered):
                    tokens.add(f"label:{label}")
                    break
    return tokens


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def extract_sensitive_groups(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    found: list[str] = []
    for label, smarts_list in SENSITIVE_GROUP_SMARTS.items():
        for smarts in smarts_list:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            if mol.HasSubstructMatch(patt):
                found.append(label)
                break
    return sorted(found)


def parse_dataset(dataset_path: str) -> dict[int, ReactionRecord]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]

    records: dict[int, ReactionRecord] = {}
    for i, raw in enumerate(raw_lines):
        indexed_line = f"{i} {raw}"
        try:
            idx, reactants_raw, reagents_raw, products_raw = split_reaction_line(indexed_line)
        except Exception:
            continue
        reactants = canonicalize_components(reactants_raw)
        products = canonicalize_components(products_raw)
        reagents = canonicalize_components(reagents_raw) if reagents_raw else []
        if not reactants or not products:
            continue
        main_substrate = pick_main_substrate(reactants)
        signature = reaction_signature(reactants, products)
        records[idx] = ReactionRecord(
            index=idx,
            raw=raw,
            reactants=reactants,
            reagents=reagents,
            products=products,
            main_substrate=main_substrate,
            transformation_signature=signature,
        )
    return records


def mine_candidate_pairs(records: dict[int, ReactionRecord]) -> list[CandidatePair]:
    by_signature: dict[tuple[int, ...], list[ReactionRecord]] = {}
    for rec in records.values():
        if not rec.transformation_signature or rec.main_substrate is None:
            continue
        by_signature.setdefault(rec.transformation_signature, []).append(rec)

    candidates: list[CandidatePair] = []
    for signature, cluster in by_signature.items():
        if len(cluster) < 2:
            continue
        if len(cluster) > MAX_CANDIDATES_PER_SIGNATURE:
            cluster = cluster[:MAX_CANDIDATES_PER_SIGNATURE]
        for a, b in combinations(cluster, 2):
            fp_a = bitvect_for_smiles(a.main_substrate) if a.main_substrate else None
            fp_b = bitvect_for_smiles(b.main_substrate) if b.main_substrate else None
            if fp_a is None or fp_b is None:
                continue
            substrate_sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
            if substrate_sim < MIN_SUBSTRATE_SIMILARITY:
                continue
            tokens_a = reagent_tokens(a.reagents)
            tokens_b = reagent_tokens(b.reagents)
            if not tokens_a or not tokens_b:
                continue
            reagent_sim = jaccard(tokens_a, tokens_b)
            if reagent_sim > MAX_REAGENT_JACCARD:
                continue
            score = substrate_sim * (1.0 - reagent_sim)
            candidates.append(
                CandidatePair(
                    idx_a=a.index,
                    idx_b=b.index,
                    substrate_similarity=substrate_sim,
                    reagent_jaccard=reagent_sim,
                    score=score,
                    signature=signature,
                )
            )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def export_candidates(
    candidates: list[CandidatePair],
    records: dict[int, ReactionRecord],
    output_path: str,
    top_n: int,
) -> None:
    exported: list[dict[str, object]] = []
    for cand in candidates[:top_n]:
        a = records[cand.idx_a]
        b = records[cand.idx_b]
        substrate_a = a.main_substrate or ""
        substrate_b = b.main_substrate or ""
        exported.append(
            {
                "idx_a": cand.idx_a,
                "idx_b": cand.idx_b,
                "reaction_a": a.raw,
                "reaction_b": b.raw,
                "main_substrate_a": substrate_a,
                "main_substrate_b": substrate_b,
                "sensitive_groups_a": extract_sensitive_groups(substrate_a) if substrate_a else [],
                "sensitive_groups_b": extract_sensitive_groups(substrate_b) if substrate_b else [],
                "reagents_a": a.reagents,
                "reagents_b": b.reagents,
                "substrate_similarity": round(cand.substrate_similarity, 4),
                "reagent_jaccard": round(cand.reagent_jaccard, 4),
                "mining_score": round(cand.score, 4),
            }
        )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(exported, f, indent=2)
    print(f"Exported {len(exported)} candidate pairs to {output_path}")


def build_question(
    idx_a: int,
    idx_b: int,
    reaction_a: str,
    reaction_b: str,
    hint_groups_a: list[str],
    hint_groups_b: list[str],
) -> str:
    hint_a = ", ".join(hint_groups_a) if hint_groups_a else "none_detected"
    hint_b = ", ".join(hint_groups_b) if hint_groups_b else "none_detected"
    return f"""
    Context: You are given two indexed reactions from the same dataset.
    Both appear to perform an analogous transformation on similar substrates, but with different reagents.

    Reaction A:
    - index: {idx_a}
    - reaction: {reaction_a}

    Reaction B:
    - index: {idx_b}
    - reaction: {reaction_b}

    Automatically detected potentially sensitive groups (heuristic only):
    - substrate_A_groups: {hint_a}
    - substrate_B_groups: {hint_b}

    Task:
    Explain why reagent choice differs between reaction A and B.
    Identify a concrete substrate structural feature that can motivate the reagent change
    (e.g., acid-labile protecting group, oxidation-sensitive handle, chemoselectivity issue).

    Guidance:
    - Keep chemistry grounded in groups present in the actual substrates.
    - Explicitly connect "feature -> condition compatibility -> reagent preference".
    - Mention both reactions and contrast them.
    - If no confident feature is visible, say uncertainty explicitly.
    - DO NOT assume/simulate output of code. Wait for the code execution before final answer.
    - DO NOT USE `FINAL` for writing a thought/comment.

    Output format (strict JSON on one line):
    {{"feature_label":"<short_label>","feature_in":"A|B|both|uncertain","reasoning":"<1-3 sentences>","confidence":"high|medium|low"}}
    """


def parse_response_fields(response: str) -> tuple[str, str, str, str]:
    feature_label = "uncertain"
    feature_in = "uncertain"
    confidence = "low"
    reasoning = response.strip()
    try:
        parsed = json.loads(response.strip())
        if isinstance(parsed, dict):
            feature_label = str(parsed.get("feature_label", "uncertain")).strip() or "uncertain"
            feature_in = str(parsed.get("feature_in", "uncertain")).strip() or "uncertain"
            confidence = str(parsed.get("confidence", "low")).strip() or "low"
            reasoning = str(parsed.get("reasoning", "")).strip() or reasoning
    except Exception:
        pass
    return feature_label.lower(), feature_in.lower(), reasoning, confidence.lower()


def score_example(
    pred_feature_label: str,
    pred_feature_in: str,
    pred_reasoning: str,
    gt_feature_labels: list[str],
    gt_feature_in: str,
    gt_keywords: list[str],
) -> dict[str, float]:
    label_hit = 1.0 if pred_feature_label in {x.lower() for x in gt_feature_labels} else 0.0
    feature_in_hit = 1.0 if pred_feature_in == gt_feature_in.lower() else 0.0
    reasoning_l = pred_reasoning.lower()
    kw_hits = sum(1 for kw in gt_keywords if kw.lower() in reasoning_l)
    kw_score = kw_hits / len(gt_keywords) if gt_keywords else 0.0
    total = 0.5 * label_hit + 0.3 * feature_in_hit + 0.2 * kw_score
    return {
        "label_hit": label_hit,
        "feature_in_hit": feature_in_hit,
        "keyword_score": kw_score,
        "total_score": total,
    }


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task13",
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
        description="Run RLM task 13 — analogous reaction pairs with reagent-choice reasoning."
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
        "--curated-pairs-path",
        type=str,
        default=CURATED_PAIRS_PATH,
        help=f"Path to hand-curated JSON pairs (default: {CURATED_PAIRS_PATH}).",
    )
    parser.add_argument(
        "--candidate-output-path",
        type=str,
        default=CANDIDATE_OUTPUT_PATH,
        help=f"Path to save mined candidates JSON (default: {CANDIDATE_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=TOP_CANDIDATES_TO_EXPORT,
        help="Number of mined candidates to export for curation.",
    )
    parser.add_argument(
        "--mine-only",
        action="store_true",
        help="Only mine candidate pairs and write JSON; skip model evaluation.",
    )
    return parser.parse_args()


def load_curated_pairs(path: str) -> list[dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Curated pairs file not found: {path}. "
            "Run with --mine-only first, then hand-curate 5-8 examples."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Curated pairs JSON must be a list.")
    return data


def main(
    model_name: str,
    dataset_path: str,
    curated_pairs_path: str,
    candidate_output_path: str,
    top_candidates: int,
    mine_only: bool,
) -> None:
    records = parse_dataset(dataset_path)
    print(f"Loaded {len(records)} parsable reactions from {dataset_path}")

    candidates = mine_candidate_pairs(records)
    print(f"Mined {len(candidates)} candidate analogous-reaction pairs.")
    export_candidates(
        candidates=candidates,
        records=records,
        output_path=candidate_output_path,
        top_n=top_candidates,
    )
    if mine_only:
        return

    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    curated_pairs = load_curated_pairs(curated_pairs_path)
    print(f"Loaded {len(curated_pairs)} curated pairs from {curated_pairs_path}")

    run = None
    if wandb is None:
        print("wandb not installed; continuing without experiment logging.")
    else:
        run = wandb.init(
            project="RLMs-Task13",
            config={
                "MODEL_NAME": model_name,
                "backend": BACKEND,
                "model_name": model_name,
                "dataset_path": dataset_path,
                "curated_pairs_path": curated_pairs_path,
                "num_questions": len(curated_pairs),
                "seed": SEED,
                "min_substrate_similarity": MIN_SUBSTRATE_SIMILARITY,
                "max_reagent_jaccard": MAX_REAGENT_JACCARD,
                "rlm_init_kwargs": rlm_init_kwargs,
                "task_description": "Explain structural reason for reagent choice in analogous reaction pairs.",
            },
        )
        wandb.define_metric("sample_iteration")
        wandb.define_metric("sample/*", step_metric="sample_iteration")

    macro_total_score = 0.0
    macro_label_hit = 0.0
    macro_feature_in_hit = 0.0
    macro_keyword_score = 0.0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, item in enumerate(curated_pairs):
        idx_a = int(item["idx_a"])
        idx_b = int(item["idx_b"])
        gt_feature_labels = [str(x) for x in item.get("gt_feature_labels", [])]
        gt_feature_in = str(item.get("gt_feature_in", "uncertain"))
        gt_keywords = [str(x) for x in item.get("gt_keywords", [])]

        rec_a = records.get(idx_a)
        rec_b = records.get(idx_b)
        if rec_a is None or rec_b is None:
            print(f"Skipping pair {i} because reaction index missing in parsed dataset: {idx_a}, {idx_b}")
            continue

        hint_groups_a = extract_sensitive_groups(rec_a.main_substrate) if rec_a.main_substrate else []
        hint_groups_b = extract_sensitive_groups(rec_b.main_substrate) if rec_b.main_substrate else []

        question = build_question(
            idx_a=idx_a,
            idx_b=idx_b,
            reaction_a=rec_a.raw,
            reaction_b=rec_b.raw,
            hint_groups_a=hint_groups_a,
            hint_groups_b=hint_groups_b,
        )

        print(
            f"\nQuestion {i + 1}/{len(curated_pairs)}: "
            f"idx_a={idx_a}, idx_b={idx_b}, gt_feature_labels={gt_feature_labels}"
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(curated_pairs),
                "task": "reagent_choice_reasoning",
                "idx_a": idx_a,
                "idx_b": idx_b,
                "gt_feature_labels": gt_feature_labels,
                "gt_feature_in": gt_feature_in,
            },
            tags=["run_rlms", "sample", "task13_REAGENT_CHOICE_REASONING"],
        ):
            completion = rlm.completion(prompt="", root_prompt=question)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        pred_feature_label, pred_feature_in, pred_reasoning, pred_confidence = parse_response_fields(response)
        scores = score_example(
            pred_feature_label=pred_feature_label,
            pred_feature_in=pred_feature_in,
            pred_reasoning=pred_reasoning,
            gt_feature_labels=gt_feature_labels,
            gt_feature_in=gt_feature_in,
            gt_keywords=gt_keywords,
        )
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        macro_total_score += scores["total_score"]
        macro_label_hit += scores["label_hit"]
        macro_feature_in_hit += scores["feature_in_hit"]
        macro_keyword_score += scores["keyword_score"]

        print(f"Response [pair={i}]: {response[:500]}{'…' if len(response) > 500 else ''}")
        print(
            f"Scores [pair={i}] -> total={scores['total_score']:.4f} "
            f"label_hit={scores['label_hit']:.4f} "
            f"feature_in_hit={scores['feature_in_hit']:.4f} "
            f"keyword_score={scores['keyword_score']:.4f}"
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

        if wandb is not None and iteration_metrics:
            last_metric = iteration_metrics[-1]
            wandb.log(
                {
                    "sample_idx": i,
                    f"sample/{i}/idx_a": idx_a,
                    f"sample/{i}/idx_b": idx_b,
                    f"sample/{i}/ground_truth_feature_labels": ",".join(gt_feature_labels),
                    f"sample/{i}/ground_truth_feature_in": gt_feature_in,
                    f"sample/{i}/ground_truth_keywords": ",".join(gt_keywords),
                    f"sample/{i}/pred_feature_label": pred_feature_label,
                    f"sample/{i}/pred_feature_in": pred_feature_in,
                    f"sample/{i}/pred_confidence": pred_confidence,
                    f"sample/{i}/pred_reasoning": pred_reasoning,
                    f"sample/{i}/response_raw": response,
                    f"sample/{i}/label_hit": scores["label_hit"],
                    f"sample/{i}/feature_in_hit": scores["feature_in_hit"],
                    f"sample/{i}/keyword_score": scores["keyword_score"],
                    f"sample/{i}/total_score": scores["total_score"],
                    f"sample/{i}/completion_root_prompt": question,
                    f"sample/{i}/final_total_input_tokens": last_metric["total_input_tokens"],
                    f"sample/{i}/final_total_output_tokens": last_metric["total_output_tokens"],
                    f"sample/{i}/final_total_tokens": last_metric["total_tokens"],
                    f"sample/{i}/iterations": len(iteration_metrics),
                    **(
                        {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                        if sample_cost_usd is not None
                        else {}
                    ),
                }
            )

    total = len(curated_pairs)
    macro_total_score = (macro_total_score / total) if total else 0.0
    macro_label_hit = (macro_label_hit / total) if total else 0.0
    macro_feature_in_hit = (macro_feature_in_hit / total) if total else 0.0
    macro_keyword_score = (macro_keyword_score / total) if total else 0.0

    print(f"\n{'=' * 60}")
    print(f"Pairs evaluated: {total}")
    print(f"Macro total score: {macro_total_score:.4f}")
    print(f"Macro feature-label hit: {macro_label_hit:.4f}")
    print(f"Macro feature-location hit: {macro_feature_in_hit:.4f}")
    print(f"Macro keyword score: {macro_keyword_score:.4f}")

    if run is not None and wandb is not None:
        run.summary["pairs_evaluated"] = total
        run.summary["macro_total_score"] = macro_total_score
        run.summary["macro_feature_label_hit"] = macro_label_hit
        run.summary["macro_feature_location_hit"] = macro_feature_in_hit
        run.summary["macro_keyword_score"] = macro_keyword_score
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
        curated_pairs_path=args.curated_pairs_path,
        candidate_output_path=args.candidate_output_path,
        top_candidates=args.top_candidates,
        mine_only=args.mine_only,
    )
