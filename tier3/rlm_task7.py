import argparse
import os
import re
import uuid
from typing import Optional

import wandb
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rlm import RLM
from rlm.tracing import init_tracing, using_tracing_attributes

os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = False

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}


def parse_indices(response: str) -> Optional[list[int]]:
    response = response.strip()
    if not response:
        return []
    normalized = response.replace(" ", "")
    if normalized == "-1":
        return []
    matches = re.findall(r"\d+", response)
    if not matches:
        return []
    indices: list[int] = []
    seen: set[int] = set()
    for match in matches:
        idx = int(match)
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


TO_ALCOHOL_SMIRKS: dict[str, str] = {
   "grignard_ketone_to_tertiary_alcohol": "[#6;+0:1]-[Mg]-[Br,I,Cl].[*:2]-[C;H0;D3;+0:3](=[O;H0;D1;+0:4])-[#6;+0:5]>>[*:2]-[C;H0;D4;+0:3](-[O;H1;D1;+0:4])(-[#6;+0:5])-[#6;+0:1]",
}

TO_ALCOHOL_LABELS: dict[str, str] = {
    "grignard_ketone_to_tertiary_alcohol": "Grignard from ketone to alcohol",
}

TO_ALCOHOL_DESCRIPTIONS: dict[str, str] = {
    "grignard_ketone_to_tertiary_alcohol": "A Grignard reaction in which an organomagnesium halide adds to a ketone to form a tertiary alcohol. The carbon nucleophile of the Grignard reagent, bonded to magnesium which in turn bears a halide (bromide, iodide, or chloride), attacks the electrophilic carbonyl carbon of the ketone. The carbonyl carbon transitions from trigonal planar (three substituents) to tetrahedral (four substituents), gaining a new carbon-carbon bond to the Grignard carbon. The carbonyl oxygen is reduced from a double bond to a single bond, gaining a hydrogen to become a hydroxyl group in the product. Since the ketone carbonyl carbon already bears two carbon substituents, the addition of the Grignard carbon yields a tertiary alcohol with no hydrogens on the central carbon. This reaction is one of the most important carbon-carbon bond-forming reactions in organic synthesis, enabling the construction of complex molecular architectures from simpler precursors."
}


def parse_reaction_sides(indexed_line: str) -> tuple[str, str]:
    _, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    reactant_side = parts[0].strip()
    product_side = parts[-1].strip()
    return reactant_side, product_side


def build_reaction_query(smarts: str) -> rdChemReactions.ChemicalReaction:
    query = rdChemReactions.ReactionFromSmarts(smarts)
    if query is None:
        raise ValueError(f"Failed to parse reaction SMARTS: {smarts}")
    return query


def reaction_matches(indexed_line, query_reaction):
    reactants, products = parse_reaction_sides(indexed_line)
    
    r_mols = [Chem.MolFromSmiles(s) for s in reactants.split(".")]
    p_mols = [Chem.MolFromSmiles(s) for s in products.split(".")]
    
    if any(mol is None for mol in r_mols):
        return False
    if any(mol is None for mol in p_mols):
        return False
    
    # Check reactant patterns match some reactant molecule
    for q in query_reaction.GetReactants():
        if not any(m.HasSubstructMatch(q) for m in r_mols if m):
            return False
    
    # Check product patterns match some product molecule
    for q in query_reaction.GetProducts():
        if not any(m.HasSubstructMatch(q) for m in p_mols if m):
            return False
    
    return True

def ground_truth_indices(
    lines: list[str], query_reaction: rdChemReactions.ChemicalReaction
) -> list[int]:
    matching_indices: list[int] = []
    for line in lines:
        idx_str, _ = line.split(" ", 1)
        idx = int(idx_str)
        if reaction_matches(line, query_reaction):
            matching_indices.append(idx)
    return matching_indices


def precision_recall_f1(
    predicted_indices: set[int], ground_truth_indices: set[int]
) -> tuple[float, float, float]:
    tp = len(predicted_indices & ground_truth_indices)
    fp = len(predicted_indices - ground_truth_indices)
    fn = len(ground_truth_indices - predicted_indices)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def build_question(reaction_label: str, reaction_description: str) -> str:
    return f"""
    Context: You are given a large string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Task:
    Count how many reactions are of the following type:
    - {reaction_label}
    
    Here is a detailed description of the reaction type:
    - {reaction_description}

    Guidance:
    - Define a single Reaction SMIRKS pattern encoding the full transformation (reactants >> products) with atom mapping to classify reactions. DO NOT match functional groups independently on reactants and products using individual SMARTS patterns.
    - You may reason about a few candidate SMIRKS, but commit to exactly one for the final answer. DO NOT aggregate counts from multiple patterns.
    - Use RdKit for all analysis and counting.
    - DO NOT count other reaction types for this question.
    - Ignore reagents (middle field).
    - Handle multi-component sides separated by dots (.).
    - Skip malformed reactions.
    - Skip reactions that errors out while matching the SMIRKS pattern with RDKit.

    Output format:
    - Return ONLY the matching reaction INDICES.
    - Format must be a comma-separated list of integers in ascending order (e.g., 3,8,21).
    - No other text, quotes, labels, punctuation, or formatting.
    - If no matching reaction exists, return -1.
    """


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task7",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLM task 7 prompt-only evaluation.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for backend (default: {MODEL_NAME}).",
    )
    return parser.parse_args()


def main(model_name: str) -> None:
    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]

    context = "\n".join(lines)
    donor_keys = list(TO_ALCOHOL_SMIRKS.keys())

    gt_indices_by_donor: dict[str, list[int]] = {}
    for donor_key in donor_keys:
        smarts = TO_ALCOHOL_SMIRKS[donor_key]
        query_reaction = build_reaction_query(smarts)
        gt_indices = ground_truth_indices(lines, query_reaction)
        gt_indices_by_donor[donor_key] = gt_indices
        print(f"Ground truth [{donor_key}] count={len(gt_indices)} ({smarts})")

    run = wandb.init(
        project="RLMs-Task7",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "num_questions": len(donor_keys),
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Count from ketone to alcohol reactions.",
            "TO_ALCOHOL_SMIRKS": TO_ALCOHOL_SMIRKS,
            "ground_truth_indices_by_donor": gt_indices_by_donor,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    exact_match_count = 0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, donor_key in enumerate(donor_keys):
        reaction_label = TO_ALCOHOL_LABELS[donor_key]
        reaction_description = TO_ALCOHOL_DESCRIPTIONS[donor_key]
        donor_smarts = TO_ALCOHOL_SMIRKS[donor_key]
        question = build_question(reaction_label=reaction_label, reaction_description=reaction_description)
        gt_indices = gt_indices_by_donor[donor_key]
        gt_set = set(gt_indices)
        completion_kwargs = {"prompt": context, "root_prompt": question}

        print(f"Question {i + 1}/{len(donor_keys)} donor={donor_key}")

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(donor_keys),
                "task": "TO_ALCOHOL_count",
                "TO_ALCOHOL_donor_key": donor_key,
                "TO_ALCOHOL_SMIRKS": donor_smarts,
            },
            tags=["run_rlms", "sample", "task7_TO_ALCOHOL_by_donor"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        parsed_indices = parse_indices(response)
        parsed_indices = parsed_indices if parsed_indices is not None else []
        pred_set = set(parsed_indices)
        precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        is_exact_match = pred_set == gt_set
        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        print(f"Response [{donor_key}]: {response}")
        print(f"Predicted [{donor_key}] count: {len(parsed_indices)}")
        print(f"Ground truth [{donor_key}] count: {len(gt_indices)}")
        print(
            f"Metrics [{donor_key}] -> precision={precision:.4f} "
            f"recall={recall:.4f} f1={f1:.4f} exact_match={is_exact_match}"
        )

        for metric in iteration_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                    f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                    f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                }
            )

        if iteration_metrics:
            last_metric = iteration_metrics[-1]
            wandb.log(
                {
                    "sample_idx": i,
                    f"sample/{i}/acyl_donor_key": donor_key,
                    f"sample/{i}/TO_ALCOHOL_SMIRKS": donor_smarts,
                    f"sample/{i}/final_total_input_tokens": last_metric["total_input_tokens"],
                    f"sample/{i}/final_total_output_tokens": last_metric["total_output_tokens"],
                    f"sample/{i}/final_total_tokens": last_metric["total_tokens"],
                    f"sample/{i}/iterations": len(iteration_metrics),
                    f"sample/{i}/response_raw": response,
                    f"sample/{i}/response_parsed_indices": ",".join(
                        str(x) for x in parsed_indices
                    ),
                    f"sample/{i}/response_parsed_count": len(parsed_indices),
                    f"sample/{i}/ground_truth_indices": ",".join(
                        str(x) for x in gt_indices
                    ),
                    f"sample/{i}/ground_truth_count": len(gt_indices),
                    f"sample/{i}/precision": precision,
                    f"sample/{i}/recall": recall,
                    f"sample/{i}/f1": f1,
                    f"sample/{i}/is_exact_match": int(is_exact_match),
                    f"sample/{i}/completion_root_prompt": question,
                    f"sample/{i}/completion_prompt_char_count": len(context),
                    **(
                        {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                        if sample_cost_usd is not None
                        else {}
                    ),
                }
            )

    total = len(donor_keys)
    exact_match_accuracy = (exact_match_count / total) if total else 0.0
    macro_precision = (macro_precision / total) if total else 0.0
    macro_recall = (macro_recall / total) if total else 0.0
    macro_f1 = (macro_f1 / total) if total else 0.0
    print(f"Exact match: {exact_match_count}/{total}")
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    for donor_key in donor_keys:
        run.summary[f"ground_truth/{donor_key}/count"] = len(gt_indices_by_donor[donor_key])
        run.summary[f"ground_truth/{donor_key}/indices"] = ",".join(
            str(x) for x in gt_indices_by_donor[donor_key]
        )

    run.summary["exact_match_correct"] = exact_match_count
    run.summary["total"] = total
    run.summary["exact_match_accuracy"] = exact_match_accuracy
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
    main(model_name=args.model_name)
