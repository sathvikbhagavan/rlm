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

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True

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


AMIDE_COUPLING_SMIRKS: dict[str, str] = {
    "acyl_chloride_with_primary_amine": "[*:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[Cl].[#7;H2;!$(N[O,N]);D1;+0:5]>>[*:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[#7;H1;D2;+0:5]",
    "acyl_chloride_with_secondary_amine": "[*:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[Cl].[#7;H1;D2;+0:5]>>[*:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[#7;H0;D3;+0:5]",
    "carboxylic_acid_with_primary_amine": "[CX3;+0:2](=[O;H0;D1;+0:3])-[O;H1;D1;+0].[#7;H2;D1;+0:5]>>[CX3;+0:2](=[O;H0;D1;+0:3])-[#7;H1;D2;+0:5]",
    "ester_with_primary_amine": "[#6:1]-[C;H0;D3;+0:2](=[O;!$(OC(C)(C)C);H0;D1;+0:3])-[O;H0;D2;+0].[#7;H2;D1;+0:5]>>[#6:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[#7;H1;D2;+0:5]",
    "ester_with_secondary_amine": "[#6:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[O;!$(OC(C)(C)C);H0;D2;+0:4].[#7;H1;D2;+0:5]>>[#6:1]-[C;H0;D3;+0:2](=[O;H0;D1;+0:3])-[#7;H0;D3;+0:5]"
}

AMIDE_COUPLING_LABELS: dict[str, str] = {
    "acyl_chloride_with_primary_amine": "Acyl chloride with primary amine to amide (Schotten-Baumann)",
    "acyl_chloride_with_secondary_amine": "Acyl chloride with secondary amine to amide",
    "carboxylic_acid_with_primary_amine": "Carboxylic acid with primary amine to amide",
    "ester_with_primary_amine": "Ester with primary amine to amide",
    "ester_with_secondary_amine": "Ester with secondary amine to amide"
}

AMIDE_COUPLING_DESCRIPTIONS: dict[str, str] = {
    "acyl_chloride_with_primary_amine": "An amide bond formation through a nucleophilic acyl substitution between an acyl chloride and a primary amine. In this reaction, the nitrogen of the primary amine attacks the electrophilic carbonyl carbon of the acyl chloride, which is highly activated due to the electron-withdrawing chlorine substituent. The chloride ion acts as the leaving group and is displaced during the process, departing along with a proton as HCl. The product is an amide featuring a new carbon-nitrogen bond, while the original carbonyl (C=O) double bond is preserved. One of the two hydrogens on the amine nitrogen is consumed in forming the new bond, resulting in a secondary-type nitrogen in the product. The primary amine is restricted to simple amines whose nitrogen is not directly bonded to another nitrogen or oxygen, thereby excluding hydrazines, hydroxylamines, and similar heteroatom-substituted amines.",
    "acyl_chloride_with_secondary_amine": "An amide bond formation through a nucleophilic acyl substitution between an acyl chloride and a secondary amine. The nitrogen of the secondary amine, which already bears two carbon substituents and one hydrogen, attacks the electrophilic carbonyl carbon of the acyl chloride. The chloride ion is displaced as the leaving group, departing along with a proton as HCl. The product is a tertiary amide in which the nitrogen has lost its only hydrogen and now forms three bonds to carbon-containing groups, while the carbonyl (C=O) double bond remains intact. Because acyl chlorides are highly activated electrophiles, this reaction proceeds readily and is a common method for constructing sterically hindered or N,N-disubstituted amides in synthetic chemistry.",
    "carboxylic_acid_with_primary_amine": "An amide bond formation through a condensation reaction between a carboxylic acid and a primary amine. In this reaction, the nucleophilic nitrogen of the amine attacks the electrophilic carbonyl carbon of the carboxylic acid. The hydroxyl group on the carboxylic acid acts as the leaving group and is displaced during the process, ultimately departing as water. The product is an amide, featuring a new carbon-nitrogen bond while the original carbonyl (C=O) double bond is preserved. One of the two hydrogens on the amine nitrogen is consumed in forming the new bond, leaving a secondary-type nitrogen in the product. This reaction is one of the most fundamental transformations in both organic chemistry and biology, serving as the basis for peptide bond formation during protein synthesis.",
    "ester_with_primary_amine": "An amide bond formation through aminolysis of an ester by a primary amine. In this reaction, the nucleophilic nitrogen of the primary amine attacks the electrophilic carbonyl carbon of the ester, displacing the alkoxy group as an alcohol leaving group. The carbonyl (C=O) double bond is preserved in the product, while a new carbon-nitrogen amide bond is formed. One of the two hydrogens on the amine nitrogen is consumed during bond formation, yielding a secondary-type nitrogen in the resulting amide. The substituent on the carbonyl carbon is restricted to a carbon-based group, confirming this is an organic ester. This transformation is commonly used in synthetic chemistry when milder conditions than acyl chloride chemistry are desired, though it typically requires heating or catalytic activation.",
    "ester_with_secondary_amine": "An amide bond formation through aminolysis of an ester by a secondary amine. The nitrogen of the secondary amine, bearing one hydrogen and two carbon substituents, attacks the electrophilic carbonyl carbon of the ester. The alkoxy group is displaced as an alcohol leaving group, while the carbonyl (C=O) double bond remains intact in the product. The sole hydrogen on the amine nitrogen is consumed during the reaction, producing a tertiary amide in which the nitrogen carries three bonds to carbon-containing groups and no hydrogens. The ester specifically excludes tert-butyl esters, where the leaving oxygen is bonded to a quaternary carbon bearing three methyl groups, to avoid competing Boc-deprotection pathways. This reaction is useful for constructing N,N-disubstituted amides and generally requires elevated temperatures or coupling reagents to proceed efficiently."
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


def reaction_matches_acylation(
    indexed_line: str, query_reaction: rdChemReactions.ChemicalReaction
) -> bool:
    try:
        reactants, products = parse_reaction_sides(indexed_line)
    except ValueError:
        return False

    if not reactants or not products:
        return False

    reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".")]
    product_mols  = [Chem.MolFromSmiles(smi) for smi in products.split(".")]

    if any(mol is None for mol in reactant_mols):
        return False
    if any(mol is None for mol in product_mols):
        return False

    query_products = query_reaction.GetProducts()

    try:
        from itertools import permutations
        for perm in permutations(reactant_mols):
            results = query_reaction.RunReactants(perm)
            if not results:
                continue
            # Verify predicted products match actual products in dataset
            for q in query_products:
                if any(mol.HasSubstructMatch(q) for mol in product_mols):
                    return True
        return False
    except Exception:
        return False

def ground_truth_indices(
    lines: list[str], query_reaction: rdChemReactions.ChemicalReaction
) -> list[int]:
    matching_indices: list[int] = []
    for line in lines:
        idx_str, _ = line.split(" ", 1)
        idx = int(idx_str)
        if reaction_matches_acylation(line, query_reaction):
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
        project_name="RLMs-Task6",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLM task 6 prompt-only evaluation.")
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
    donor_keys = list(AMIDE_COUPLING_SMIRKS.keys())

    gt_indices_by_donor: dict[str, list[int]] = {}
    for donor_key in donor_keys:
        smarts = AMIDE_COUPLING_SMIRKS[donor_key]
        query_reaction = build_reaction_query(smarts)
        gt_indices = ground_truth_indices(lines, query_reaction)
        gt_indices_by_donor[donor_key] = gt_indices
        print(f"Ground truth [{donor_key}] count={len(gt_indices)} ({smarts})")

    run = wandb.init(
        project="RLMs-Task6",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "num_questions": len(donor_keys),
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Count amide acylation reactions per acyl donor class.",
            "AMIDE_COUPLING_SMIRKS": AMIDE_COUPLING_SMIRKS,
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
        reaction_label = AMIDE_COUPLING_LABELS[donor_key]
        reaction_description = AMIDE_COUPLING_DESCRIPTIONS[donor_key]
        donor_smarts = AMIDE_COUPLING_SMIRKS[donor_key]
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
                "task": "amide_acylation_count",
                "acyl_donor_key": donor_key,
                "AMIDE_COUPLING_SMIRKS": donor_smarts,
            },
            tags=["run_rlms", "sample", "task6_amide_acylation_by_donor"],
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
                    f"sample/{i}/AMIDE_COUPLING_SMIRKS": donor_smarts,
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
