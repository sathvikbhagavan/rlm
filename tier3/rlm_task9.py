import argparse
import os
import re
import uuid
from typing import Optional
from itertools import permutations

import wandb
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rlm import RLM
from rlm.tracing import init_tracing, using_tracing_attributes

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "x-ai/grok-4-fast"
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


NAMED_REACTIONS_SMIRKS: dict[str, str] = {
    "suzuki_coupling_with_boronic_acids": "[#6;$([#6]:[#6]),$([#6]=[#6]),$([#6]#[#6]);+0:1]-[B;H0;D3;+0](-[O;H1;D1;+0])-[O;H1;D1;+0].[#6;$([#6]=[#6]),$([#6]~[#6]:[#6]),$([#6]~n);+0:2][Cl,Br,I]>>[#6;$([#6]:[#6]),$([#6]=[#6]),$([#6]#[#6]);+0:1]-[#6;$([#6]=[#6]),$([#6]~[#6]:[#6]),$([#6]~n);+0:2]",
    "mitsunobu_sulfonamide": "[C;H1&$(C([#6])[#6]),H2&$(C[#6]):1][OH1].[NH1;$(N([#6])S(=O)=O):2]>>[C:1][N:2]",
    # "sonogashira_coupling_terminal_alkyne_with_aryl_halide": "[c:0]-[Cl,Br,I].[#6;H0;D2;+0:1]#[C;H1;D1;+0:2]>>[c:0]-[#6;H0;D2;+0:1]#[C;H1;D1;+0:2]",
    "buchwald_hartwig_n_arylation_primary_amine": "[c;H0;D3;+0:0]-[F,Cl,Br,I].[#6;+0:1]-[N;H2;D1;+0:2]>>[c;H0;D3;+0:0]-[N;H1;D2;+0:2]-[#6;+0:1]",
    "stille_reaction_aryl": "[C;H2,H3;+0]-[Sn;H0;D4;+0](-[C;H2,H3;+0])(-[C;H2,H3;+0])-[c;H0;D3;+0:0].[#6;+0:2]-[F,Cl,Br,I]>>[#6;+0:2]-[c;H0;D3;+0:0]",
    "wittig_with_phosphonium": "[#6:1]-[#6;+0:2](=O).[P;+1]-[C;H2;D2;+0:3]-[*:4]>>[#6:1]-[#6;+0:2]=[C;H1;D2;+0:3]-[*:4]"
}

NAMED_REACTIONS_LABELS: dict[str, str] = {
    "suzuki_coupling_with_boronic_acids": "Suzuki coupling with boronic acids",
    "mitsunobu_sulfonamide": "Mitsunobu sulfonamide",
    # "sonogashira_coupling_terminal_alkyne_with_aryl_halide": "Sonogashira coupling of terminal alkyne with aryl halide",
    "buchwald_hartwig_n_arylation_primary_amine": "Buchwald-Hartwig Ullmann-Goldberg N-arylation primary amine",
    "stille_reaction_aryl": "Stille reaction aryl",
    "wittig_with_phosphonium": "Wittig with Phosphonium"
}

NAMED_REACTIONS_DESCRIPTIONS: dict[str, str] = {
    "suzuki_coupling_with_boronic_acids": "A Suzuki cross-coupling reaction in which a boronic acid reacts with an organohalide under palladium catalysis to form a new carbon-carbon bond. The boronic acid partner is restricted to aryl, vinyl, or alkynyl carbons attached to a B(OH)₂ group, while the halide partner carries a chlorine, bromine, or iodide leaving group on an aryl, vinyl, or heteroaryl carbon. In the product, the boron moiety and halide are both lost, and a direct C-C bond forms between the two coupling partners. Both sides of the coupling are restricted to sp2 or sp carbons, consistent with the mechanistic requirements of oxidative addition and transmetalation in the Suzuki catalytic cycle.",
    "mitsunobu_sulfonamide": "A Mitsunobu reaction in which a sulfonamide nitrogen displaces a hydroxyl group on a primary or secondary alcohol, forming a new carbon-nitrogen bond with inversion of stereochemistry. The alcohol carbon is restricted to either a secondary carbon with one hydrogen and two carbon neighbors, or a primary carbon with two hydrogens and one carbon neighbor — excluding tertiary alcohols and methanol. The nitrogen nucleophile is a sulfonamide bearing one hydrogen, bonded to a carbon substituent and a sulfonyl group (S(=O)=O). In the product, the hydroxyl group is lost and a direct C-N bond forms between the alcohol carbon and the sulfonamide nitrogen. This reaction is mediated by a phosphine (typically triphenylphosphine) and a dialkyl azodicarboxylate (DIAD or DEAD), which together activate the alcohol as a leaving group and enable the SN2 displacement.",
    # "sonogashira_coupling_terminal_alkyne_with_aryl_halide": "A Sonogashira cross-coupling reaction in which a terminal alkyne couples with an aryl halide to form an aryl-alkyne (C-C) bond. The aryl halide consists of an aromatic carbon bearing a chlorine, bromine, or iodine leaving group. The terminal alkyne has a substituted carbon with no hydrogens and two connections (one to the substituent, one to the triple bond) and a terminal carbon with one hydrogen and one connection. In the product, the halide is displaced and the aromatic carbon forms a new bond to the substituted alkyne carbon, which retains its two connections and gains no hydrogens, while the terminal alkyne carbon remains unchanged with its hydrogen intact. This reaction is typically catalyzed by a palladium complex with a copper(I) co-catalyst and a base, and is widely used for introducing alkyne functionality onto aromatic rings.",
    "buchwald_hartwig_n_arylation_primary_amine": "A palladium- or copper-catalyzed N-arylation in which a primary amine couples with an aryl halide to form a new aryl carbon-nitrogen bond. The aryl halide consists of a neutral aromatic carbon with no hydrogens and three connections, bearing a fluorine, chlorine, bromine, or iodine leaving group. The primary amine has a neutral nitrogen with two hydrogens and one connection, bonded to a carbon-based substituent. In the product, the halide is displaced and the nitrogen forms a direct bond to the aromatic carbon, losing one hydrogen (going from two to one) and gaining one connection (going from one to two), yielding a secondary arylamine. This transformation encompasses several named reactions including Buchwald-Hartwig amination, Ullmann-Goldberg coupling, and nucleophilic aromatic substitution, depending on the catalyst and conditions employed.",
    "stille_reaction_aryl": "A Stille cross-coupling reaction in which an aryl group is transferred from an organostannane to an organohalide under palladium catalysis, forming a new carbon-carbon bond. The organostannane consists of a tin center with no hydrogens and four connections — three alkyl substituents (methyl or longer chain, with two or three hydrogens on the carbon directly bonded to tin) and one aromatic carbon with no hydrogens and three connections that serves as the transferred group. The coupling partner is a neutral carbon bearing a fluorine, chlorine, bromine, or iodine leaving group. In the product, the tin moiety and halide are both lost, and a direct bond forms between the electrophilic carbon and the aromatic carbon from the stannane. This reaction is valued for its tolerance of diverse functional groups and mild reaction conditions.",
    "wittig_with_phosphonium": "A Wittig olefination in which a phosphonium ylide reacts with an aldehyde or ketone to form a new carbon-carbon double bond. The carbonyl component has a neutral carbon bonded to a carbon substituent and a double-bonded oxygen. The phosphonium salt consists of a positively charged phosphorus bonded to a methylene carbon with two hydrogens and two connections, carrying one substituent. In the product, the carbonyl oxygen and phosphorus are both lost, and the carbonyl carbon forms a double bond to the ylide carbon, which loses one hydrogen (going from two to one) while retaining two connections. The resulting alkene bridges the two original substituents. This template specifically covers monosubstituted phosphonium ylides and is one of the most widely used methods for constructing alkenes with defined geometry."
}


def parse_reaction_sides(indexed_line: str) -> tuple[str, str]:
    _, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    reactant_smiles = [s for s in parts[0].split(".") if s]
    product_smiles = [s for s in parts[2].split(".") if s]
    reactants = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    products = [Chem.MolFromSmiles(s) for s in product_smiles]
    reactants = [m for m in reactants if m is not None]
    products = [m for m in products if m is not None]
    return reactants, products


def build_reaction_query(smarts: str) -> rdChemReactions.ChemicalReaction:
    query = rdChemReactions.ReactionFromSmarts(smarts)
    if query is None:
        raise ValueError(f"Failed to parse reaction SMARTS: {smarts}")
    return query


def canonical_smiles_set(mols):
    """Return a set of canonical SMILES for a list of molecules."""
    return {Chem.MolToSmiles(m) for m in mols if m is not None}


def reaction_matches(reaction_smiles, query_reaction):
    reactants, products = parse_reaction_sides(reaction_smiles)
    template = query_reaction
    template.Initialize()
    actual_product_smiles = canonical_smiles_set(products)
    num_template_reactants = template.GetNumReactantTemplates()
    for perm in permutations(reactants, min(num_template_reactants, len(reactants))):
        if len(perm) != num_template_reactants:
            continue
        try:
            product_sets = template.RunReactants(perm)
        except Exception:
            continue
        for prod_set in product_sets:
            generated_smiles = set()
            for mol in prod_set:
                try:
                    Chem.SanitizeMol(mol)
                    generated_smiles.add(Chem.MolToSmiles(mol))
                except Exception:
                    continue
            # Match if every generated product appears in the actual products
            if generated_smiles and generated_smiles.issubset(actual_product_smiles):
                return True
    return False


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
    - DO NOT assume/simulate output of the code. Wait for the code to get executed and only then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.

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
        project_name="RLMs-Task9",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLM task 9 prompt-only evaluation.")
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
    reaction_keys = list(NAMED_REACTIONS_SMIRKS.keys())

    gt_indices_by_reaction: dict[str, list[int]] = {}
    for reaction_key in reaction_keys:
        smarts = NAMED_REACTIONS_SMIRKS[reaction_key]
        query_reaction = build_reaction_query(smarts)
        gt_indices = ground_truth_indices(lines, query_reaction)
        gt_indices_by_reaction[reaction_key] = gt_indices
        print(f"Ground truth [{reaction_key}] count={len(gt_indices)} ({smarts})")

    run = wandb.init(
        project="RLMs-Task9",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "num_questions": len(reaction_keys),
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Count from named reactions.",
            "NAMED_REACTIONS_SMIRKS": NAMED_REACTIONS_SMIRKS,
            "ground_truth_indices_by_reaction": gt_indices_by_reaction,
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

    for i, reaction_key in enumerate(reaction_keys):
        reaction_label = NAMED_REACTIONS_LABELS[reaction_key]
        reaction_description = NAMED_REACTIONS_DESCRIPTIONS[reaction_key]
        reaction_smirks = NAMED_REACTIONS_SMIRKS[reaction_key]
        question = build_question(reaction_label=reaction_label, reaction_description=reaction_description)
        gt_indices = gt_indices_by_reaction[reaction_key]
        gt_set = set(gt_indices)
        completion_kwargs = {"prompt": context, "root_prompt": question}

        print(f"Question {i + 1}/{len(reaction_keys)} donor={reaction_key}")

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(reaction_keys),
                "task": "NAMED_REACTIONS_count",
                "NAMED_REACTIONS_reaction_key": reaction_key,
                "NAMED_REACTIONS_SMIRKS": reaction_smirks,
            },
            tags=["run_rlms", "sample", "task9_NAMED_REACTIONS"],
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

        print(f"Response [{reaction_key}]: {response}")
        print(f"Predicted [{reaction_key}] count: {len(parsed_indices)}")
        print(f"Ground truth [{reaction_key}] count: {len(gt_indices)}")
        print(
            f"Metrics [{reaction_key}] -> precision={precision:.4f} "
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
                    f"sample/{i}/reaction_key": reaction_key,
                    f"sample/{i}/NAMED_REACTIONS_SMIRKS": reaction_smirks,
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

    total = len(reaction_keys)
    exact_match_accuracy = (exact_match_count / total) if total else 0.0
    macro_precision = (macro_precision / total) if total else 0.0
    macro_recall = (macro_recall / total) if total else 0.0
    macro_f1 = (macro_f1 / total) if total else 0.0
    print(f"Exact match: {exact_match_count}/{total}")
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    for reaction_key in reaction_keys:
        run.summary[f"ground_truth/{reaction_key}/count"] = len(gt_indices_by_reaction[reaction_key])
        run.summary[f"ground_truth/{reaction_key}/indices"] = ",".join(
            str(x) for x in gt_indices_by_reaction[reaction_key]
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
