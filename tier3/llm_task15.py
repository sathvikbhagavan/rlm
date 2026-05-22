import argparse
import asyncio
import math
import os
import random
import uuid

import wandb
from rdkit import Chem
from rdkit.Chem import rdFMCS
from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter

from rlm.codeact_helpers import (
    build_retriever,
    extract_response_text,
    extract_usage_metrics,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023_cleaned.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
SEED = 42
CONTEXT_SIZE = 500
GROUND_TRUTH_FRACTION_PER_CONTEXT = 0.2
RETRIEVER_NAME = "random"
MAX_OUTPUT_TOKENS = 30_000
REASONING_EFFORT = "high"
MIN_CORE_ATOMS = 3
MIN_CORE_FRACTION = 0.5
MCS_TIMEOUT_SECONDS = 2

REACTION_KEY = "achiral_to_chiral"
REACTION_LABEL = "Achiral substrates to chiral products"
REACTION_DESCRIPTION = (
    "A reaction matches when all reactant-side molecules are achiral and at least one "
    "stereocenter exists in the largest product on the atoms belonging to the preserved "
    "core of the largest reactant."
)


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="LLM-Task15",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM task 15 evaluation.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    parser.add_argument(
        "--ground-truth-fraction-per-context",
        type=float,
        default=GROUND_TRUTH_FRACTION_PER_CONTEXT,
    )
    return parser.parse_args()


def parse_reaction_sides(indexed_line: str) -> tuple[int, list[str], list[str]]:
    idx_str, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    if len(parts) != 3:
        raise ValueError("Reaction must have reactants>reagents>products format.")
    reactant_smiles = [s.strip() for s in parts[0].split(".") if s.strip()]
    product_smiles = [s.strip() for s in parts[2].split(".") if s.strip()]
    return int(idx_str), reactant_smiles, product_smiles


def mols_from_smiles(smiles_list: list[str]) -> list[Chem.Mol]:
    mols: list[Chem.Mol] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        mols.append(mol)
    return mols


def find_chiral_centers(mol: Chem.Mol) -> list[tuple[int, str]]:
    return Chem.FindMolChiralCenters(
        mol,
        includeUnassigned=True,
        useLegacyImplementation=False,
    )


def get_preserved_core_product_atoms(substrate: Chem.Mol, product: Chem.Mol) -> set[int]:
    substrate_heavy = substrate.GetNumHeavyAtoms()
    if substrate_heavy == 0:
        return set()

    mcs_params = rdFMCS.MCSParameters()
    mcs_params.Timeout = MCS_TIMEOUT_SECONDS
    mcs_params.AtomCompareParameters.MatchChiralTag = False
    mcs = rdFMCS.FindMCS([substrate, product], parameters=mcs_params)
    if mcs.canceled or mcs.numAtoms == 0:
        return set()

    min_core_atoms = max(MIN_CORE_ATOMS, math.ceil(MIN_CORE_FRACTION * substrate_heavy))
    if mcs.numAtoms < min_core_atoms:
        return set()

    core_pattern = Chem.MolFromSmarts(mcs.smartsString)
    if core_pattern is None:
        return set()
    matches = product.GetSubstructMatches(core_pattern, useChirality=False)
    if not matches:
        return set()

    atom_indices: set[int] = set()
    for match in matches:
        atom_indices.update(match)
    return atom_indices


def is_achiral_to_chiral(indexed_line: str) -> bool:
    _, reactant_smiles, product_smiles = parse_reaction_sides(indexed_line)
    reactants = mols_from_smiles(reactant_smiles)
    products = mols_from_smiles(product_smiles)
    if not reactants or not products:
        return False

    all_reactants_achiral = all(len(find_chiral_centers(mol)) == 0 for mol in reactants)
    if not all_reactants_achiral:
        return False

    substrate = max(reactants, key=lambda mol: mol.GetNumHeavyAtoms())
    product = max(products, key=lambda mol: mol.GetNumHeavyAtoms())

    product_chiral_centers = find_chiral_centers(product)
    if not product_chiral_centers:
        return False

    preserved_core_atoms = get_preserved_core_product_atoms(substrate, product)
    if not preserved_core_atoms:
        return False

    return any(atom_idx in preserved_core_atoms for atom_idx, _ in product_chiral_centers)


def ground_truth_indices(lines: list[str]) -> list[int]:
    indices: list[int] = []
    for line in lines:
        try:
            idx_str, _ = line.split(" ", 1)
            idx = int(idx_str)
            if is_achiral_to_chiral(line):
                indices.append(idx)
        except Exception:
            continue
    indices.sort()
    return indices


def build_question() -> str:
    return f"""
    Context: You are given a large string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Task:
    Return all reaction indices that match this reaction type:
    - {REACTION_LABEL}

    Description:
    - {REACTION_DESCRIPTION}

    Guidance:
    - Use RDKit for parsing and stereochemistry analysis.
    - Ignore reagents (middle field).
    - All reactant-side molecules must be achiral (no assigned or unassigned stereocenters).
    - Select largest reactant and largest product by heavy atom count.
    - Largest product must have at least one stereocenter.
    - Confirm substrate continuity by MCS between largest reactant and largest product.
    - At least one largest-product stereocenter must lie on atoms in that preserved core.
    - Skip malformed reactions.
    - DO NOT assume/simulate output of the code. Wait for code execution and then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.

    Output format:
    - Return ONLY the matching reaction INDICES.
    - Format must be a comma-separated list of integers in ascending order (e.g., 3,8,21).
    - No other text, quotes, labels, punctuation, or formatting.
    - If no matching reaction exists, return -1.
    """


async def main(
    model_name: str,
    context_size: int,
    ground_truth_fraction_per_context: float,
) -> None:
    maybe_init_tracing()
    lines = load_lines(DATASET_PATH)
    rng = random.Random(SEED)
    reaction_keys = [REACTION_KEY]
    run_session_id = f"llm-task15-{uuid.uuid4()}"

    full_gt_indices_by_reaction = {REACTION_KEY: ground_truth_indices(lines)}
    retriever = build_retriever(
        name=RETRIEVER_NAME,
        lines=lines,
        rng=rng,
        ground_truth_indices_by_reaction=full_gt_indices_by_reaction,
        ground_truth_fraction_per_context=ground_truth_fraction_per_context,
    )
    llm = OpenRouter(
        model=model_name,
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
    )

    run = wandb.init(
        project="LLM-Task15",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "seed": SEED,
            "context_size": context_size,
            "ground_truth_fraction_per_context": ground_truth_fraction_per_context,
            "retriever_name": RETRIEVER_NAME,
            "reasoning_effort": REASONING_EFFORT,
            "task_description": "Return reaction indices for achiral-to-chiral transformations.",
            "full_ground_truth_indices_by_reaction": full_gt_indices_by_reaction,
            "mode": "llm_baseline_no_tools",
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
        question = build_question()
        retrieved_context = retriever.build_context(query=reaction_key, target_index=-1, k=context_size)
        retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
        context_coverage = len(retrieved_lines) / len(lines) if lines else 0.0
        gt_indices = ground_truth_indices(retrieved_lines)
        gt_set = set(gt_indices)
        total_gt_count = len(full_gt_indices_by_reaction[reaction_key])

        completion_prompt = f"""
        You are given a subset of chemical reactions in SMILES format and a question.
        <context>
        {retrieved_context}
        </context>
        <question>
        {question}
        </question>
        """
        print(f"Question {i + 1}/{len(reaction_keys)} for reaction_key={reaction_key}")
        print(
            f"Ground truth present in context: {len(gt_indices)}/{total_gt_count} "
            f"(context lines: {len(retrieved_lines)})"
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(reaction_keys),
                "reaction_key": reaction_key,
                "agent": "llm_baseline",
            },
            tags=["llm-baseline", "sample", "task15_achiral_to_chiral"],
        ):
            response = await llm.achat([ChatMessage(role="user", content=completion_prompt)])

        response_text = extract_response_text(response)
        parsed_indices = parse_indices(response_text)
        pred_set = set(parsed_indices)
        precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
        is_exact_match = pred_set == gt_set
        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        usage_metrics = extract_usage_metrics(response)
        prompt_tokens = int(usage_metrics.get("prompt_tokens", 0))
        completion_tokens = int(usage_metrics.get("completion_tokens", 0))
        total_tokens = int(usage_metrics.get("total_tokens", 0))
        sample_cost = float(usage_metrics["cost_usd"]) if "cost_usd" in usage_metrics else None
        if total_tokens == 0:
            prompt_tokens = count_tokens([{"role": "user", "content": completion_prompt}], model_name)
            completion_tokens = count_tokens(
                [{"role": "assistant", "content": response_text}],
                model_name,
            )
            total_tokens = prompt_tokens + completion_tokens
        if sample_cost is not None:
            total_cost_usd += sample_cost
            samples_with_cost += 1

        wandb.log(
            {
                "sample_iteration": 1,
                f"sample/{i}/iteration_input_tokens": prompt_tokens,
                f"sample/{i}/iteration_output_tokens": completion_tokens,
                f"sample/{i}/iteration_total_tokens": total_tokens,
                **({f"sample/{i}/iteration_cost_usd": sample_cost} if sample_cost is not None else {}),
            }
        )
        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/reaction_key": reaction_key,
                f"sample/{i}/final_total_input_tokens": prompt_tokens,
                f"sample/{i}/final_total_output_tokens": completion_tokens,
                f"sample/{i}/final_total_tokens": total_tokens,
                f"sample/{i}/iterations": 1,
                f"sample/{i}/is_exact_match": int(is_exact_match),
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/ground_truth_count": len(gt_indices),
                f"sample/{i}/prediction_count": len(parsed_indices),
                f"sample/{i}/ground_truth_indices": ",".join(str(x) for x in gt_indices),
                f"sample/{i}/predicted_indices": ",".join(str(x) for x in parsed_indices),
                f"sample/{i}/response_raw": response_text,
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_coverage": context_coverage,
                **({f"sample/{i}/final_total_cost_usd": sample_cost} if sample_cost is not None else {}),
            }
        )
        wandb.log(
            {
                "running_exact_match_accuracy": exact_match_count / (i + 1),
                "running_macro_precision": macro_precision / (i + 1),
                "running_macro_recall": macro_recall / (i + 1),
                "running_macro_f1": macro_f1 / (i + 1),
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
    run.summary["full_ground_truth/achiral_to_chiral/count"] = len(
        full_gt_indices_by_reaction[REACTION_KEY]
    )
    run.summary["full_ground_truth/achiral_to_chiral/indices"] = ",".join(
        str(x) for x in full_gt_indices_by_reaction[REACTION_KEY]
    )
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            model_name=args.model_name,
            context_size=args.context_size,
            ground_truth_fraction_per_context=args.ground_truth_fraction_per_context,
        )
    )
