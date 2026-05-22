import argparse
import asyncio
import os
import random
import uuid
from itertools import permutations

import wandb
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from llama_index.core.workflow import Context
from llama_index.llms.openrouter import OpenRouter

from rlm.codeact_core import (
    CodeActAgent,
    INDEX_CODEACT_SYSTEM_PROMPT,
    INDEX_FORCE_LOOP_MESSAGE,
    INDEX_OBSERVATION_FOLLOWUP,
    make_simple_code_executor,
    run_agent_verbose,
)
from rlm.codeact_helpers import (
    build_retriever,
    extract_response_text,
    load_lines,
    parse_indices,
)
from rlm.tracing import get_tracer, init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
MODEL_NAME = "openai/gpt-5-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 1200.0
SEED = 42
CONTEXT_SIZE = 500
GROUND_TRUTH_FRACTION_PER_CONTEXT = 0.2
RETRIEVER_NAME = "random"
MAX_OUTPUT_TOKENS = 50_000
REASONING_EFFORT = "high"
MAX_ITERATIONS = 8
# os.environ["WANDB_MODE"] = "disabled"


TO_FG_SMIRKS: dict[str, str] = {
   "grignard_ketone_to_tertiary_alcohol": "[#6;+0:1]-[Mg]-[Br,I,Cl].[*:2]-[C;H0;D3;+0:3](=[O;H0;D1;+0:4])-[#6;+0:5]>>[*:2]-[C;H0;D4;+0:3](-[O;H1;D1;+0:4])(-[#6;+0:5])-[#6;+0:1]",
   "grignard_aldehyde_to_secondary_alcohol": "[#6;+0:1]-[Mg]-[Br,I,Cl].[*:2]-[C;H1;D2;+0:3](=[O;H0;D1;+0:4])>>[*:2]-[C;H1;D3;+0:3](-[O;H1;D1;+0:4])-[#6;+0:1]",
   "nitrile_to_amine": "[#6;+0:0]-[C;H0;D2;+0:1]#[N;H0;D1;+0:2]>>[#6;+0:0]-[C;H2;D2;+0:1]-[N;H2;D1;+0:2]",
   "nitro_groups_to_amines": "[N;H0;D3;+1:1](=[O;H0;D1;+0])-[O;H0;D1;-1]>>[N;H2;D1;+0:1]",
   "alcohol_to_azide": "[#6;+0:0]-[O;H1;D1;+0]>>[#6;+0:0]-[N;H0;D2;+0]=[N;H0;D2;+1]=[N;H0;D1;-1]",
   "alcohol_to_carboxylic_acid": "[C;H2;D2;+0:1]-[O;H1;D1;+0:2]>>[C;H0;D3;+0:1](=[O;H0;D1;+0:2])-[O;H1;D1;+0]"
}

TO_FG_LABELS: dict[str, str] = {
    "grignard_ketone_to_tertiary_alcohol": "Grignard from ketone to alcohol",
    "grignard_aldehyde_to_secondary_alcohol": "Grignard from aldehyde to secondary alcohol",
    "nitrile_to_amine": "Reduction of nitrile to amine",
    "nitro_groups_to_amines": "Reduction of nitro groups to amines",
    "alcohol_to_azide": "Alcohol to azide",
    "alcohol_to_carboxylic_acid": "Oxidation of alcohol to carboxylic acid"
}

TO_FG_DESCRIPTIONS: dict[str, str] = {
    "grignard_ketone_to_tertiary_alcohol": "A Grignard reaction in which an organomagnesium halide adds to a ketone to form a tertiary alcohol. The carbon nucleophile of the Grignard reagent, bonded to magnesium which in turn bears a halide (bromide, iodide, or chloride), attacks the electrophilic carbonyl carbon of the ketone. The carbonyl carbon transitions from trigonal planar (three substituents) to tetrahedral (four substituents), gaining a new carbon-carbon bond to the Grignard carbon. The carbonyl oxygen is reduced from a double bond to a single bond, gaining a hydrogen to become a hydroxyl group in the product. Since the ketone carbonyl carbon already bears two carbon substituents, the addition of the Grignard carbon yields a tertiary alcohol with no hydrogens on the central carbon. This reaction is one of the most important carbon-carbon bond-forming reactions in organic synthesis, enabling the construction of complex molecular architectures from simpler precursors.",
    "grignard_aldehyde_to_secondary_alcohol": "A Grignard reaction in which an organomagnesium halide adds to an aldehyde to form a secondary alcohol. The carbon nucleophile of the Grignard reagent attacks the electrophilic carbonyl carbon of the aldehyde, forming a new carbon-carbon bond. The carbonyl carbon transitions from two substituents to three, and the carbonyl oxygen is reduced from a double bond to a hydroxyl group. Since the aldehyde carbon originally bears one hydrogen and one substituent, the addition of the Grignard carbon yields a secondary alcohol with one hydrogen remaining on the central carbon.",
    "nitrile_to_amine": "Reduction of a nitrile to a primary amine. The reactant contains a carbon-based substituent bonded to a neutral nitrile carbon with no hydrogens and two connections, which is triple-bonded to a nitrogen with no hydrogens and one connection. In the product, the triple bond is fully reduced to a single bond — the carbon gains two hydrogens (going from zero to two) while retaining two connections, and the nitrogen also gains two hydrogens (going from zero to two) while remaining singly connected. The result is a primary amine, with four hydrogen atoms added overall across the carbon and nitrogen.",
    "nitro_groups_to_amines": "Reduction of a nitro group to a primary amine. The reactant contains a nitrogen with no hydrogens, three connections, and a positive formal charge, double-bonded to one oxygen and single-bonded to another oxygen carrying a negative formal charge — the canonical representation of a nitro group (-NO₂). Both oxygens are unmapped and are completely removed during the transformation. In the product, the nitrogen loses all its oxygen substituents, drops from three connections to one, changes from a positive to neutral charge, and gains two hydrogens, becoming a free primary amine.",
    "alcohol_to_azide": "Conversion of an alcohol to an azide via substitution of the hydroxyl group. The reactant contains a carbon-based substituent bonded to a neutral oxygen with one hydrogen and one connection — a simple hydroxyl group. In the product, the hydroxyl is replaced by an azide moiety consisting of three nitrogen atoms in a linear arrangement: the first nitrogen has no hydrogens, two connections, and is neutral; the middle nitrogen has no hydrogens, two connections, and carries a positive formal charge; and the terminal nitrogen has no hydrogens, one connection, and carries a negative formal charge. The oxygen is unmapped and fully removed, while the entire azide group is unmapped and newly introduced.",
    "alcohol_to_carboxylic_acid": "Oxidation of a primary alcohol to a carboxylic acid. The reactant contains a neutral carbon with two hydrogens and two connections, bonded to a neutral hydroxyl oxygen with one hydrogen and one connection — a primary alcohol (R-CH₂-OH). In the product, the carbon loses both hydrogens (going from two to zero) and gains an additional connection (going from two to three). The original oxygen loses its hydrogen (going from one to zero) and becomes double-bonded to the carbon, forming a carbonyl. A new hydroxyl oxygen, unmapped in the reactant and freshly introduced, appears single-bonded to the carbon with one hydrogen and one connection. The result is a carboxylic acid (R-C(=O)-OH), representing a four-electron oxidation."
}


def parse_reaction_sides(indexed_line: str) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
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


def canonical_smiles_set(mols: list[Chem.Mol]) -> set[str]:
    return {Chem.MolToSmiles(m) for m in mols if m is not None}


def reaction_matches(indexed_line: str, query_reaction: rdChemReactions.ChemicalReaction) -> bool:
    reactants, products = parse_reaction_sides(indexed_line)
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
            generated_smiles: set[str] = set()
            for mol in prod_set:
                try:
                    Chem.SanitizeMol(mol)
                    generated_smiles.add(Chem.MolToSmiles(mol))
                except Exception:
                    continue
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
    predicted_indices: set[int], gt_indices: set[int]
) -> tuple[float, float, float]:
    tp = len(predicted_indices & gt_indices)
    fp = len(predicted_indices - gt_indices)
    fn = len(gt_indices - predicted_indices)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
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
    - DO NOT do `exit()` in the code in any case.
    
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
        project_name="CodeAct-Task7",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAct task 7 evaluation.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    parser.add_argument(
        "--ground-truth-fraction-per-context",
        type=float,
        default=GROUND_TRUTH_FRACTION_PER_CONTEXT,
    )
    return parser.parse_args()


def build_code_executor(lines: list[str]):
    return make_simple_code_executor(
        extra_locals={"lines": lines},
        extra_globals={
            "np": __import__("numpy"),
            "rdkit": __import__("rdkit"),
        },
    )


async def main(
    model_name: str,
    context_size: int,
    ground_truth_fraction_per_context: float,
) -> None:
    maybe_init_tracing()
    tracer = get_tracer("codeact-task7")
    lines = load_lines()
    context = "\n".join(lines)
    rng = random.Random(SEED)
    reaction_keys = list(TO_FG_SMIRKS.keys())
    run_session_id = f"codeact-task7-{uuid.uuid4()}"

    full_gt_indices_by_reaction: dict[str, list[int]] = {}
    for reaction_key in reaction_keys:
        query_reaction = build_reaction_query(TO_FG_SMIRKS[reaction_key])
        full_gt_indices_by_reaction[reaction_key] = ground_truth_indices(lines, query_reaction)

    retriever = build_retriever(
        name=RETRIEVER_NAME,
        lines=lines,
        rng=rng,
        ground_truth_indices_by_reaction=full_gt_indices_by_reaction,
        ground_truth_fraction_per_context=ground_truth_fraction_per_context,
    )
    retriever_name = RETRIEVER_NAME if context_size >= 0 else "all_lines"

    run = wandb.init(
        project="CodeAct-Task7",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": SEED,
            "context_size": context_size,
            "ground_truth_fraction_per_context": ground_truth_fraction_per_context,
            "retriever_name": retriever_name,
            "reasoning_effort": REASONING_EFFORT,
            "task_description": "Return reaction indices for Grignard alcohol-formation subtypes.",
            "TO_FG_SMIRKS": TO_FG_SMIRKS,
            "full_ground_truth_indices_by_reaction": full_gt_indices_by_reaction,
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
        reaction_label = TO_FG_LABELS[reaction_key]
        reaction_description = TO_FG_DESCRIPTIONS[reaction_key]
        reaction_smirks = TO_FG_SMIRKS[reaction_key]
        query_reaction = build_reaction_query(reaction_smirks)
        question = build_question(reaction_label=reaction_label, reaction_description=reaction_description)

        if context_size < 0:
            retrieved_context = context
            retrieved_lines = lines
        else:
            retrieved_context = retriever.build_context(query=reaction_key, target_index=-1, k=context_size)
            retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
        context_coverage = len(retrieved_lines) / len(lines) if lines else 0.0
        gt_indices_full = full_gt_indices_by_reaction[reaction_key]
        gt_indices_in_context = ground_truth_indices(retrieved_lines, query_reaction)
        # Evaluate against only the ground-truth reactions present in retrieved context.
        gt_indices = gt_indices_in_context
        gt_set = set(gt_indices)
        total_gt_count = len(gt_indices_full)

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
            f"Ground truth count (full dataset): {len(gt_indices_full)}/{total_gt_count}, "
            f"in-context: {len(gt_indices_in_context)} (context lines: {len(retrieved_lines)})"
        )

        executor = build_code_executor(lines=retrieved_lines)
        agent = CodeActAgent(
            code_execute_fn=executor.execute,
            llm=OpenRouter(
                model=model_name,
                api_key=OPENROUTER_API_KEY,
                max_tokens=MAX_OUTPUT_TOKENS,
                reasoning_effort=REASONING_EFFORT,
                additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
            ),
            system_prompt=INDEX_CODEACT_SYSTEM_PROMPT,
            max_iterations=MAX_ITERATIONS,
            force_loop_message=INDEX_FORCE_LOOP_MESSAGE,
            observation_followup=INDEX_OBSERVATION_FOLLOWUP,
            timeout=WORKFLOW_TIMEOUT_S,
        )
        ctx = Context(agent)

        with tracer.start_as_current_span(f"codeact_task7_sample_{i}") as sample_span:
            sample_span.set_attributes(
                {
                    "sample.index": i,
                    "sample.count": len(reaction_keys),
                    "reaction.key": reaction_key,
                    "reaction.smirks": reaction_smirks,
                    "agent.name": "codeact",
                }
            )
            with using_tracing_attributes(
                session_id=run_session_id,
                metadata={
                    "sample_index": i,
                    "sample_count": len(reaction_keys),
                    "reaction_key": reaction_key,
                    "reaction_smirks": reaction_smirks,
                    "agent": "codeact",
                },
                tags=["codeact", "sample", "task7_to_alcohol"],
            ):
                response = await run_agent_verbose(agent, ctx, completion_prompt)

        response_text = extract_response_text(response)
        llm_turn_metrics = await ctx.store.get("llm_turn_metrics", default=[])
        if not llm_turn_metrics:
            estimated_prompt_tokens = count_tokens(
                [{"role": "user", "content": completion_prompt}],
                model_name,
            )
            estimated_completion_tokens = count_tokens(
                [{"role": "assistant", "content": response_text}],
                model_name,
            )
            llm_turn_metrics = [
                {
                    "iteration": 1,
                    "iteration_input_tokens": estimated_prompt_tokens,
                    "iteration_output_tokens": estimated_completion_tokens,
                    "iteration_total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
                }
            ]

        parsed_indices = parse_indices(response_text)
        pred_set = set(parsed_indices)
        precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
        is_exact_match = pred_set == gt_set
        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        for metric in llm_turn_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                    f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                    f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                    **(
                        {f"sample/{i}/iteration_cost_usd": metric["iteration_cost_usd"]}
                        if "iteration_cost_usd" in metric
                        else {}
                    ),
                }
            )

        final_input_tokens = sum(int(m.get("iteration_input_tokens", 0)) for m in llm_turn_metrics)
        final_output_tokens = sum(int(m.get("iteration_output_tokens", 0)) for m in llm_turn_metrics)
        final_total_tokens = sum(int(m.get("iteration_total_tokens", 0)) for m in llm_turn_metrics)
        final_cost = sum(float(m.get("iteration_cost_usd", 0.0)) for m in llm_turn_metrics)
        has_cost = any("iteration_cost_usd" in m for m in llm_turn_metrics)
        if has_cost:
            total_cost_usd += final_cost
            samples_with_cost += 1

        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/reaction_key": reaction_key,
                f"sample/{i}/reaction_smirks": reaction_smirks,
                f"sample/{i}/final_total_input_tokens": final_input_tokens,
                f"sample/{i}/final_total_output_tokens": final_output_tokens,
                f"sample/{i}/final_total_tokens": final_total_tokens,
                f"sample/{i}/iterations": len(llm_turn_metrics),
                f"sample/{i}/is_exact_match": int(is_exact_match),
                f"sample/{i}/precision": precision,
                f"sample/{i}/recall": recall,
                f"sample/{i}/f1": f1,
                f"sample/{i}/ground_truth_count": len(gt_indices),
                f"sample/{i}/ground_truth_in_context_count": len(gt_indices_in_context),
                f"sample/{i}/ground_truth_full_count": len(gt_indices_full),
                f"sample/{i}/prediction_count": len(parsed_indices),
                f"sample/{i}/ground_truth_indices": ",".join(str(x) for x in gt_indices),
                f"sample/{i}/ground_truth_in_context_indices": ",".join(
                    str(x) for x in gt_indices_in_context
                ),
                f"sample/{i}/predicted_indices": ",".join(str(x) for x in parsed_indices),
                f"sample/{i}/response_raw": response_text,
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_coverage": context_coverage,
                **({f"sample/{i}/final_total_cost_usd": final_cost} if has_cost else {}),
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
    for reaction_key in reaction_keys:
        run.summary[f"full_ground_truth/{reaction_key}/count"] = len(
            full_gt_indices_by_reaction[reaction_key]
        )
        run.summary[f"full_ground_truth/{reaction_key}/indices"] = ",".join(
            str(x) for x in full_gt_indices_by_reaction[reaction_key]
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
