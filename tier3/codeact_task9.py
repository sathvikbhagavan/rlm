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
    build_reaction_query,
    build_retriever,
    extract_response_text,
    load_lines,
    parse_indices,
    precision_recall_f1,
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
        project_name="CodeAct-Task9",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAct task 9 evaluation.")
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
    tracer = get_tracer("codeact-task9")
    lines = load_lines()
    context = "\n".join(lines)
    rng = random.Random(SEED)
    reaction_keys = list(NAMED_REACTIONS_SMIRKS.keys())
    run_session_id = f"codeact-task9-{uuid.uuid4()}"

    full_gt_indices_by_reaction: dict[str, list[int]] = {}
    for reaction_key in reaction_keys:
        query_reaction = build_reaction_query(NAMED_REACTIONS_SMIRKS[reaction_key])
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
        project="CodeAct-Task9",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": SEED,
            "context_size": context_size,
            "ground_truth_fraction_per_context": ground_truth_fraction_per_context,
            "retriever_name": retriever_name,
            "reasoning_effort": REASONING_EFFORT,
            "task_description": "Return reaction indices for named reaction patterns.",
            "NAMED_REACTIONS_SMIRKS": NAMED_REACTIONS_SMIRKS,
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
        reaction_label = NAMED_REACTIONS_LABELS[reaction_key]
        reaction_description = NAMED_REACTIONS_DESCRIPTIONS[reaction_key]
        reaction_smirks = NAMED_REACTIONS_SMIRKS[reaction_key]
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

        with tracer.start_as_current_span(f"codeact_task9_sample_{i}") as sample_span:
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
                tags=["codeact", "sample", "task9_named_reactions"],
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
