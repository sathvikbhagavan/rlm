import argparse
import asyncio
import os
import random
import uuid

import wandb
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
    ground_truth_indices,
    load_lines,
    parse_indices,
    precision_recall_f1,
)
from rlm.tracing import get_tracer, init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
MODEL_NAME = "x-ai/grok-4-fast"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 1200.0
SEED = 42
CONTEXT_SIZE = 100
GROUND_TRUTH_FRACTION_PER_CONTEXT = 0.2
RETRIEVER_NAME = "random"
MAX_OUTPUT_TOKENS = 50_000
REASONING_EFFORT = "high"
MAX_ITERATIONS = 8
# os.environ["WANDB_MODE"] = "disabled"

BOC_SMIRKS: dict[str, str] = {
    "boc_primary_amine_deprotection": "[C;H3;D1;+0]-[C;H0;D4;+0](-[C;H3;D1;+0])(-[C;H3;D1;+0])-[O;H0;D2;+0]-[C;H0;D3;+0](=[O;H0;D1;+0])-[#7;H1;+0:1]>>[#7;H2;+0:1]",
    "boc_secondary_amine_deprotection": "[C;H3;D1;+0]-[C;H0;D4;+0](-[C;H3;D1;+0])(-[C;H3;D1;+0])-[O;H0;D2;+0]-[C;H0;D3;+0](=[O;H0;D1;+0])-[#7;H0;+0:1]>>[#7;H1;+0:1]",
    "boc_amine_protection_of_secondary_amine": "[C;H3;D1;+0:1]-[C;H0;D4;+0:2](-[C;H3;D1;+0:3])(-[C;H3;D1;+0:4])-[O;H0;D2;+0:5]-[C;H0;D3;+0:6](=[O;H0;D1;+0:7])-[O;H0;D2;+0].[#7;H1;+0:8]>>[C;H3;D1;+0:1]-[C;H0;D4;+0:2](-[C;H3;D1;+0:3])(-[C;H3;D1;+0:4])-[O;H0;D2;+0:5]-[C;H0;D3;+0:6](=[O;H0;D1;+0:7])-[#7;H0;+0:8]",
    "boc_amine_protection_of_primary_amine": "[C;H3;D1;+0:1]-[C;H0;D4;+0:2](-[C;H3;D1;+0:3])(-[C;H3;D1;+0:4])-[O;H0;D2;+0:5]-[C;H0;D3;+0:6](=[O;H0;D1;+0:7])-[O;H0;D2;+0].[#7;H2;+0:8]>>[C;H3;D1;+0:1]-[C;H0;D4;+0:2](-[C;H3;D1;+0:3])(-[C;H3;D1;+0:4])-[O;H0;D2;+0:5]-[C;H0;D3;+0:6](=[O;H0;D1;+0:7])-[#7;H1;+0:8]"
}

BOC_LABELS: dict[str, str] = {
    "boc_primary_amine_deprotection": "BOC deprotection of primary amine",
    "boc_secondary_amine_deprotection": "BOC deprotection of secondary amine",
    "boc_amine_protection_of_secondary_amine": "BOC amine protection of secondary amine",
    "boc_amine_protection_of_primary_amine": "BOC amine protection of primary amine"
}

BOC_DESCRIPTIONS: dict[str, str] = {
    "boc_primary_amine_deprotection": "Acid-mediated removal of a tert-butyloxycarbonyl (Boc) protecting group from a nitrogen atom, regenerating a primary amine. The reactant contains the full Boc group — a tert-butyl moiety consisting of a quaternary carbon with no hydrogens and four connections bearing three methyl groups, connected through a neutral oxygen with no hydrogens and two connections to a carbonyl carbon with no hydrogens and three connections, double-bonded to an oxygen with no hydrogens and one connection. The carbonyl is bonded to the protected nitrogen which has one hydrogen and is neutral. The entire Boc group is unmapped and fully cleaved off, and the nitrogen gains a hydrogen (going from one to two), regenerating a free primary amine.",
    "boc_secondary_amine_deprotection": "Acid-mediated removal of a tert-butyloxycarbonyl (Boc) protecting group from a nitrogen atom, regenerating a secondary amine. The reactant contains the full Boc group — a tert-butyl moiety consisting of a quaternary carbon with no hydrogens and four connections bearing three methyl groups, connected through a neutral oxygen with no hydrogens and two connections to a carbonyl carbon with no hydrogens and three connections, double-bonded to an oxygen with no hydrogens and one connection. The carbonyl is bonded to the protected nitrogen which has no hydrogens and is neutral. The entire Boc group is unmapped and fully cleaved off, and the nitrogen gains a hydrogen (going from zero to one), regenerating a free secondary amine.",
    "boc_amine_protection_of_secondary_amine": "Protection of a secondary amine with a tert-butyloxycarbonyl (Boc) group. The reactant side contains a Boc reagent — a tert-butyl moiety (quaternary carbon bearing three methyl groups) connected through an oxygen to a carbonyl, which has a second oxygen acting as the leaving group (unmapped). The secondary amine, bearing one hydrogen on nitrogen, attacks the carbonyl carbon, displacing the leaving group oxygen. In the product, the nitrogen loses its hydrogen (going from H1 to H0) and is now bonded to the Boc carbonyl, forming a carbamate linkage. The full Boc group — tert-butyl, oxygen, and carbonyl — is preserved intact in the product.",
    "boc_amine_protection_of_primary_amine": "Protection of a primary amine with a tert-butyloxycarbonyl (Boc) group. The reactant side contains a Boc reagent — a tert-butyl moiety consisting of a quaternary carbon with no hydrogens and four connections, bearing three methyl groups each with three hydrogens and one connection — connected through a neutral oxygen with no hydrogens and two connections to a carbonyl carbon with no hydrogens and three connections, which is double-bonded to an oxygen with no hydrogens and one connection. A second unmapped oxygen on the carbonyl acts as the leaving group. The primary amine nitrogen, bearing two hydrogens and neutral charge, attacks the carbonyl carbon, displacing the leaving group oxygen. In the product, the nitrogen loses one hydrogen (going from two to one), forming a new bond to the Boc carbonyl, while the full Boc group is preserved intact."
}


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
        project_name="CodeAct-Task8",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAct task 8 evaluation.")
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
    tracer = get_tracer("codeact-task8")
    lines = load_lines()
    context = "\n".join(lines)
    rng = random.Random(SEED)
    reaction_keys = list(BOC_SMIRKS.keys())
    run_session_id = f"codeact-task8-{uuid.uuid4()}"

    full_gt_indices_by_reaction: dict[str, list[int]] = {}
    for reaction_key in reaction_keys:
        query_reaction = build_reaction_query(BOC_SMIRKS[reaction_key])
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
        project="CodeAct-Task8",
        config={
            "MODEL_NAME": model_name,
            "dataset_path": DATASET_PATH,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": SEED,
            "context_size": context_size,
            "ground_truth_fraction_per_context": ground_truth_fraction_per_context,
            "retriever_name": retriever_name,
            "reasoning_effort": REASONING_EFFORT,
            "task_description": "Return reaction indices for BOC-protection/deprotection subtypes.",
            "BOC_SMIRKS": BOC_SMIRKS,
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
        reaction_label = BOC_LABELS[reaction_key]
        reaction_description = BOC_DESCRIPTIONS[reaction_key]
        reaction_smirks = BOC_SMIRKS[reaction_key]
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

        with tracer.start_as_current_span(f"codeact_task8_sample_{i}") as sample_span:
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
                tags=["codeact", "sample", "task8_boc"],
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
