from __future__ import annotations

from dataclasses import dataclass

from tier1.task1_hardcoded_cases import (
    TASK1_HARDCODED_GROUND_TRUTH_INDICES,
    TASK1_HARDCODED_PRODUCTS,
)
from tier2.task5_hardcoded_ground_truth import TASK5_HARDCODED_GROUND_TRUTH_INDICES
from tier3.task10_hardcoded_ground_truth import (
    TASK10_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION,
)
from tier3.task10_prompt_config import build_task10_question


@dataclass(frozen=True)
class MapReduceQuestion:
    question_id: str
    benchmark_task: str
    question: str
    ground_truth_indices: frozenset[int]


def task1_question(product: str) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Find all the indices of the reactions for the following PRODUCT
    (and not the reactants/reagents): {product}

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If the product is not found, report: -1
"""


def task5_question(weight_threshold: int, ring_x: int) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Find all the indices of the reactions that satisfy BOTH conditions:
      1) weight(heaviest product) - weight(heaviest reactant) > {weight_threshold} Da
      2) max over all pairs [rings(product_component) - rings(reactant_component)] == {ring_x}

    Guidance:
    - You may use RDKit functions if needed (for example, Chem.MolFromSmiles, Descriptors.MolWt, rdMolDescriptors.CalcNumRings).
    - "heaviest" means the largest molecular weight among dot-separated molecules on that side.
    - For ring condition, split each side by dot and compute ring count for each valid component.
    - For each reaction, compute ALL pairwise ring deltas:
      rings(product_component) - rings(reactant_component)
      and use the maximum of those deltas.
    - Ignore reagents (middle field).
    - For each side (reactants/products), ignore invalid or empty dot-separated molecules.
    - Skip a reaction only if reactant side or product side has no valid molecules left after filtering.

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If no matching reaction exists, report: -1
"""


def selected_questions() -> dict[str, MapReduceQuestion]:
    product = TASK1_HARDCODED_PRODUCTS[0]
    task5_key = (100, 1)
    task10_key = "wittig_olefination"
    questions = (
        MapReduceQuestion(
            question_id="tier1-task1-q01",
            benchmark_task="tier1/task1",
            question=task1_question(product),
            ground_truth_indices=frozenset(TASK1_HARDCODED_GROUND_TRUTH_INDICES[0]),
        ),
        MapReduceQuestion(
            question_id="tier2-task5-weight100-ring1",
            benchmark_task="tier2/task5",
            question=task5_question(*task5_key),
            ground_truth_indices=frozenset(TASK5_HARDCODED_GROUND_TRUTH_INDICES[task5_key]),
        ),
        MapReduceQuestion(
            question_id="tier3-task10-wittig",
            benchmark_task="tier3/task10",
            question=build_task10_question(task10_key, allow_code=False),
            ground_truth_indices=frozenset(
                TASK10_HARDCODED_GROUND_TRUTH_INDICES_BY_REACTION[task10_key]
            ),
        ),
    )
    return {question.question_id: question for question in questions}


def mapper_prompt(*, question: str, chunk_text: str) -> str:
    return f"""
You are one independent mapper in a fixed, non-recursive map-and-union baseline.
Examine only the assigned reaction chunk. Do not infer or report an index that is
not present in this chunk. Return every matching index from this chunk.

<context>
{chunk_text}
</context>
<question>
{question}
</question>

Return only comma-separated integer indices, or -1 when this chunk has no match.
""".strip()
