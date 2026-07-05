"""Hardcoded longest-chain ground truth for tier4 task12.

Chains were selected from reactionSmilesFigShareUSPTO2023_cleaned.txt using
exact canonical SMILES product-to-reactant linking with index-asc DAG ordering
(no global molecule-frequency filter). Targets were chosen so each longest
chain has 8 reactions and fits in context windows of 100 and 500.
"""

TASK12_TOTAL_REACTIONS = 122456
TASK12_DAG_MODE = "index_asc"
TASK12_GROUND_TRUTH_DEFINITION = (
    "longest ordered distinct reaction indices [r_0, ..., r_k] with r_i < r_{i+1}, "
    "each consecutive pair linked by exact canonical SMILES equality between at least "
    "one product of r_i and one reactant of r_{i+1}, final reaction producing the "
    "target product, considering only reactions present in the provided context; "
    "tie-break equal-length chains lexicographically"
)

# Two fixed target-product questions.
FIXED_TARGET_PRODUCTS: list[str] = [
    "Cc1nc2[nH]c(C(=O)O)cc2c(F)c1F",
    "Cc1ncc2[nH]c(C(=O)O)c(C)c2c1F",
]

HARDCODED_GT_LONGEST_CHAIN: dict[str, tuple[int, ...]] = {
    "Cc1nc2[nH]c(C(=O)O)cc2c(F)c1F": (
        573, 574, 575, 576, 654, 655, 656, 657,
    ),
    "Cc1ncc2[nH]c(C(=O)O)c(C)c2c1F": (
        525, 735, 736, 737, 777, 778, 779, 780,
    ),
}


def chain_indices_for_product(target_product: str) -> set[int]:
    chain = HARDCODED_GT_LONGEST_CHAIN.get(target_product)
    if chain is None:
        raise KeyError(f"No hardcoded longest chain for target_product={target_product}")
    return set(chain)
