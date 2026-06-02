"""Shared fixed Task-1 products and ground-truth indices.

These cases were generated with:
- dataset: reactionSmilesFigShareUSPTO2023_cleaned.txt
- seed: 42
"""

from typing import Final


TASK1_HARDCODED_PRODUCTS: Final[list[str]] = [
    "Cc1nn(-c2ccc(S(C)(=O)=O)cn2)c(O)c1-c1ccc(C#N)cn1",
    "C[C@H]1NC/C=C/[C@H](O)[C@@H]2CC[C@H]2CN2C[C@@]3(CCCc4cc(Cl)ccc43)COc3ccc(cc32)S(=O)(=O)NC1=O",
    "CC(C)(C)OC(=O)c1ccnc(-c2ccc3nccc(N)c3c2)n1",
    "C=C1CCN2C[C@H](C1)n1cc(C(=O)NCc3c(F)cc(F)cc3F)c(=O)c(OCc3ccccc3)c1C2=O",
    "FC(F)(F)C(F)(F)COc1ccc(Br)nc1",
    "COc1c(C(=O)O)ccnc1Cl",
    "O=C1CC(=O)C2(CCC2)CN1",
    "CCOC(=O)c1cc(OC)n(-c2ccc(Br)cc2)n1",
    "Cc1cc(-c2ncnc3[nH]c(-c4ccc(N5CCN(CC6CCN(c7ccc(N8CCC(=O)NC8=O)cc7)CC6)C(=O)C5)cc4)cc23)ccc1[C@@H](C)NC(=O)c1nc(C(C)(C)C)no1",
    "CC(C)(C)OC(=O)N1CCC(c2ccc(F)cn2)CC1",
]

TASK1_HARDCODED_GROUND_TRUTH_INDICES: Final[list[list[int]]] = [
    [104086],
    [3571],
    [38737],
    [7110, 115648],
    [22265, 22280],
    [10517, 10525, 116709],
    [89821],
    [19565, 114983],
    [15683],
    [85227, 117210],
]


if len(TASK1_HARDCODED_PRODUCTS) != len(TASK1_HARDCODED_GROUND_TRUTH_INDICES):
    raise ValueError(
        "TASK1_HARDCODED_PRODUCTS and TASK1_HARDCODED_GROUND_TRUTH_INDICES must match in length."
    )
