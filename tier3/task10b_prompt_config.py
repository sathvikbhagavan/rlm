"""Prompt-facing configuration for tier3 task10b mechanism questions."""

from __future__ import annotations

REACTION_KEYS = (
    "keto_alpha_alkylation",
    "base_catalyzed_transesterification",
    "grignard_carbonyl_addition_two_stage",
    "staudinger_reduction_without_duplicate_n2_step",
    "alpha_ketone_bromination",
)

TASK_LABELS = {
    "keto_alpha_alkylation": "Keto alpha-alkylation",
    "base_catalyzed_transesterification": "Base-catalyzed transesterification",
    "grignard_carbonyl_addition_two_stage": "Grignard carbonyl addition",
    "staudinger_reduction_without_duplicate_n2_step": "Staudinger reduction",
    "alpha_ketone_bromination": "Alpha-ketone bromination",
}

TASK_DESCRIPTIONS = {
    "keto_alpha_alkylation": (
        "A reaction matches when an acidic alpha carbon next to a carbonyl or nitrile "
        "is deprotonated to an enolate or carbanion and then alkylated by an alkyl "
        "halide or sulfonate electrophile. The reported product should contain the "
        "new C-C bond at the alpha position."
    ),
    "base_catalyzed_transesterification": (
        "A reaction matches when an ester reacts with an alcohol under basic conditions "
        "through alkoxide addition, tetrahedral intermediate collapse, and protonation "
        "of the leaving alkoxide to give a different ester. The reported product side "
        "may omit alcohol coproducts or salts."
    ),
    "grignard_carbonyl_addition_two_stage": (
        "A reaction matches when an organomagnesium halide adds to an aldehyde, ketone, "
        "ester, or related carbonyl electrophile to form a new C-C bond, followed by "
        "the carbonyl-addition/workup outcome represented by the mechanism. Count "
        "reactions whose reported product is consistent with that addition product."
    ),
    "staudinger_reduction_without_duplicate_n2_step": (
        "A reaction matches when an organic azide reacts with a trivalent phosphine and "
        "water through a Staudinger reduction pathway to give the corresponding amine "
        "and phosphine oxide. Do not count azide-alkyne cycloadditions or catalytic "
        "hydrogenations."
    ),
    "alpha_ketone_bromination": (
        "A reaction matches when a ketone or related carbonyl compound enolizes or forms "
        "an alpha carbanion/enol and then reacts with molecular bromine to install "
        "bromine at the alpha carbon. The reported product should be the alpha-brominated "
        "carbonyl compound."
    ),
}

TASK_EVALUATION_GUIDANCE = {
    "keto_alpha_alkylation": """
    Matching rule:
    - Count reactions where an alpha carbon adjacent to a carbonyl or nitrile is alkylated.
    - The reported product side may omit halide, sulfonate, salts, base, or other coproducts.
    - Count a reaction if every reported product is consistent with the final mechanism state and at least one reported product is the direct alpha-alkylation product.
    - Do not count unrelated alkylations where the carbonyl/nitrile substrate is only a spectator.
""",
    "base_catalyzed_transesterification": """
    Matching rule:
    - Count reactions where an ester is converted into a different ester by alcohol/alkoxide exchange under basic conditions.
    - The reported product side may omit the displaced alcohol, alkoxide, salts, or other coproducts.
    - Count a reaction if every reported product is consistent with the final mechanism state and at least one reported product is the direct transesterification product.
    - Do not count ester hydrolysis, amidation, or acid-catalyzed transesterification.
""",
    "grignard_carbonyl_addition_two_stage": """
    Matching rule:
    - Count reactions where an organomagnesium halide adds to a carbonyl electrophile to form a new C-C bond.
    - The reported product side may omit magnesium salts or workup coproducts.
    - Count a reaction if every reported product is consistent with the final mechanism state and at least one reported product is the direct carbonyl-addition product.
    - Do not count Grignard preparation or organometallic reactions where the carbonyl is a spectator.
""",
    "staudinger_reduction_without_duplicate_n2_step": """
    Matching rule:
    - Count reactions where an organic azide is reduced by a phosphine/water Staudinger pathway.
    - The reported product side may omit nitrogen gas, phosphine oxide, water, or other coproducts.
    - Count a reaction if every reported product is consistent with the final mechanism state and at least one reported product is the direct amine product.
    - Do not count azide-alkyne cycloaddition or catalytic hydrogenation.
""",
    "alpha_ketone_bromination": """
    Matching rule:
    - Count reactions where bromine is installed at the alpha carbon of a ketone or related carbonyl compound.
    - The reported product side may omit hydrogen bromide, bromide, acids, or other coproducts.
    - Count a reaction if every reported product is consistent with the final mechanism state and at least one reported product is the direct alpha-brominated product.
    - Do not count aromatic bromination, alkene bromination, or benzylic bromination.
""",
}


def build_task10b_question(reaction_key: str, *, allow_code: bool) -> str:
    code_guidance = ""
    if allow_code:
        code_guidance = """
    - Use RDKit for parsing reactions and programmatic classification.
    - DO NOT assume/simulate output of the code. Wait for the code to get executed and only then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.
    - DO NOT do `exit()` in the code in any case.
"""

    return f"""
    Context: You are given a large string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Task:
    Return all reaction indices that match this reaction type:
    - {TASK_LABELS[reaction_key]}

    Description:
    - {TASK_DESCRIPTIONS[reaction_key]}

    Guidance:
    - Treat reactants and reagents as the starting pool of molecules.
    - Use reaction-level reasoning to decide whether the actual products can arise from the mechanism described.
    - Handle multi-component sides separated by dots (.).
{TASK_EVALUATION_GUIDANCE[reaction_key]}{code_guidance}
    - Skip malformed reactions and matching failures.

    Output format:
    - Report INDICES separated by commas.
    - Do not include additional text, quotes, punctuation, or formatting.
    - If no matching reaction exists, return -1.
    """
