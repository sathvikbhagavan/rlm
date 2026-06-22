"""Prompt-facing configuration for tier3 task10 mechanism questions."""

from __future__ import annotations

REACTION_KEYS = (
    "ester_hydrolysis_deprotection_with_oh",
    "mitsunobu_reaction_family",
    "wittig_olefination",
    "knoevenagel_aldol_condensation",
    "azide_alkyne_huisgen_cycloaddition",
)

TASK_LABELS = {
    "ester_hydrolysis_deprotection_with_oh": (
        "Ester hydrolysis / carboxylic ester deprotection with hydroxide"
    ),
    "mitsunobu_reaction_family": "Mitsunobu reaction family",
    "wittig_olefination": "Wittig olefination",
    "knoevenagel_aldol_condensation": "Knoevenagel condensation / aldol dehydration",
    "azide_alkyne_huisgen_cycloaddition": "Azide-alkyne Huisgen cycloaddition",
}

TASK_DESCRIPTIONS = {
    "ester_hydrolysis_deprotection_with_oh": (
        "A reaction matches when a carboxylic ester reacts with hydroxide or an alkali "
        "metal hydroxide through the standard base-promoted acyl substitution mechanism: "
        "hydroxide adds to the ester carbonyl to form a tetrahedral intermediate, then "
        "the alkoxy group leaves to give the carboxylic acid product and the alkoxide "
        "derived from the ester substituent. Dataset product sides may omit coproducts, "
        "but any reported product must be consistent with the mechanism. Treat reactants "
        "and reagents together as the available starting state."
    ),
    "mitsunobu_reaction_family": (
        "A reaction matches when an alcohol is converted into a substitution product "
        "under Mitsunobu conditions, typically involving a triaryl/alkyl phosphine and "
        "an azo dicarboxylate reagent such as DEAD or DIAD. Count aryl ether, imide, "
        "sulfonamide, ester, thioether, amide, and amine variants when the reported "
        "product is the substitution product expected from the alcohol and nucleophile."
    ),
    "wittig_olefination": (
        "A reaction matches when a phosphorus ylide or equivalent phosphorane reacts "
        "with an aldehyde or ketone to form an alkene through a Wittig olefination. "
        "The reported product should include the newly formed C=C product and be "
        "consistent with loss of the corresponding phosphine oxide byproduct."
    ),
    "knoevenagel_aldol_condensation": (
        "A reaction matches when an active methylene or related carbon acid adds to an "
        "aldehyde or ketone and then dehydrates to form an alkene, typically an "
        "electron-poor or conjugated alkene product. Count Knoevenagel-style and "
        "aldol-dehydration variants when the reported product is the condensation "
        "product rather than an unrelated carbonyl transformation."
    ),
    "azide_alkyne_huisgen_cycloaddition": (
        "A reaction matches when an organic azide and an alkyne undergo Huisgen "
        "cycloaddition to form a triazole ring. Count reactions where the reported "
        "product is the cycloaddition product, including substituted triazoles, and do "
        "not count reactions where the azide or alkyne is only a spectator."
    ),
}

TASK_EVALUATION_GUIDANCE = {
    "ester_hydrolysis_deprotection_with_oh": """
    Matching rule:
    - Count reactions where a carboxylic ester is hydrolyzed under hydroxide or alkali-metal hydroxide conditions.
    - The full mechanism generates the corresponding carboxylic acid product and the alkoxide product from the ester substituent.
    - The reported product side may omit coproducts, salts, or byproducts.
    - Count a reaction if every reported product is consistent with the final mechanism state and at least one reported product is a direct product of ester hydrolysis.
    - Do not count reactions where the ester is only a spectator.
    - Do not count acid-catalyzed ester hydrolysis, transesterification, amide hydrolysis, or unrelated deprotections.
""",
    "mitsunobu_reaction_family": """
    Matching rule:
    - Count reactions with the characteristic Mitsunobu reagent pattern and alcohol substitution outcome.
    - Count aryl ether, imide, sulfonamide, ester, thioether, amide, and amine variants when the product forms by replacing the alcohol oxygen with the nucleophile.
    - The reported product side may omit phosphine oxide, hydrazine dicarboxylate, salts, or other coproducts.
    - Count a reaction if every reported product is consistent with the final mechanism state and at least one reported product is the direct substitution product.
    - Do not count reactions where phosphine/azo reagents or alcohols are spectators.
""",
    "wittig_olefination": """
    Matching rule:
    - Count reactions where a phosphorus ylide or phosphorane reacts with an aldehyde or ketone to form an alkene.
    - The reported product side may omit phosphine oxide or other coproducts.
    - Count a reaction if every reported product is consistent with the final mechanism state and at least one reported product is the direct alkene product.
    - Do not count Horner-Wadsworth-Emmons, Julia, simple aldol, or other olefinations unless they follow the Wittig ylide-carbonyl pattern.
""",
    "knoevenagel_aldol_condensation": """
    Matching rule:
    - Count reactions where a carbon acid or active methylene component condenses with an aldehyde or ketone and dehydrates to an alkene product.
    - The reported product side may omit water, base, salts, or other coproducts.
    - Count a reaction if every reported product is consistent with the final mechanism state and at least one reported product is the direct condensation product.
    - Do not count reactions where the carbonyl or active methylene compound is only a spectator.
""",
    "azide_alkyne_huisgen_cycloaddition": """
    Matching rule:
    - Count reactions where an azide and an alkyne form a triazole by cycloaddition.
    - The reported product side may omit catalysts, salts, or spectator reagents.
    - Count a reaction if every reported product is consistent with the final mechanism state and at least one reported product is the direct triazole product.
    - Do not count azide substitutions, azide reductions, alkyne functionalizations, or click-condition reactions without triazole formation.
""",
}


def build_task10_question(reaction_key: str, *, allow_code: bool) -> str:
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
