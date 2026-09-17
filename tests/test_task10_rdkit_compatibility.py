from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIER3 = ROOT / "tier3"
if str(TIER3) not in sys.path:
    sys.path.insert(0, str(TIER3))

reaction_line_matches_mechanism = importlib.import_module(
    "task10_mechanism_evaluator"
).reaction_line_matches_mechanism


HYPERVALENT_PHOSPHORUS_MITSUNOBU = (
    "Cc1cnc(C(=O)N2CCC(C#N)(c3ccccc3)CC2)cc1C(=O)N1CC=CC(O)C1."
    "Oc1cc(F)cc(F)c1Br>CC(C)(C#N)/N=N/C(C)(C)C#N."
    "CCCCP(=CC#N)(CCCC)CCCC.C[Si](C)(C)[SiH]([Si](C)(C)C)[Si](C)(C)C."
    "Cc1ccccc1>Cc1cnc(C(=O)N2CCC(C#N)(c3ccccc3)CC2)cc1C(=O)N1CC=CC("
    "Oc2cc(F)cc(F)c2Br)C1"
)


def test_mitsunobu_membership_matches_frozen_rdkit_2022_semantics() -> None:
    assert reaction_line_matches_mechanism(
        HYPERVALENT_PHOSPHORUS_MITSUNOBU,
        "mitsunobu_reaction_family",
    )
