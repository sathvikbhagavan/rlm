import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from rdkit import Chem
except ImportError:
    Chem = None  # type: ignore[assignment]

try:
    from rxnmapper import RXNMapper
except ImportError:
    RXNMapper = None  # type: ignore[assignment]


DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
DEFAULT_OUTPUT_PATH = "/home/bhagavan/rlms/rlm/tier3/negative_reactions.jsonl"
DEFAULT_EXAMPLES_PER_CATEGORY = 10
DEFAULT_SEED = 42
RXNMAPPER_CONFIDENCE_THRESHOLD = 0.8

CORRUPTION_TYPES = (
    "structurally_impossible",
    "conservation_violation",
    "product_swapping",
    "handcrafted_wrong_reagent_or_product",
)

# Single-atom SMILES for elements that are very rarely (effectively never) the
# sole element injected into an organic reaction product.  Used to corrupt a
# reaction by adding a product atom whose element is absent from the whole
# equation, making the violation unambiguously detectable.
FOREIGN_ELEMENT_SMILES: tuple[str, ...] = (
    "[Se]", "[Te]", "[As]", "[Ge]", "[Sn]", "[Bi]", "[Sb]", "[Zr]",
)

RANDOM_REAGENT_CANDIDATES = (
    "O",
    "ClCCl",
    "CCN(CC)CC",
    "C1CCCO1",
    "CS(=O)C",
    "C(C)(=O)O",
    "CO[H]",
)


@dataclass
class NegativeReaction:
    id: int
    corruption_type: str
    source_index_1: int
    source_index_2: int | None
    original_reaction_1: str
    original_reaction_2: str | None
    corrupted_reaction: str
    note: str


def load_dataset(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_reaction(reaction: str) -> tuple[str, str, str] | None:
    parts = reaction.split(">")
    if len(parts) != 3:
        return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def split_side(side: str) -> list[str]:
    return [component for component in side.split(".") if component]


def join_side(components: list[str]) -> str:
    return ".".join(component for component in components if component)


def reaction_is_parseable(reaction: str) -> bool:
    parsed = parse_reaction(reaction)
    if parsed is None:
        return False
    reactants, _, products = parsed
    if not reactants or not products:
        return False
    if Chem is None:
        return True
    for component in split_side(reactants) + split_side(products):
        if Chem.MolFromSmiles(component) is None:
            return False
    return True


def build_structurally_impossible(
    indexed_reaction: tuple[int, str], rng: random.Random
) -> NegativeReaction | None:
    idx, reaction = indexed_reaction
    parsed = parse_reaction(reaction)
    if parsed is None:
        return None
    reactants, reagents, products = parsed
    product_components = split_side(products)
    if not product_components:
        return None

    mode = rng.choice(("valence_violation", "broken_syntax"))
    target_i = rng.randrange(len(product_components))
    original_product = product_components[target_i]

    if mode == "valence_violation":
        # Carbonyl carbon with two extra substituents forces impossible valence.
        product_components[target_i] = "CC(=O)(N)(Cl)F"
    else:
        if len(original_product) < 4:
            return None
        broken = original_product[:-1]
        if broken.count("(") > 0:
            broken = broken.replace(")", "", 1)
        product_components[target_i] = broken

    corrupted = f"{reactants}>{reagents}>{join_side(product_components)}"
    return NegativeReaction(
        id=-1,
        corruption_type="structurally_impossible",
        source_index_1=idx,
        source_index_2=None,
        original_reaction_1=reaction,
        original_reaction_2=None,
        corrupted_reaction=corrupted,
        note=f"Injected {mode} in product SMILES",
    )


def _elements_in_smiles_list(smiles_list: list[str]) -> set[str]:
    """Return the set of element symbols present across a list of SMILES (requires RDKit)."""
    elements: set[str] = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            for atom in mol.GetAtoms():
                elements.add(atom.GetSymbol())
    return elements


def _foreign_element(
    all_smiles: list[str], rng: random.Random
) -> tuple[str, str] | None:
    """Return (foreign_smiles, element_symbol) for an element absent from all_smiles."""
    present = _elements_in_smiles_list(all_smiles)
    candidates = []
    for smi in FOREIGN_ELEMENT_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        symbol = mol.GetAtomWithIdx(0).GetSymbol()
        if symbol not in present:
            candidates.append((smi, symbol))
    if not candidates:
        return None
    return rng.choice(candidates)


def build_conservation_violation(
    indexed_reaction: tuple[int, str], rng: random.Random
) -> NegativeReaction | None:
    """Inject a foreign element (absent from the whole equation) into the products.

    Since the dataset reactions are unbalanced, atom-count checks are unreliable.
    The only unambiguous signal is an element in the products that was never present
    anywhere in the equation (reactants + reagents + products).  Two injection modes:

    * append_foreign_product  — add a single-atom molecule of the foreign element
                                as an extra product component.
    * substitute_terminal     — replace a terminal non-ring atom in a product
                                molecule with the foreign element.
    """
    if Chem is None:
        return None
    idx, reaction = indexed_reaction
    parsed = parse_reaction(reaction)
    if parsed is None:
        return None
    reactants, reagents, products = parsed
    product_components = split_side(products)
    if not product_components:
        return None

    all_smiles = split_side(reactants) + split_side(reagents) + split_side(products)
    foreign = _foreign_element(all_smiles, rng)
    if foreign is None:
        return None
    foreign_smi, foreign_symbol = foreign

    mode = rng.choice(("append_foreign_product", "substitute_terminal"))

    if mode == "substitute_terminal":
        target_i = rng.randrange(len(product_components))
        mol = Chem.MolFromSmiles(product_components[target_i])
        if mol is None or mol.GetNumAtoms() < 2:
            mode = "append_foreign_product"
        else:
            removable = [
                atom.GetIdx()
                for atom in mol.GetAtoms()
                if atom.GetDegree() == 1 and not atom.IsInRing()
            ]
            if not removable:
                mode = "append_foreign_product"
            else:
                rw = Chem.RWMol(mol)
                atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(foreign_symbol)
                rw.GetAtomWithIdx(rng.choice(removable)).SetAtomicNum(atomic_num)
                try:
                    Chem.SanitizeMol(rw)
                    product_components[target_i] = Chem.MolToSmiles(rw)
                except Exception:
                    mode = "append_foreign_product"

    if mode == "append_foreign_product":
        product_components.append(foreign_smi)
        note = f"Appended foreign element {foreign_symbol!r} as spurious product"
    else:
        note = f"Substituted terminal product atom with foreign element {foreign_symbol!r}"

    corrupted = f"{reactants}>{reagents}>{join_side(product_components)}"
    return NegativeReaction(
        id=-1,
        corruption_type="conservation_violation",
        source_index_1=idx,
        source_index_2=None,
        original_reaction_1=reaction,
        original_reaction_2=None,
        corrupted_reaction=corrupted,
        note=note,
    )


def build_product_swap(
    indexed_reaction_1: tuple[int, str], indexed_reaction_2: tuple[int, str]
) -> NegativeReaction | None:
    idx1, rxn1 = indexed_reaction_1
    idx2, rxn2 = indexed_reaction_2
    parsed_1 = parse_reaction(rxn1)
    parsed_2 = parse_reaction(rxn2)
    if parsed_1 is None or parsed_2 is None:
        return None
    reactants_1, reagents_1, products_1 = parsed_1
    _, _, products_2 = parsed_2
    if products_1 == products_2:
        return None
    corrupted = f"{reactants_1}>{reagents_1}>{products_2}"
    return NegativeReaction(
        id=-1,
        corruption_type="product_swapping",
        source_index_1=idx1,
        source_index_2=idx2,
        original_reaction_1=rxn1,
        original_reaction_2=rxn2,
        corrupted_reaction=corrupted,
        note="Swapped product side with product from a different real reaction",
    )


def build_handcrafted_wrong(
    indexed_reaction: tuple[int, str], rng: random.Random
) -> NegativeReaction | None:
    idx, reaction = indexed_reaction
    parsed = parse_reaction(reaction)
    if parsed is None:
        return None
    reactants, reagents, products = parsed
    reactant_components = split_side(reactants)
    reagent_components = split_side(reagents)
    product_components = split_side(products)
    if not reactant_components or not product_components:
        return None

    mode = rng.choice(("add_different_random_reagent", "wrong_product_from_reagent"))
    if mode == "add_different_random_reagent":
        existing_reagents = set(reagent_components)
        candidates = [r for r in RANDOM_REAGENT_CANDIDATES if r not in existing_reagents]
        if not candidates:
            return None
        extra_reagent = rng.choice(candidates)
        new_reagents = reagent_components + [extra_reagent]
        corrupted = f"{reactants}>{join_side(new_reagents)}>{products}"
        note = f"Added different random reagent '{extra_reagent}'"
    else:
        if reagent_components:
            wrong_product = rng.choice(reagent_components)
            note = "Replaced product with a reagent molecule"
        else:
            wrong_product = rng.choice(reactant_components)
            note = "Replaced product with an unchanged reactant molecule"
        corrupted = f"{reactants}>{reagents}>{wrong_product}"

    return NegativeReaction(
        id=-1,
        corruption_type="handcrafted_wrong_reagent_or_product",
        source_index_1=idx,
        source_index_2=None,
        original_reaction_1=reaction,
        original_reaction_2=None,
        corrupted_reaction=corrupted,
        note=note,
    )


def allocate_counts(per_category: int) -> dict[str, int]:
    if per_category <= 0:
        raise ValueError("examples_per_category must be > 0.")
    return {key: per_category for key in CORRUPTION_TYPES}


def generate_negatives(
    dataset_path: str,
    examples_per_category: int,
    seed: int,
) -> list[NegativeReaction]:
    rng = random.Random(seed)
    all_reactions = load_dataset(dataset_path)
    indexed = list(enumerate(all_reactions))
    usable = [entry for entry in indexed if reaction_is_parseable(entry[1])]
    if len(usable) < 20:
        raise RuntimeError("Too few parseable reactions found in dataset.")

    target_counts = allocate_counts(examples_per_category)
    negatives: list[NegativeReaction] = []
    detected_conservation_pool = []

    if target_counts.get("conservation_violation", 0) > 0:
        for idx, reaction in indexed:
            hit = detect_conservation_violation(idx, reaction)
            if hit is not None:
                detected_conservation_pool.append(hit)
        rng.shuffle(detected_conservation_pool)
        # Keep all detector-found conservation violations in output JSONL.
        target_counts["conservation_violation"] = len(detected_conservation_pool)

    def add_records(generator_name: str, target_count: int) -> None:
        attempts = 0
        while target_count > 0:
            attempts += 1
            if attempts > 5000:
                raise RuntimeError(f"Failed to generate enough samples for {generator_name}")

            if generator_name == "structurally_impossible":
                record = build_structurally_impossible(rng.choice(usable), rng)
            elif generator_name == "conservation_violation":
                hit = detected_conservation_pool.pop()
                record = NegativeReaction(
                    id=-1,
                    corruption_type="conservation_violation",
                    source_index_1=hit.index,
                    source_index_2=None,
                    original_reaction_1=hit.reaction,
                    original_reaction_2=None,
                    corrupted_reaction=hit.reaction,
                    note=f"Detected in source dataset: {hit.note}",
                )
            elif generator_name == "product_swapping":
                first = rng.choice(usable)
                second = rng.choice(usable)
                if first[0] == second[0]:
                    continue
                record = build_product_swap(first, second)
            else:
                record = build_handcrafted_wrong(rng.choice(usable), rng)

            if record is None:
                continue
            if any(r.corrupted_reaction == record.corrupted_reaction for r in negatives):
                continue
            negatives.append(record)
            target_count -= 1

    for corruption_type in CORRUPTION_TYPES:
        add_records(corruption_type, target_counts[corruption_type])

    for i, record in enumerate(negatives):
        record.id = i
    return negatives


@dataclass
class DetectedReaction:
    index: int
    reaction: str
    corruption_type: str
    note: str


_RXNMAPPER_INSTANCE = None


def _get_rxnmapper():
    global _RXNMAPPER_INSTANCE
    if RXNMapper is None:
        return None
    if _RXNMAPPER_INSTANCE is None:
        _RXNMAPPER_INSTANCE = RXNMapper()
    return _RXNMAPPER_INSTANCE


def detect_structurally_impossible(idx: int, reaction: str) -> DetectedReaction | None:
    """Product SMILES that RDKit cannot parse or fails sanitization."""
    if Chem is None:
        return None
    parsed = parse_reaction(reaction)
    if parsed is None:
        return None
    _, _, products = parsed
    for smi in split_side(products):
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                return DetectedReaction(idx, reaction, "structurally_impossible",
                                        f"Unparseable product SMILES: {smi!r}")
            Chem.SanitizeMol(mol)
        except Exception as exc:
            return DetectedReaction(idx, reaction, "structurally_impossible",
                                    f"Product SMILES fails sanitization ({exc}): {smi!r}")
    return None



def detect_conservation_violation(idx: int, reaction: str) -> DetectedReaction | None:
    """Product contains an element not present in reactants or reagents.

    Atom-count balance is not checked (USPTO reactions are unbalanced), but an
    element appearing in products that is entirely absent from reactants+reagents
    is unambiguously wrong regardless of stoichiometry.
    """
    if Chem is None:
        return None
    parsed = parse_reaction(reaction)
    if parsed is None:
        return None
    reactants, reagents, products = parsed

    input_elements = _elements_in_smiles_list(split_side(reactants) + split_side(reagents))
    if not input_elements:
        return None

    for smi in split_side(products):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in input_elements:
                return DetectedReaction(
                    idx, reaction, "conservation_violation",
                    f"Product {smi!r} contains element '{atom.GetSymbol()}' "
                    f"absent from reactants/reagents",
                )
    return None


def detect_low_confidence_rxnmapper(idx: int, reaction: str) -> DetectedReaction | None:
    """Flag reactions whose RXNMapper confidence is below threshold."""
    mapper = _get_rxnmapper()
    if mapper is None:
        return None
    try:
        mapped = mapper.get_attention_guided_atom_maps([reaction])
    except Exception as exc:
        return DetectedReaction(
            idx,
            reaction,
            "rxnmapper_low_confidence",
            f"RXNMapper failed to map reaction ({exc})",
        )
    if not mapped:
        return DetectedReaction(
            idx,
            reaction,
            "rxnmapper_low_confidence",
            "RXNMapper returned no mapping result",
        )
    confidence = mapped[0].get("confidence")
    if not isinstance(confidence, (int, float)):
        return DetectedReaction(
            idx,
            reaction,
            "rxnmapper_low_confidence",
            "RXNMapper returned missing/invalid confidence",
        )
    if confidence < RXNMAPPER_CONFIDENCE_THRESHOLD:
        return DetectedReaction(
            idx,
            reaction,
            "rxnmapper_low_confidence",
            (
                f"RXNMapper confidence={confidence:.4f} "
                f"< {RXNMAPPER_CONFIDENCE_THRESHOLD:.2f}"
            ),
        )
    return None


# product_swapping cannot be detected from a single reaction in isolation —
# it requires cross-referencing two reactions.
# handcrafted_wrong_reagent_or_product is not detectable: reagents legitimately
# participate in reactions and can appear as products, so matching product SMILES
# against reagent/reactant SMILES produces too many false positives.
DETECTORS = {
    "structurally_impossible": detect_structurally_impossible,
    "conservation_violation": detect_conservation_violation,
    # "rxnmapper_low_confidence": detect_low_confidence_rxnmapper,
}


def scan_dataset(
    dataset_path: str,
    corruption_types: list[str] | None = None,
    limit: int | None = None,
) -> list[DetectedReaction]:
    """Scan the dataset for reactions that already exhibit negative-reaction patterns."""
    active = {k: v for k, v in DETECTORS.items()
              if corruption_types is None or k in corruption_types}
    reactions = load_dataset(dataset_path)
    if limit is not None:
        reactions = reactions[:limit]

    found: list[DetectedReaction] = []
    for idx, reaction in enumerate(reactions):
        for detector in active.values():
            hit = detector(idx, reaction)
            if hit is not None:
                found.append(hit)
                break  # one hit per reaction is enough
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate negative (chemically impossible) reactions from USPTO SMILES."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DATASET_PATH,
        help=f"Path to reaction dataset (default: {DATASET_PATH}).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write JSONL negatives (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=DEFAULT_EXAMPLES_PER_CATEGORY,
        help=(
            "Number of negatives to generate for each category "
            f"(default: {DEFAULT_EXAMPLES_PER_CATEGORY})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )

    # ---- scan mode ----
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan the dataset for pre-existing negative reactions instead of generating new ones.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        metavar="N",
        help="Only scan the first N reactions (default: all).",
    )
    parser.add_argument(
        "--scan-types",
        nargs="+",
        choices=list(DETECTORS),
        default=None,
        metavar="TYPE",
        help="Corruption types to detect (default: all). Choices: " + ", ".join(DETECTORS),
    )
    parser.add_argument(
        "--scan-output",
        type=str,
        default=None,
        metavar="PATH",
        help="Write detected reactions to this JSONL file (default: print only).",
    )
    return parser.parse_args()


def write_per_type_reaction_files(negatives: list[NegativeReaction], output_path: Path) -> None:
    grouped: dict[str, list[str]] = {name: [] for name in CORRUPTION_TYPES}
    for item in negatives:
        grouped.setdefault(item.corruption_type, []).append(item.corrupted_reaction)

    for corruption_type, reactions in grouped.items():
        type_file = output_path.parent / f"{output_path.stem}_{corruption_type}.txt"
        with type_file.open("w", encoding="utf-8") as handle:
            for reaction in reactions:
                # Keep exactly one raw reaction string per line, matching dataset style.
                handle.write(f"{reaction}\n")
        print(f"Wrote {len(reactions)} reactions to {type_file}")


def main() -> None:
    args = parse_args()

    if args.scan:
        hits = scan_dataset(
            dataset_path=args.dataset_path,
            corruption_types=args.scan_types,
            limit=args.scan_limit,
        )
        by_type: dict[str, list[DetectedReaction]] = {}
        for h in hits:
            by_type.setdefault(h.corruption_type, []).append(h)

        for ctype, records in sorted(by_type.items()):
            print(f"\n=== {ctype} ({len(records)} found) ===")
            for r in records:
                print(f"  [{r.index}] {r.note}")
                print(f"       {r.reaction}")

        print("\nSummary:")
        for ctype in DETECTORS:
            count = len(by_type.get(ctype, []))
            print(f"  {ctype}: {count}")
        print(f"  Total: {len(hits)} (out of {args.scan_limit or 'all'} scanned)")
        if RXNMapper is None and (args.scan_types is None or "rxnmapper_low_confidence" in args.scan_types):
            print("Warning: rxnmapper not installed; rxnmapper_low_confidence detector skipped.")

        if args.scan_output:
            out = Path(args.scan_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as fh:
                for h in hits:
                    fh.write(json.dumps(asdict(h), ensure_ascii=True) + "\n")
            print(f"Wrote {len(hits)} detected reactions to {out}")
        return

    negatives = generate_negatives(
        dataset_path=args.dataset_path,
        examples_per_category=args.examples_per_category,
        seed=args.seed,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in negatives:
            handle.write(json.dumps(asdict(item), ensure_ascii=True) + "\n")
    write_per_type_reaction_files(negatives, output_path)

    if Chem is None:
        print("Warning: RDKit not available; generated negatives with string-level corruption.")
    print(f"Wrote {len(negatives)} negative reactions to {output_path}")
    for item in negatives:
        print(f"[{item.id}] {item.corruption_type}: {item.corrupted_reaction}")


if __name__ == "__main__":
    main()
