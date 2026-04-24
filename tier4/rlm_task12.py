import argparse
import re
import uuid
import os
import wandb
from rdkit import Chem
from rlm import RLM
from rlm.tracing import init_tracing, using_tracing_attributes

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"
BACKEND = "openrouter"
MODEL_NAME = "openai/gpt-5-mini"
ENABLE_TRACING = True
SEED = 42

NUM_QUESTIONS = 6
MAX_MOLECULE_FREQ = 200
DAG_MODE = "index_asc"

# Fixed evaluation set: 6 target products with DAG-ground-truth longest chains.
FIXED_TARGET_PRODUCTS: list[str] = [
    "O=CCc1csc(Br)n1",
    "Cc1cc(C)c2c(n1)CNC2C",
    "COc1cc2c(cc1N)CN(C)C(C)C2",
    "C#CC(C)N1C(=O)c2ccccc2C1=O",
    "CC(C)(C)c1sc2c(c1N=C=O)CCC2",
    "CCOC(=O)COc1cc(C)c(CO)c(C)c1F",
]

# Hardcoded longest-chain ground truth for each target product in FIXED_TARGET_PRODUCTS.
HARDCODED_GT_LONGEST_CHAIN: dict[str, tuple[int, ...]] = {
    "O=CCc1csc(Br)n1": (
        2203, 2204, 2206, 2207, 35348, 38139,
        38140, 38141, 38142, 38143, 47208, 47209,
    ),
    "Cc1cc(C)c2c(n1)CNC2C": (
        59702, 59703, 59734, 59735, 59736, 59737,
        72776, 74210, 89903, 125750, 125751, 125752,
    ),
    "COc1cc2c(cc1N)CN(C)C(C)C2": (
        19525, 19526, 19527, 19528, 20556, 20557,
        20558, 20584, 20585, 22090, 22091, 22092,
    ),
    "C#CC(C)N1C(=O)c2ccccc2C1=O": (
        54214, 54215, 54216, 54217, 54326, 54447,
        54448, 54449, 67036, 67037, 67038, 82185,
    ),
    "CC(C)(C)c1sc2c(c1N=C=O)CCC2": (
        67156, 67157, 67158, 67159, 67160, 67161,
        67162, 67334, 67335, 67336, 67337, 67338,
    ),
    "CCOC(=O)COc1cc(C)c(CO)c(C)c1F": (
        3974, 3975, 3976, 3977, 17789, 17793,
        39886, 39887, 39888, 39889, 39890, 39891,
    ),
}

RLM_INIT_KWARGS = {
    "backend": BACKEND,
    "backend_kwargs": {"model_name": MODEL_NAME},
    "verbose": True,
    "max_depth": 2,
}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_chain(response: str) -> tuple[int, ...]:
    response = response.strip()
    if not response:
        return tuple()
    if response.replace(" ", "") == "-1":
        return tuple()

    for line in response.splitlines():
        line = line.strip()
        if not line or line == "-1":
            continue
        nums = re.findall(r"\d+", line)
        if nums:
            return tuple(int(n) for n in nums)
    return tuple()


# ---------------------------------------------------------------------------
# Chain verification + ordered metrics
# ---------------------------------------------------------------------------

def parse_reaction_sides(indexed_line: str) -> tuple[str, str]:
    _, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    return parts[0].strip(), parts[-1].strip()


def canonicalize_components(smiles_str: str) -> list[str]:
    canonical: list[str] = []
    for smi in smiles_str.split("."):
        smi = smi.strip()
        if not smi:
            continue
        try:
            csmi = Chem.CanonSmiles(smi)
            if csmi:
                canonical.append(csmi)
        except Exception:
            continue
    return canonical


def build_reaction_component_maps(
    lines: list[str], max_molecule_freq: int
) -> tuple[dict[int, list[str]], dict[int, list[str]], set[str]]:
    products_by_idx: dict[int, list[str]] = {}
    reactants_by_idx: dict[int, list[str]] = {}
    producers: dict[str, set[int]] = {}
    consumers: dict[str, set[int]] = {}

    for line in lines:
        idx_str, _ = line.split(" ", 1)
        idx = int(idx_str)
        try:
            reactant_side, product_side = parse_reaction_sides(line)
        except (ValueError, IndexError):
            continue

        r_canon = canonicalize_components(reactant_side)
        p_canon = canonicalize_components(product_side)
        if not r_canon or not p_canon:
            continue

        reactants_by_idx[idx] = r_canon
        products_by_idx[idx] = p_canon

        for smi in p_canon:
            producers.setdefault(smi, set()).add(idx)
        for smi in r_canon:
            consumers.setdefault(smi, set()).add(idx)

    frequent_molecules: set[str] = set()
    for smi, idxs in producers.items():
        if len(idxs) > max_molecule_freq:
            frequent_molecules.add(smi)
    for smi, idxs in consumers.items():
        if len(idxs) > max_molecule_freq:
            frequent_molecules.add(smi)

    return reactants_by_idx, products_by_idx, frequent_molecules


def verify_chain_leads_to_target(
    chain: tuple[int, ...],
    target_product_smiles: str,
    reactants_by_idx: dict[int, list[str]],
    products_by_idx: dict[int, list[str]],
    frequent_molecules: set[str],
    dag_mode: str = "index_asc",
) -> tuple[bool, str]:
    if not chain:
        return False, "chain is empty"
    if len(set(chain)) != len(chain):
        return False, "chain has repeated reaction indices"

    try:
        target_canon = Chem.CanonSmiles(target_product_smiles)
    except Exception:
        return False, "target product SMILES cannot be canonicalized"

    for idx in chain:
        if idx not in products_by_idx or idx not in reactants_by_idx:
            return False, f"reaction index {idx} missing parsed reactants/products"

    for a, b in zip(chain, chain[1:]):
        if dag_mode == "index_asc" and not (a < b):
            return False, f"DAG order violated: {a} !< {b}"
        if dag_mode == "index_desc" and not (a > b):
            return False, f"DAG order violated: {a} !> {b}"

        a_products = set(products_by_idx[a]) - frequent_molecules
        b_reactants = set(reactants_by_idx[b]) - frequent_molecules
        if not (a_products & b_reactants):
            return False, f"no valid product->reactant link between {a} and {b}"

    last_idx = chain[-1]
    if target_canon not in set(products_by_idx[last_idx]):
        return False, f"final reaction {last_idx} does not produce target {target_canon}"

    return True, "ok"


def common_prefix_len(pred: tuple[int, ...], gt: tuple[int, ...]) -> int:
    n = min(len(pred), len(gt))
    i = 0
    while i < n and pred[i] == gt[i]:
        i += 1
    return i


def position_accuracy(pred: tuple[int, ...], gt: tuple[int, ...]) -> float:
    denom = max(len(pred), len(gt), 1)
    matches = sum(1 for a, b in zip(pred, gt) if a == b)
    return matches / denom


def lcs_length(pred: tuple[int, ...], gt: tuple[int, ...]) -> int:
    n = len(pred)
    m = len(gt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        pi = pred[i - 1]
        for j in range(1, m + 1):
            if pi == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def normalized_edit_distance(pred: tuple[int, ...], gt: tuple[int, ...]) -> float:
    n = len(pred)
    m = len(gt)
    if n == 0 and m == 0:
        return 0.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[n][m] / max(n, m)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def precision_recall_f1(
    predicted: set[int], ground_truth: set[int]
) -> tuple[float, float, float]:
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_question(target_product_smiles: str) -> str:
    return f"""
    Context: You are given a large set of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Each side (reactants / products) may contain multiple molecules separated by dots (.).
    Ignore reagents (middle field between the two > delimiters).

    Task:
    Find the SINGLE longest synthetic chain of reactions that produces target product:
    {target_product_smiles}

    A synthetic chain is an ordered sequence of DISTINCT reaction indices [r_0, r_1, ..., r_k]
    where for every consecutive pair (r_i, r_{{i+1}}), at least one PRODUCT of reaction r_i
    is identical to at least one REACTANT of reaction r_{{i+1}} by canonical SMILES equality.

    A chain "produces the target product" if the final reaction r_k has the target product
    among its products (canonical SMILES match).

    IMPORTANT DAG RULE:
    - To avoid cycles, only allow links from lower index to higher index (index_asc):
      r_i < r_{{i+1}}.

    Molecule identity must use canonical SMILES (RDKit Chem.CanonSmiles).
    Do NOT use substructure matching. Exclude trivially common molecules that appear in more than {MAX_MOLECULE_FREQ} reactions
    as products or reactants.

    If multiple longest chains have the same maximum length, return the lexicographically
    smallest chain of indices.

    Guidance:
    - Use RDKit for SMILES canonicalization.
    - Skip malformed reactions or malformed molecules.
    - Build graph edges product->reactant under the DAG rule.
    - Compute the longest path ending at any reaction that produces the target product.
    - DO NOT assume/simulate output of the code. Wait for the code to get executed and only then return the final answer.
    - DO NOT USE `FINAL` for writing a comment/thought. Only use this for the final answer.
    - DO NOT WRITE `FINAL` without observing the output of the code.

    Output format:
    - Return ONLY one comma-separated sequence of reaction indices for the longest chain.
    - No other text, labels, punctuation, or formatting.
    - If no producing chain exists, return -1.
    """


# ---------------------------------------------------------------------------
# Tracing / CLI
# ---------------------------------------------------------------------------

def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="RLMs-Task12",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RLM task 12 — longest product chain (DAG) evaluation."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier for backend (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=NUM_QUESTIONS,
        help=f"Number of target products to evaluate (default: {NUM_QUESTIONS}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_name: str, num_questions: int) -> None:
    maybe_init_tracing()
    rlm_init_kwargs = dict(RLM_INIT_KWARGS)
    rlm_init_kwargs["backend_kwargs"] = {"model_name": model_name}
    rlm = RLM(**rlm_init_kwargs)
    run_session_id = f"run-rlms-{uuid.uuid4()}"

    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        lines = [f"{i} {line}" for i, line in enumerate(raw_lines)]

    print(f"Loaded {len(lines)} reactions from {DATASET_PATH}")

    if num_questions != len(FIXED_TARGET_PRODUCTS):
        print(
            f"num_questions={num_questions} ignored; using fixed set of "
            f"{len(FIXED_TARGET_PRODUCTS)} target products."
        )
    target_products = list(FIXED_TARGET_PRODUCTS)
    print(f"Using {len(target_products)} fixed target products.")

    reactants_by_idx, products_by_idx, frequent_molecules = build_reaction_component_maps(
        lines, max_molecule_freq=MAX_MOLECULE_FREQ
    )

    gt_chains: list[tuple[int, ...]] = []
    for product in target_products:
        chain = HARDCODED_GT_LONGEST_CHAIN.get(product)
        if chain is None:
            raise ValueError(f"Missing hardcoded GT chain for product={product}")
        is_valid_gt, reason = verify_chain_leads_to_target(
            chain=chain,
            target_product_smiles=product,
            reactants_by_idx=reactants_by_idx,
            products_by_idx=products_by_idx,
            frequent_molecules=frequent_molecules,
            dag_mode=DAG_MODE,
        )
        if not is_valid_gt:
            raise ValueError(
                f"Invalid hardcoded GT chain for product={product}: {reason}. "
                f"chain={chain}"
            )
        gt_chains.append(chain)
        print(f"Ground truth [product={product}] verified: len={len(chain)}")

    context = "\n".join(lines)

    run = wandb.init(
        project="RLMs-Task12",
        config={
            "MODEL_NAME": model_name,
            "backend": BACKEND,
            "model_name": model_name,
            "dataset_path": DATASET_PATH,
            "num_questions": len(target_products),
            "fixed_target_products": target_products,
            "dag_mode": DAG_MODE,
            "max_molecule_freq": MAX_MOLECULE_FREQ,
            "seed": SEED,
            "rlm_init_kwargs": rlm_init_kwargs,
            "task_description": "Find longest DAG chain that produces a target product.",
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    exact_match_count = 0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    macro_prefix_ratio = 0.0
    macro_position_accuracy = 0.0
    macro_lcs_ratio = 0.0
    macro_norm_edit_distance = 0.0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, (product, gt_chain) in enumerate(zip(target_products, gt_chains)):
        question = build_question(target_product_smiles=product)
        gt_set = set(gt_chain)
        completion_kwargs = {"prompt": context, "root_prompt": question}

        print(
            f"\nQuestion {i + 1}/{len(target_products)}: "
            f"target_product={product}, gt_len={len(gt_chain)}"
        )

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(target_products),
                "task": "longest_product_chain_dag",
                "target_product": product,
                "gt_chain_length": len(gt_chain),
            },
            tags=["run_rlms", "sample", "task12_LONGEST_PRODUCT_CHAIN"],
        ):
            completion = rlm.completion(**completion_kwargs)
            response = completion.response

        iteration_metrics = rlm.get_last_iteration_metrics()
        pred_chain = parse_chain(response)
        pred_set = set(pred_chain)
        precision, recall, f1 = precision_recall_f1(pred_set, gt_set)
        prefix_len = common_prefix_len(pred_chain, gt_chain)
        prefix_ratio = prefix_len / len(gt_chain) if gt_chain else 0.0
        pos_acc = position_accuracy(pred_chain, gt_chain)
        lcs_len = lcs_length(pred_chain, gt_chain)
        lcs_ratio = lcs_len / len(gt_chain) if gt_chain else 0.0
        norm_edit_distance = normalized_edit_distance(pred_chain, gt_chain)
        sample_cost_usd = completion.usage_summary.total_cost
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1

        is_exact_match = pred_chain == gt_chain
        if is_exact_match:
            exact_match_count += 1
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        macro_prefix_ratio += prefix_ratio
        macro_position_accuracy += pos_acc
        macro_lcs_ratio += lcs_ratio
        macro_norm_edit_distance += norm_edit_distance

        print(f"Response [target={product}]: {response[:500]}{'…' if len(response) > 500 else ''}")
        print(f"Predicted chain length [target={product}]: {len(pred_chain)}")
        print(f"Ground truth chain length [target={product}]: {len(gt_chain)}")
        print(
            f"Metrics [target={product}] -> precision={precision:.4f} "
            f"recall={recall:.4f} f1={f1:.4f} exact_match={is_exact_match} "
            f"prefix_ratio={prefix_ratio:.4f} position_acc={pos_acc:.4f} "
            f"lcs_ratio={lcs_ratio:.4f} norm_edit_dist={norm_edit_distance:.4f}"
        )

        gt_str = ",".join(str(x) for x in gt_chain)
        pred_str = ",".join(str(x) for x in pred_chain)

        for metric in iteration_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                    f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                    f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                }
            )

        if iteration_metrics:
            last_metric = iteration_metrics[-1]
            wandb.log(
                {
                    "sample_idx": i,
                    f"sample/{i}/target_product": product,
                    f"sample/{i}/ground_truth_chain_length": len(gt_chain),
                    f"sample/{i}/predicted_chain_length": len(pred_chain),
                    f"sample/{i}/final_total_input_tokens": last_metric["total_input_tokens"],
                    f"sample/{i}/final_total_output_tokens": last_metric["total_output_tokens"],
                    f"sample/{i}/final_total_tokens": last_metric["total_tokens"],
                    f"sample/{i}/iterations": len(iteration_metrics),
                    f"sample/{i}/response_raw": response,
                    f"sample/{i}/response_parsed_chain": pred_str,
                    f"sample/{i}/ground_truth_chain": gt_str,
                    f"sample/{i}/precision": precision,
                    f"sample/{i}/recall": recall,
                    f"sample/{i}/f1": f1,
                    f"sample/{i}/prefix_match_length": prefix_len,
                    f"sample/{i}/prefix_match_ratio": prefix_ratio,
                    f"sample/{i}/position_accuracy": pos_acc,
                    f"sample/{i}/lcs_length": lcs_len,
                    f"sample/{i}/lcs_ratio": lcs_ratio,
                    f"sample/{i}/normalized_edit_distance": norm_edit_distance,
                    f"sample/{i}/is_exact_match": int(is_exact_match),
                    f"sample/{i}/completion_root_prompt": question,
                    f"sample/{i}/completion_prompt_char_count": len(context),
                    **(
                        {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                        if sample_cost_usd is not None
                        else {}
                    ),
                }
            )

    total = len(target_products)
    exact_match_accuracy = (exact_match_count / total) if total else 0.0
    macro_precision = (macro_precision / total) if total else 0.0
    macro_recall = (macro_recall / total) if total else 0.0
    macro_f1 = (macro_f1 / total) if total else 0.0
    macro_prefix_ratio = (macro_prefix_ratio / total) if total else 0.0
    macro_position_accuracy = (macro_position_accuracy / total) if total else 0.0
    macro_lcs_ratio = (macro_lcs_ratio / total) if total else 0.0
    macro_norm_edit_distance = (macro_norm_edit_distance / total) if total else 0.0
    print(f"\n{'=' * 60}")
    print(f"Exact match: {exact_match_count}/{total}")
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro prefix match ratio: {macro_prefix_ratio:.4f}")
    print(f"Macro position accuracy: {macro_position_accuracy:.4f}")
    print(f"Macro LCS ratio: {macro_lcs_ratio:.4f}")
    print(f"Macro normalized edit distance: {macro_norm_edit_distance:.4f}")

    for product, gt_chain in zip(target_products, gt_chains):
        run.summary[f"ground_truth/product_{product}/chain"] = ",".join(str(x) for x in gt_chain)
        run.summary[f"ground_truth/product_{product}/length"] = len(gt_chain)

    run.summary["exact_match_correct"] = exact_match_count
    run.summary["total"] = total
    run.summary["exact_match_accuracy"] = exact_match_accuracy
    run.summary["macro_precision"] = macro_precision
    run.summary["macro_recall"] = macro_recall
    run.summary["macro_f1"] = macro_f1
    run.summary["macro_prefix_match_ratio"] = macro_prefix_ratio
    run.summary["macro_position_accuracy"] = macro_position_accuracy
    run.summary["macro_lcs_ratio"] = macro_lcs_ratio
    run.summary["macro_normalized_edit_distance"] = macro_norm_edit_distance
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        num_questions=args.num_questions,
    )

