from __future__ import annotations

import random
import re
from itertools import permutations
from typing import Any, Iterable, Optional

from rdkit import Chem
from rdkit.Chem import rdChemReactions


DEFAULT_DATASET_PATH = "/home/bhagavan/rlms/datasets/reactionSmilesFigShareUSPTO2023.txt"


def load_lines(dataset_path: str = DEFAULT_DATASET_PATH) -> list[str]:
    with open(dataset_path, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
    return [f"{i} {line}" for i, line in enumerate(raw_lines)]


class BaseContextPipeline:
    def __init__(self, name: str):
        self.name = name

    def build_context(
        self,
        *,
        context_size: int,
        correct_indices: Iterable[int] | None = None,
        query: str = "",
        excluded_indices: Iterable[int] | None = None,
    ) -> str:
        raise NotImplementedError


class RandomContextPipeline(BaseContextPipeline):
    def __init__(
        self,
        lines: list[str],
        rng: random.Random,
        ground_truth_indices_by_reaction: dict[str, list[int]] | None = None,
        ground_truth_fraction_per_context: float = 0.0,
        min_selected_ground_truth: int = 1,
    ):
        super().__init__(name="random")
        self.lines = lines
        self.rng = rng
        self.ground_truth_indices_by_reaction = ground_truth_indices_by_reaction or {}
        self.ground_truth_fraction_per_context = min(1.0, max(0.0, ground_truth_fraction_per_context))
        self.min_selected_ground_truth = max(1, int(min_selected_ground_truth))
        self.line_by_idx: dict[int, str] = {}
        for line in lines:
            idx_str, _ = line.split(" ", 1)
            self.line_by_idx[int(idx_str)] = line

    def _valid_line_indices(self, indices: Iterable[int] | None) -> set[int]:
        return {
            idx
            for idx in (indices or [])
            if isinstance(idx, int) and 0 <= idx < len(self.lines)
        }

    def _sample_with_seed(
        self,
        top_k: int,
        forced_indices: list[int],
        excluded_indices: Iterable[int] | None = None,
    ) -> str:
        valid_forced = [idx for idx in forced_indices if 0 <= idx < len(self.lines)]
        dedup_forced = list(dict.fromkeys(valid_forced))[:top_k]
        forced_set = set(dedup_forced)
        excluded_set = self._valid_line_indices(excluded_indices)
        remainder_pool = [
            i for i in range(len(self.lines)) if i not in forced_set and i not in excluded_set
        ]
        random_take = min(top_k - len(dedup_forced), len(remainder_pool))
        random_indices = self.rng.sample(remainder_pool, k=random_take)
        sampled = dedup_forced + random_indices
        self.rng.shuffle(sampled)
        return "\n".join(self.lines[i] for i in sampled)

    def build_context(
        self,
        *,
        context_size: int,
        correct_indices: Iterable[int] | None = None,
        query: str = "",
        excluded_indices: Iterable[int] | None = None,
    ) -> str:
        pipeline_excluded = self._valid_line_indices(excluded_indices)
        if context_size < 0:
            if not pipeline_excluded:
                return "\n".join(self.lines)
            return "\n".join(
                line
                for idx, line in enumerate(self.lines)
                if idx not in pipeline_excluded
            )
        top_k = min(context_size, len(self.lines))
        if top_k == 0:
            return ""

        if correct_indices is not None:
            valid_correct = sorted(
                {
                    idx
                    for idx in correct_indices
                    if isinstance(idx, int) and 0 <= idx < len(self.lines)
                }
            )
            answer_count = len(valid_correct)
            if answer_count > 0:
                ratio = answer_count / len(self.lines)
                ratio_scaled = ratio * top_k
                ratio_scaled_floor = int(ratio_scaled)
                half_cap = top_k // 2
                forced_count = min(
                    answer_count,
                    half_cap,
                    max(self.min_selected_ground_truth, ratio_scaled_floor),
                )
                print(
                    f"[PIPELINE] context_size_requested={context_size} context_size_effective={top_k} "
                    f"answers_total={answer_count} dataset_size={len(self.lines)} "
                    f"ratio={ratio:.6f} ratio_times_context={ratio_scaled:.4f} "
                    f"ratio_floor={ratio_scaled_floor} half_cap={half_cap} "
                    f"min_gt={self.min_selected_ground_truth} "
                    f"selected_ground_truth={forced_count}"
                )
                if forced_count > 0:
                    forced = self.rng.sample(valid_correct, k=forced_count)
                    non_forced_correct = set(valid_correct) - set(forced)
                    return self._sample_with_seed(
                        top_k=top_k,
                        forced_indices=forced,
                        excluded_indices=non_forced_correct | pipeline_excluded,
                    )
            else:
                print(
                    f"[PIPELINE] context_size_requested={context_size} context_size_effective={top_k} "
                    f"answers_total=0 selected_ground_truth=0"
                )

        reaction_key = query.strip()
        gt_indices = self.ground_truth_indices_by_reaction.get(reaction_key, [])
        gt_lines = [
            self.line_by_idx[idx]
            for idx in gt_indices
            if idx in self.line_by_idx and idx not in pipeline_excluded
        ]
        if gt_lines and self.ground_truth_fraction_per_context > 0:
            desired_gt = int(round(top_k * self.ground_truth_fraction_per_context))
            if desired_gt == 0:
                desired_gt = 1
            gt_take = min(top_k, len(gt_lines), desired_gt)
            seeded_gt_lines = self.rng.sample(gt_lines, k=gt_take)
            seeded_gt_set = set(seeded_gt_lines)
            remainder_pool = [
                line
                for idx, line in enumerate(self.lines)
                if line not in seeded_gt_set and idx not in pipeline_excluded
            ]
            random_take = top_k - gt_take
            random_lines = self.rng.sample(remainder_pool, k=random_take)
            sampled = seeded_gt_lines + random_lines
            self.rng.shuffle(sampled)
            return "\n".join(sampled)

        return self._sample_with_seed(
            top_k=top_k,
            forced_indices=[],
            excluded_indices=pipeline_excluded,
        )


def build_context_pipeline(
    name: str,
    lines: list[str],
    rng: random.Random,
    ground_truth_indices_by_reaction: dict[str, list[int]] | None = None,
    ground_truth_fraction_per_context: float = 0.0,
    min_selected_ground_truth: int = 1,
) -> BaseContextPipeline:
    if name == "random":
        return RandomContextPipeline(
            lines=lines,
            rng=rng,
            ground_truth_indices_by_reaction=ground_truth_indices_by_reaction,
            ground_truth_fraction_per_context=ground_truth_fraction_per_context,
            min_selected_ground_truth=min_selected_ground_truth,
        )
    raise ValueError(f"Unsupported context pipeline: {name}")


def parse_count(response: str) -> Optional[int]:
    cleaned = response.strip().replace('"', "").replace("'", "")
    if cleaned.isdigit():
        return int(cleaned)
    answer_match = re.search(r"ANSWER:\s*(-?\d+)", cleaned, flags=re.IGNORECASE)
    if answer_match:
        return int(answer_match.group(1))
    all_ints = re.findall(r"-?\d+", cleaned)
    if all_ints:
        return int(all_ints[-1])
    return None


def parse_indices(response: str) -> list[int]:
    cleaned = response.strip()
    if not cleaned:
        return []
    answer_match = re.search(r"ANSWER:\s*(.*)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    candidate = answer_match.group(1).strip() if answer_match else cleaned
    if candidate.replace(" ", "") == "-1":
        return []
    int_tokens = re.findall(r"-?\d+", candidate)
    if not int_tokens:
        return []
    seen: set[int] = set()
    indices: list[int] = []
    for token in int_tokens:
        idx = int(token)
        if idx < 0:
            continue
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def extract_response_text(response_obj: Any) -> str:
    if response_obj is None:
        return ""
    result = getattr(response_obj, "result", None)
    if result is not None:
        message = getattr(result, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
        content = getattr(result, "content", None)
        if isinstance(content, str):
            return content
    message = getattr(response_obj, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    content = getattr(response_obj, "content", None)
    if isinstance(content, str):
        return content
    return str(response_obj)


def extract_usage_metrics(response: Any) -> dict[str, float | int]:
    def _from_usage_obj(usage: Any) -> tuple[int, int, int, float | None]:
        if usage is None:
            return 0, 0, 0, None
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
            completion = int(
                usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
            )
            total = int(usage.get("total_tokens", 0) or (prompt + completion))
            cost = usage.get("cost")
            if cost is None:
                cost = (
                    usage.get("cost_details", {}).get("upstream_inference_cost")
                    if isinstance(usage.get("cost_details"), dict)
                    else None
                )
            return prompt, completion, total, (float(cost) if cost is not None else None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
        completion = int(
            getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
        )
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
        cost = getattr(usage, "cost", None)
        extra = getattr(usage, "model_extra", None)
        if cost is None and isinstance(extra, dict):
            cost = extra.get("cost")
            if cost is None and isinstance(extra.get("cost_details"), dict):
                cost = extra["cost_details"].get("upstream_inference_cost")
        return prompt, completion, total, (float(cost) if cost is not None else None)

    candidates: list[Any] = []
    raw = getattr(response, "raw", None)
    if raw is not None:
        candidates.append(raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None))
    candidates.append(getattr(response, "usage", None))
    msg = getattr(response, "message", None)
    if msg is not None:
        msg_kwargs = getattr(msg, "additional_kwargs", None)
        if isinstance(msg_kwargs, dict):
            candidates.append(msg_kwargs.get("usage"))
    add_kwargs = getattr(response, "additional_kwargs", None)
    if isinstance(add_kwargs, dict):
        candidates.append(add_kwargs.get("usage"))

    for usage in candidates:
        p, c, t, cost = _from_usage_obj(usage)
        if p or c or t or cost is not None:
            result: dict[str, float | int] = {
                "prompt_tokens": p,
                "completion_tokens": c,
                "total_tokens": t,
            }
            if cost is not None:
                result["cost_usd"] = cost
            return result
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def parse_reaction_sides(indexed_line: str) -> tuple[str, str]:
    _, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    reactant_side = parts[0].strip()
    product_side = parts[-1].strip()
    return reactant_side, product_side


def build_reaction_query(smarts: str) -> rdChemReactions.ChemicalReaction:
    query = rdChemReactions.ReactionFromSmarts(smarts)
    if query is None:
        raise ValueError(f"Failed to parse reaction SMARTS: {smarts}")
    return query


def reaction_matches(indexed_line: str, query_reaction: rdChemReactions.ChemicalReaction) -> bool:
    reactants, products = parse_reaction_sides(indexed_line)
    r_mols = [Chem.MolFromSmiles(s) for s in reactants.split(".")]
    p_mols = [Chem.MolFromSmiles(s) for s in products.split(".")]
    if any(mol is None for mol in r_mols + p_mols):
        return False
    r_templates = list(query_reaction.GetReactants())
    n = len(r_templates)
    reactant_match = False
    for perm in permutations(r_mols, n):
        if all(perm[i].HasSubstructMatch(r_templates[i]) for i in range(n)):
            reactant_match = True
            break
    if not reactant_match:
        return False
    for q in query_reaction.GetProducts():
        if not any(m.HasSubstructMatch(q) for m in p_mols):
            return False
    return True


def ground_truth_indices(
    lines: list[str],
    query_reaction: rdChemReactions.ChemicalReaction,
) -> list[int]:
    matching_indices: list[int] = []
    for line in lines:
        idx_str, _ = line.split(" ", 1)
        idx = int(idx_str)
        if reaction_matches(line, query_reaction):
            matching_indices.append(idx)
    return matching_indices


def precision_recall_f1(
    predicted_indices: set[int],
    ground_truth_index_set: set[int],
) -> tuple[float, float, float]:
    tp = len(predicted_indices & ground_truth_index_set)
    fp = len(predicted_indices - ground_truth_index_set)
    fn = len(ground_truth_index_set - predicted_indices)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def build_reaction_index_question(reaction_label: str, reaction_description: str) -> str:
    return f"""
    Context: You are given a string of chemical reactions in SMILES format, separated by newlines.
    Each reaction is in one of these forms:
    - "index reactants>reagents>products"
    - "index reactants>>products"

    Task:
    Return all reaction indices that match this reaction type:
    - {reaction_label}

    Description:
    - {reaction_description}

    Guidance:
    - Define a single Reaction SMIRKS pattern encoding the full transformation (reactants >> products) with atom mapping to classify reactions.
    - Do not count by separately matching functional groups on each side using independent SMARTS.
    - Use RDKit for matching and filtering.
    - Ignore reagents (middle field).
    - Handle multi-component sides separated by dots (.).
    - Skip malformed reactions or RDKit matching failures.

    Output format:
    - Final response must be exactly one line:
      ANSWER: <comma-separated indices in ascending order>
    - If no matches exist, output:
      ANSWER: -1
    - Do not include any additional prose in the final response.
"""
