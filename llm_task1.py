import os
import random
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import faiss
import numpy as np
import wandb
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from openai import OpenAI
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rlm.tracing import init_tracing, using_tracing_attributes

# os.environ["WANDB_MODE"] = "disabled"

DATASET_PATH = "/workspace/datasets/reactionSmilesFigShareUSPTO2023.txt"
MODEL_NAME = "openai/gpt-5-mini"
SEED = 42
NUM_QUESTIONS = 3
CONTEXT_SIZE = 7500
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2  # ECFP4 / Morgan r=2
RETRIEVER_NAME = "random"
INDEX_CACHE_DIR = "/workspace/datasets/morgan_faiss"
ENABLE_TRACING = True
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Reference scaling setup:
# CONTEXT_SIZE is tuned for google/gemini-3-flash-preview (1M context).
REFERENCE_MODEL_FOR_CONTEXT_SIZE = "google/gemini-3-flash-preview"
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "openai/gpt-5-mini": 200_000,
    "google/gemini-3-flash-preview": 1_000_000,
    "codestral-2508": 250_000,
}
REFERENCE_CONTEXT_TOKENS = MODEL_CONTEXT_WINDOWS[REFERENCE_MODEL_FOR_CONTEXT_SIZE]


def get_model_context_window(model_name: str) -> int:
    model_name_lower = model_name.lower()
    for key, window in MODEL_CONTEXT_WINDOWS.items():
        if key in model_name_lower:
            return window
    return REFERENCE_CONTEXT_TOKENS


def get_dynamic_context_size(model_name: str, base_context_size: int = CONTEXT_SIZE) -> int:
    """
    Scale retrieval k by model context window.
    - base_context_size is calibrated for REFERENCE_MODEL_FOR_CONTEXT_SIZE.
    - returns at least 1.
    """
    model_window = get_model_context_window(model_name)
    scaled = int(base_context_size * (model_window / REFERENCE_CONTEXT_TOKENS))
    return max(1, scaled)


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="llm-rag-product-lookup",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_indices(response: str) -> list[int]:
    response = response.strip()
    if not response:
        return []
    if response.isdigit():
        return [int(response)]
    return [int(num.strip()) for num in response.split(",") if num.strip().isdigit()]


def extract_product(indexed_line: str) -> str:
    _, reaction_smiles = indexed_line.split(" ", 1)
    return reaction_smiles.split(">")[-1].strip()


def load_lines():
    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        return [f"{i} {line}" for i, line in enumerate(raw_lines)]


class BaseRetriever(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def build_context(self, query: str, target_index: int, k: int) -> str:
        pass


class RandomRetriever(BaseRetriever):
    def __init__(self, lines: list[str], rng: random.Random) -> None:
        super().__init__(name="random")
        self.lines = lines
        self.rng = rng

    def build_context(self, query: str, target_index: int, k: int) -> str:
        """
        Randomly sample k equations from the dataset, always including the one
        at target_index.
        """
        del query  # Random retriever is query-agnostic.
        other_indices = [i for i in range(len(self.lines)) if i != target_index]
        sampled = self.rng.sample(other_indices, k=min(k - 1, len(other_indices)))
        sampled.append(target_index)
        self.rng.shuffle(sampled)
        return "\n".join(self.lines[i] for i in sampled)


class MorganFAISSRetriever(BaseRetriever):
    def __init__(self, lines: list[str], dataset_path: str) -> None:
        super().__init__(name="morgan_faiss")
        self.lines = lines
        self.products = [extract_product(line) for line in lines]
        self.fp_bits = FINGERPRINT_BITS
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=FINGERPRINT_RADIUS,
            fpSize=self.fp_bits,
        )
        self.index = None
        self.popcounts = np.zeros(len(lines), dtype=np.int32)
        self.cache_dir = Path(__file__).resolve().parent / INDEX_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = Path(dataset_path).stem
        cache_key = f"{dataset_name}_bits{self.fp_bits}_r{FINGERPRINT_RADIUS}"
        self.index_path = self.cache_dir / f"{cache_key}.faissb"
        self.popcounts_path = self.cache_dir / f"{cache_key}_popcounts.npy"
        self.meta_path = self.cache_dir / f"{cache_key}.meta.npz"

        if self._cache_is_valid(dataset_path):
            print(f"Loading Morgan FAISS index from cache: {self.index_path}")
            self._load_cache()
        else:
            print("Building Morgan FAISS index from dataset...")
            self._build_index()
            self._save_cache(dataset_path)

    def _cache_is_valid(self, dataset_path: str) -> bool:
        if not (
            self.index_path.exists() and self.popcounts_path.exists() and self.meta_path.exists()
        ):
            return False

        try:
            meta = np.load(self.meta_path, allow_pickle=False)
            cached_size = int(meta["dataset_size"])
            cached_mtime = float(meta["dataset_mtime"])
            cached_lines = int(meta["num_lines"])
            cached_bits = int(meta["fp_bits"])
            cached_radius = int(meta["fp_radius"])
        except Exception:
            return False

        try:
            stat = os.stat(dataset_path)
        except OSError:
            return False

        return (
            cached_size == int(stat.st_size)
            and cached_mtime == float(stat.st_mtime)
            and cached_lines == len(self.lines)
            and cached_bits == self.fp_bits
            and cached_radius == FINGERPRINT_RADIUS
        )

    def _build_index(self) -> None:
        index = faiss.IndexBinaryFlat(self.fp_bits)
        packed_fps = []
        total = len(self.products)
        progress_step = max(1, total // 100)  # ~1% increments

        for i, product in enumerate(self.products):
            packed, popcount = self._smiles_to_packed_fp(product)
            packed_fps.append(packed)
            self.popcounts[i] = popcount
            if (i + 1) % progress_step == 0 or (i + 1) == total:
                pct = 100.0 * (i + 1) / total
                print(f"\rMorgan index build progress: {i + 1}/{total} ({pct:.1f}%)", end="", flush=True)

        print()

        fp_matrix = np.ascontiguousarray(np.vstack(packed_fps), dtype=np.uint8)
        index.add(fp_matrix)
        self.index = index

    def _save_cache(self, dataset_path: str) -> None:
        if self.index is None:
            raise ValueError("Morgan FAISS index is not initialized.")

        stat = os.stat(dataset_path)
        faiss.write_index_binary(self.index, str(self.index_path))
        np.save(self.popcounts_path, self.popcounts)
        np.savez(
            self.meta_path,
            dataset_size=np.int64(stat.st_size),
            dataset_mtime=np.float64(stat.st_mtime),
            num_lines=np.int64(len(self.lines)),
            fp_bits=np.int64(self.fp_bits),
            fp_radius=np.int64(FINGERPRINT_RADIUS),
        )
        print(f"Saved Morgan FAISS cache to: {self.index_path}")

    def _load_cache(self) -> None:
        self.index = faiss.read_index_binary(str(self.index_path))
        self.popcounts = np.load(self.popcounts_path, allow_pickle=False).astype(np.int32)

    def _smiles_to_packed_fp(self, smiles: str) -> tuple[np.ndarray, int]:
        """
        Convert SMILES to packed uint8 fingerprint and bit-popcount.
        Invalid SMILES map to an all-zero fingerprint.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            bitvect = np.zeros((self.fp_bits,), dtype=np.uint8)
        else:
            rd_fp = self.morgan_generator.GetFingerprint(mol)
            bitvect = np.zeros((self.fp_bits,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(rd_fp, bitvect)

        packed = np.packbits(bitvect, bitorder="little")
        popcount = int(bitvect.sum())
        return packed, popcount

    def _tanimoto_from_hamming(
        self,
        query_popcount: int,
        candidate_popcounts: np.ndarray,
        hamming_distances: np.ndarray,
    ) -> np.ndarray:
        common_bits = (query_popcount + candidate_popcounts - hamming_distances) / 2.0
        denominator = query_popcount + candidate_popcounts - common_bits
        # Avoid division by zero for degenerate all-zero vectors.
        return np.divide(
            common_bits,
            denominator,
            out=np.zeros_like(common_bits, dtype=np.float32),
            where=denominator > 0,
        )

    def build_context(self, query: str, target_index: int, k: int) -> str:
        del target_index  # Retrieval is purely query-driven.
        if self.index is None:
            raise ValueError("Morgan FAISS index is not initialized.")

        query_packed, query_popcount = self._smiles_to_packed_fp(query)
        query_packed = np.ascontiguousarray(query_packed.reshape(1, -1), dtype=np.uint8)

        top_k = min(k, len(self.lines))
        distances, indices = self.index.search(query_packed, top_k)

        candidate_indices = indices[0]
        candidate_distances = distances[0].astype(np.int32)
        candidate_popcounts = self.popcounts[candidate_indices]
        tanimoto_scores = self._tanimoto_from_hamming(
            query_popcount=query_popcount,
            candidate_popcounts=candidate_popcounts,
            hamming_distances=candidate_distances,
        )

        # Re-rank by Tanimoto (descending), then by lower Hamming distance.
        order = np.lexsort((candidate_distances, -tanimoto_scores))
        ranked_indices = candidate_indices[order]
        return "\n".join(self.lines[idx] for idx in ranked_indices)


class BM25LangChainRetriever(BaseRetriever):
    def __init__(self, lines: list[str]) -> None:
        super().__init__(name="bm25")
        documents = [Document(page_content=line, metadata={"index": idx}) for idx, line in enumerate(lines)]
        self.retriever = BM25Retriever.from_documents(documents)

    def build_context(self, query: str, target_index: int, k: int) -> str:
        del target_index  # Retrieval is purely query-driven.
        self.retriever.k = k
        retrieved_docs = self.retriever.invoke(query)
        return "\n".join(doc.page_content for doc in retrieved_docs)


def build_retriever(name: str, lines: list[str], rng: random.Random) -> BaseRetriever:
    if name == "random":
        return RandomRetriever(lines=lines, rng=rng)
    if name == "bm25":
        return BM25LangChainRetriever(lines=lines)
    if name == "morgan_faiss":
        return MorganFAISSRetriever(lines=lines, dataset_path=DATASET_PATH)
    raise ValueError(f"Unknown retriever: {name}")


def evaluate_retriever(
    client: OpenAI,
    lines: list[str],
    sampled_indices: list[int],
    retriever: BaseRetriever,
) -> None:
    dynamic_context_size = get_dynamic_context_size(MODEL_NAME, base_context_size=CONTEXT_SIZE)
    run_session_id = f"run-llms-{uuid.uuid4()}"
    run = wandb.init(
        project="LLMs-RAG-Product-Lookup",
        config={
            "MODEL_NAME": MODEL_NAME,
            "SEED": SEED,
            "NUM_QUESTIONS": NUM_QUESTIONS,
            "CONTEXT_SIZE": CONTEXT_SIZE,
            "dynamic_context_size": dynamic_context_size,
            "model_context_window": get_model_context_window(MODEL_NAME),
            "reference_context_tokens": REFERENCE_CONTEXT_TOKENS,
            "reference_model_for_context_size": REFERENCE_MODEL_FOR_CONTEXT_SIZE,
            "dataset_path": DATASET_PATH,
            "num_questions": len(sampled_indices),
            "retriever_name": retriever.name,
        },
        reinit=True,
        name=f"{retriever.name}-{MODEL_NAME}",
    )
    correct = 0
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost_usd = 0.0
    samples_with_cost = 0
    retrieval_hits = 0
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    def extract_sample_cost(usage: object | None) -> float | None:
        if usage is None:
            return None
        cost = getattr(usage, "cost", None)
        if cost:
            return float(cost)
        extra = getattr(usage, "model_extra", None)
        if isinstance(extra, dict):
            if extra.get("cost"):
                return float(extra["cost"])
            details = extra.get("cost_details")
            if isinstance(details, dict) and details.get("upstream_inference_cost"):
                return float(details["upstream_inference_cost"])
        return None

    for i, target_index in enumerate(sampled_indices):
        print(f"Question {i + 1}/{len(sampled_indices)}")
        target_line = lines[target_index]
        target_product = extract_product(target_line)
        question = f"""
            Context is a big string of chemical equations in SMILES format, separated by newlines.
            Find the index/indices (number at the start) of the equation for the following PRODUCT (and not the reactants/reagents): {target_product}.
            Report the INDICES separated by commas. DO NOT INCLUDE any other text in your response including quotes, punctuation, or formatting.
            If the product is not found, report an empty string. 
            Beware of outputting too many internal thoughts because of limit on the number of tokens. Just respond with the indices.
            """
        retrieved_context = retriever.build_context(
            query=target_product,
            target_index=target_index,
            k=dynamic_context_size,
        )
        context_has_ground_truth = str(target_index) in retrieved_context
        retrieval_hits += int(context_has_ground_truth)
        if not context_has_ground_truth:
            print(f"[WARNING] Ground truth missing from retrieved context for target_index={target_index}")
        completion_prompt = f"""
        You are given a subset of chemical reactions in SMILES format and a question.
        <context>
        {retrieved_context}
        </context>
        <question>
        {question}
        </question>
        """
        print("-" * 60)
        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(sampled_indices),
                "target_index": target_index,
                "retriever_name": retriever.name,
            },
            tags=["run_llms", "sample", retriever.name],
        ):
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": completion_prompt}],
            )
        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        response = choice.message.content or ""
        if not response:
            print(f"  [WARNING] Empty response. finish_reason={finish_reason!r}")
        parsed = parse_indices(response)
        results.append(parsed)
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        tokens = usage.total_tokens if usage else 0
        sample_cost_usd = extract_sample_cost(usage)
        if sample_cost_usd is not None:
            total_cost_usd += sample_cost_usd
            samples_with_cost += 1
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += tokens
        is_correct = target_index in parsed
        if is_correct:
            correct += 1
        else:
            print(f"❌ Error: {target_index} not in {parsed}")
            print(f"Line: {target_line}")
            print(f"Product: {target_product}")
            print(f"Response: {response!r}")
            print(f"finish_reason: {finish_reason!r}")
            print("-" * 60)
        # Per-sample token usage (single LLM call => iteration=1).
        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/final_total_input_tokens": prompt_tokens,
                f"sample/{i}/final_total_output_tokens": completion_tokens,
                f"sample/{i}/final_total_tokens": tokens,
                f"sample/{i}/iterations": 1,
                f"sample/{i}/is_correct": int(is_correct),
                f"sample/{i}/target_index": target_index,
                f"sample/{i}/target_product": target_product,
                f"sample/{i}/response_raw": response,
                f"sample/{i}/response_parsed": ",".join(str(x) for x in parsed),
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/context_size": dynamic_context_size,
                f"sample/{i}/finish_reason": finish_reason,
                f"sample/{i}/context_has_ground_truth": int(context_has_ground_truth),
                **(
                    {f"sample/{i}/final_total_cost_usd": sample_cost_usd}
                    if sample_cost_usd is not None
                    else {}
                ),
            }
        )
        wandb.log(
            {
                "running_accuracy": correct / (i + 1),
                "running_retrieval_hit_rate": retrieval_hits / (i + 1),
            }
        )
    total = len(results)
    accuracy = (correct / total) if total else 0.0
    retrieval_hit_rate = (retrieval_hits / total) if total else 0.0
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Retrieval hit-rate (ground truth in context): {retrieval_hit_rate:.4f}")
    run.summary["retriever_name"] = retriever.name
    run.summary["correct"] = correct
    run.summary["total"] = total
    run.summary["accuracy"] = accuracy
    run.summary["retrieval_hits"] = retrieval_hits
    run.summary["retrieval_hit_rate"] = retrieval_hit_rate
    run.summary["total_prompt_tokens"] = total_prompt_tokens
    run.summary["total_completion_tokens"] = total_completion_tokens
    run.summary["total_tokens"] = total_tokens
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


def main():
    maybe_init_tracing()
    if not OPENROUTER_API_KEY:
        raise ValueError("Set OPENROUTER_API_KEY in your environment before running.")

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    lines = load_lines()

    rng = random.Random(SEED)
    sampled_indices = rng.sample(range(len(lines)), k=min(NUM_QUESTIONS, len(lines)))

    retriever = build_retriever(name=RETRIEVER_NAME, lines=lines, rng=rng)
    evaluate_retriever(
        client=client,
        lines=lines,
        sampled_indices=sampled_indices,
        retriever=retriever,
    )


if __name__ == "__main__":
    main()