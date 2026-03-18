from typing import Any, Callable, Dict, List, Optional, Tuple
import io
import contextlib
import ast
import traceback
import asyncio
import random
import re
import os
import uuid

import wandb
from rdkit import Chem
from rdkit.Chem import Descriptors
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import (
    Context,
    Event,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.llms.openrouter import OpenRouter

from rlm.tracing import init_tracing, using_tracing_attributes
from rlm.utils.token_utils import count_tokens


DATASET_PATH = "/workspace/datasets/reactionSmilesFigShareUSPTO2023.txt"
MODEL_NAME = "x-ai/grok-4.1-fast"  # or try something simpler first
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_TRACING = True
WORKFLOW_TIMEOUT_S = 300.0
THRESHOLDS = list(range(100, 300, 10))
SEED = 42
CONTEXT_SIZE = 10
RETRIEVER_NAME = "random"
MAX_OUTPUT_TOKENS = 10_000
MAX_ITERATIONS = 8
# os.environ["WANDB_MODE"] = "disabled"


def load_lines() -> list[str]:
    with open(DATASET_PATH, "r") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
    return [f"{i} {line}" for i, line in enumerate(raw_lines)]


class BaseRetriever:
    def __init__(self, name: str):
        self.name = name

    def build_context(self, query: str, target_index: int, k: int) -> str:
        raise NotImplementedError


class RandomRetriever(BaseRetriever):
    def __init__(self, lines: list[str], rng: random.Random):
        super().__init__(name="random")
        self.lines = lines
        self.rng = rng

    def build_context(self, query: str, target_index: int, k: int) -> str:
        del query, target_index  # Random retrieval ignores query/target.
        if k < 0:
            return "\n".join(self.lines)
        top_k = min(k, len(self.lines))
        sampled = self.rng.sample(self.lines, k=top_k)
        return "\n".join(sampled)


def build_retriever(name: str, lines: list[str], rng: random.Random) -> BaseRetriever:
    if name == "random":
        return RandomRetriever(lines=lines, rng=rng)
    raise ValueError(f"Unsupported retriever for codeact_task2: {name}")


def parse_count(response: str) -> Optional[int]:
    cleaned = response.strip().replace('"', "").replace("'", "")
    if cleaned.isdigit():
        return int(cleaned)

    # Prefer an explicit final answer if present.
    answer_match = re.search(r"ANSWER:\s*(-?\d+)", cleaned, flags=re.IGNORECASE)
    if answer_match:
        return int(answer_match.group(1))

    # Fallback: use the last integer mentioned in the response.
    all_ints = re.findall(r"-?\d+", cleaned)
    if all_ints:
        return int(all_ints[-1])
    return None


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


def maybe_init_tracing() -> None:
    if not ENABLE_TRACING:
        return
    initialized = init_tracing(
        project_name="codeact-delta-weight-count",
        auto_instrument=True,
        batch=False,
    )
    if not initialized:
        print(
            "Tracing requested, but Phoenix/OpenInference dependencies are unavailable. "
            "Install with: pip install '.[tracing]'"
        )


def parse_reaction_sides(indexed_line: str) -> tuple[str, str]:
    _, reaction_smiles = indexed_line.split(" ", 1)
    parts = reaction_smiles.split(">")
    reactant_side = parts[0].strip()
    product_side = parts[-1].strip()
    return reactant_side, product_side


def heaviest_component_weight(side_smiles: str) -> Optional[float]:
    if not side_smiles:
        return None

    weights: list[float] = []
    for comp in side_smiles.split("."):
        comp = comp.strip()
        if not comp:
            continue
        mol = Chem.MolFromSmiles(comp)
        if mol is None:
            continue
        weights.append(Descriptors.MolWt(mol))

    if not weights:
        return None
    return max(weights)


def reaction_delta_weight(indexed_line: str) -> Optional[float]:
    reactants, products = parse_reaction_sides(indexed_line)
    heaviest_reactant = heaviest_component_weight(reactants)
    heaviest_product = heaviest_component_weight(products)
    if heaviest_reactant is None or heaviest_product is None:
        return None
    return heaviest_product - heaviest_reactant


def count_matches(lines: list[str], threshold: float) -> int:
    total = 0
    for line in lines:
        delta = reaction_delta_weight(line)
        if delta is not None and delta > threshold:
            total += 1
    return total


def _extract_usage_metrics(response: Any) -> dict[str, float | int]:
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


def _extract_finish_reason(response: Any) -> Optional[str]:
    raw = getattr(response, "raw", None)
    if raw is None:
        return None

    choices = raw.get("choices") if isinstance(raw, dict) else getattr(raw, "choices", None)
    if not choices:
        return None

    first = choices[0]
    if isinstance(first, dict):
        reason = first.get("finish_reason")
    else:
        reason = getattr(first, "finish_reason", None)
    return str(reason) if reason is not None else None


class SimpleCodeExecutor:
    """
    Executes Python code with persistent state.
    NOTE: not safe for production use.
    """

    def __init__(self, locals: Dict[str, Any], globals: Dict[str, Any]):
        # Use one shared namespace for exec() so imports/defs are visible
        # consistently (avoids NameError from split globals/locals scope).
        self.namespace: Dict[str, Any] = {}
        self.namespace.update(globals)
        self.namespace.update(locals)

    def execute(self, code: str) -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        output = ""
        return_value = None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    tree = ast.parse(code)
                    last_node = tree.body[-1] if tree.body else None

                    if isinstance(last_node, ast.Expr):
                        last_line = code.rstrip().split("\n")[-1]
                        exec_code = code[: -len(last_line)] + "\n__result__ = " + last_line
                        exec(exec_code, self.namespace, self.namespace)
                        return_value = self.namespace.get("__result__")
                    else:
                        exec(code, self.namespace, self.namespace)
                except Exception:
                    exec(code, self.namespace, self.namespace)

            output = stdout.getvalue()
            if stderr.getvalue():
                output += "\n" + stderr.getvalue()
        except Exception as e:
            output = f"Error: {type(e).__name__}: {str(e)}\n"
            output += traceback.format_exc()

        if return_value is not None:
            output += "\n\n" + str(return_value)
        return output


class InputEvent(Event):
    input: list[ChatMessage]


class StreamEvent(Event):
    delta: str


class CodeExecutionEvent(Event):
    code: str


CODEACT_SYSTEM_PROMPT = """
You are a helpful assistant in a CodeAct (Code + Acting) loop that can execute Python code to help you answer questions.  
You must follow this format for each step:  

1. THINK: Reason about what you need to do next 
2. ACT: Take an action (execute code)  

**ENCOURAGED: Use Python code execution when helpful!** - Code execution is verifiable and helps you check your work programmatically 
- Use code to solve problems, verify calculations, analyze data, and validate your reasoning 
- Code execution results are reliable and help you build confidence in your answers 
- When in doubt, writing code to check, verify, or compute can be helpful 
- If the user/task instructions require code execution, you must execute code at least once before finalizing.  

AVAILABLE ACTIONS: 
- Execute Python code: Write code in 

```python 
CODE...
```
The code will be executed and results returned. Always output your Python code in a code block starting with ```python and end with exactly ``` on a new line. Do not omit the closing backticks.

- Provide final answer: When you have enough information, provide your final answer exactly as "ANSWER: <integer>".

FORMAT REQUIREMENTS:
- For non-final turns, start with "THINK: " followed by your reasoning.
- For non-final turns, then write Python code in ```python blocks to execute.
- Final turn exception: when you already have enough information from observations, output exactly `ANSWER: <integer>` and do not include THINK or a code block.
- You can execute code multiple times.
- Code execution results will be returned to you automatically including error or stack trace if any.
- Variables persist across code executions in the same session.
- **CRITICAL: Code is executed in a persistent Python environment. Variables and imports persist across executions. You must include all necessary imports, data definitions, and context within your code blocks. Do not use fillers (e.g. FILL IN WITH REAL DATA), they have to be written in code.**  
- **DO NOT SIMULATE EXECUTION**: Never invent code output, never claim "this would print", and never infer final numeric results without receiving a message produced by actual execution.
- **WAIT FOR OBSERVATION**: After you send a Python block, stop and wait for the next tool result. Only then continue reasoning or provide `ANSWER: <integer>`.

EXAMPLE:

Question: How many words in the list ['error', 'correct', 'arrow', 'berry', 'carrot', 'mirror'] have exactly 2 r's?  

Turn 1:
```
THINK: I need to count how many words in the list have exactly 2 r's. I can write Python code using regex to do this. 
```python 
    import re  
    words = ['error', 'correct', 'arrow', 'berry', 'carrot', 'mirror'] 
    pattern = r'^[^r]*r[^r]*r[^r]*$' # Matches words with exactly 2 r's 
    count = 0 
    matching_words = [] 
    for word in words: if re.match(pattern, word): 
        count += 1 matching_words.append(word) 
        print(f"{word} has 2 r's") 
    print(f"Total words with 2 r's: {count}") 
```  
```

[Code execution results returned...]  
Total words with 2 r's: 4

Turn 2:
```
ANSWER: 4
```

The question is now answered with 4 as the final answer.

---

IMPORTANT RULES: 
- For non-final turns, start with THINK to reason about your next step. 
- Be strategic to avoid exceeding the context window 
- **CODE EXECUTION**: Use code to verify, check, and solve problems programmatically when helpful. If the task explicitly requires code execution, do not skip it. NEVER GUESS the answer.
- **CODE EXECUTION CONTEXT**: Your code is executed in the python environment. You must explicitly include all imports, data, and context needed. Variables persist across executions, but each code block must be self-contained with all necessary setup.
- **NO FAKE RESULTS**: If you have not seen the response for the current code block, you must not provide computed numbers or a final answer yet.
- When you have enough information to answer, provide your final answer in the format: "ANSWER: [your answer]" without any additional text or formatting.
- After executing code, you must wait for the code execution results to be returned before PROVIDING a final answer. Do not think the answer is just the code output or the code itself. 
- Remember, doing `print(answer)` does not mean the answer is just the code output. You have to answer like "ANSWER: <integer>".
- Once an observation gives enough information, immediately finalize on the next turn with exactly `ANSWER: <integer>`.
"""


class CodeActAgent(Workflow):
    def __init__(
        self,
        code_execute_fn: Callable,
        llm: LLM | None = None,
        **workflow_kwargs: Any,
    ) -> None:
        super().__init__(**workflow_kwargs)
        self.code_execute_fn = code_execute_fn
        self.llm = llm
        self.system_message = ChatMessage(
            role="system",
            content=CODEACT_SYSTEM_PROMPT,
        )

    def _parse_code(self, response: str) -> str | None:
        fenced_matches = re.findall(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if fenced_matches:
            return "\n\n".join(block.strip() for block in fenced_matches if block.strip())
        return None

    @step
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        memory = await ctx.store.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        user_input = ev.get("user_input")
        if user_input is None:
            raise ValueError("user_input kwarg is required")
        memory.put(ChatMessage(role="user", content=user_input))

        await ctx.store.set("memory", memory)
        chat_history = memory.get()
        return InputEvent(input=[self.system_message, *chat_history])

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> CodeExecutionEvent | StopEvent:
        chat_history = ev.input

        # --- iteration guard ---
        iteration = await ctx.store.get("iteration", default=0)
        iteration += 1
        await ctx.store.set("iteration", iteration)

        # --- single-shot response (simpler and less truncation-prone than delta assembly) ---
        response = await self.llm.achat(
            chat_history,
        )
        if response is None:
            raise ValueError("LLM returned no response")
        if response.message is None:
            response.message = ChatMessage(role="assistant", content="")

        full_content = response.message.content or ""

        print(f"\n\n===== ITERATION {iteration} =====")

        if "THINK:" in full_content:
            print("---- THINK/ACT ----")

        print(full_content)

        if "ANSWER:" in full_content:
            print("---- FINAL ANSWER DETECTED ----")

        finish_reason = _extract_finish_reason(response)
        print(f"---- FINISH REASON: {finish_reason} ----")

        print("=" * 40 + "\n")

        content = full_content.strip()

        # --- store in memory ---
        memory = await ctx.store.get("memory")
        memory.put(response.message)
        await ctx.store.set("memory", memory)

        # --- metrics ---
        llm_turn_metrics = await ctx.store.get("llm_turn_metrics", default=[])
        usage_metrics = _extract_usage_metrics(response)

        llm_turn_metrics.append(
            {
                "iteration": iteration,
                "iteration_input_tokens": int(usage_metrics.get("prompt_tokens", 0)),
                "iteration_output_tokens": int(usage_metrics.get("completion_tokens", 0)),
                "iteration_total_tokens": int(usage_metrics.get("total_tokens", 0)),
                **(
                    {"iteration_cost_usd": float(usage_metrics["cost_usd"])}
                    if "cost_usd" in usage_metrics
                    else {}
                ),
            }
        )
        await ctx.store.set("llm_turn_metrics", llm_turn_metrics)

        # --- stopping condition ---
        if "ANSWER:" in content or iteration > MAX_ITERATIONS:
            return StopEvent(result=response)

        # --- extract code ---
        code = self._parse_code(content)

        if not code:
            # force model back into loop
            memory.put(
                ChatMessage(
                    role="user",
                    content=(
                        "You must follow THINK → ACT.\n"
                        "Write Python code to proceed.\n"
                        "Or finish with: ANSWER: <integer>"
                    ),
                )
            )
            await ctx.store.set("memory", memory)

            return InputEvent(input=[self.system_message, *memory.get()])

        return CodeExecutionEvent(code=code)

    @step
    async def handle_code_execution(self, ctx: Context, ev: CodeExecutionEvent) -> InputEvent:
        print("\n⚡ EXECUTION STEP TRIGGERED ⚡")

        print("\n[CODE]")
        print(ev.code)
        print("[END CODE]\n")

        output = self.code_execute_fn(ev.code)

        print("[OUTPUT]")
        print(output)
        print("[END OUTPUT]\n")

        memory = await ctx.store.get("memory")

        memory.put(
            ChatMessage(
                role="user",
                content=(
                    "Code execution observation:\n"
                    f"{output}\n\n"
                    "If this is sufficient, respond now with exactly: ANSWER: <integer>. "
                    "If not sufficient, continue with THINK and one Python code block."
                ),
            )
        )

        await ctx.store.set("memory", memory)

        return InputEvent(input=[self.system_message, *memory.get()])


def build_question(threshold: int) -> str:
    return f"""
    There is a list of chemical reactions in SMILES format loaded into a variable `lines`.
    Each reaction is in one of these forms:
      - "index reactants>reagents>products"
      - "index reactants>>products"

    Task:
    Count how many reactions satisfy:
      weight(heaviest product) - weight(heaviest reactant) > {threshold} Da

    Guidance:
    - You may use RDKit functions if needed (for example, Chem.MolFromSmiles and Descriptors.MolWt).
    - "heaviest" means the largest molecular weight among dot-separated molecules on that side.
    - Ignore reagents (middle field).
    - For each side (reactants/products), ignore invalid or empty dot-separated molecules instead of treating them as 0.0.
    - Compute the side's heaviest weight from the remaining valid molecules only.
    - Skip a reaction only if reactant side or product side has no valid molecules left after filtering.
    - Do NOT assign 0.0 as a fallback for missing/invalid molecules.
    - If you copy SMILES text into a Python string literal, handle backslashes safely:
      use a raw string (for example, r\"\"\"...\"\"\") or escape backslashes as "\\\\".

    Output format:
    - Final response must be exactly: ANSWER: <integer>
      Example: ANSWER: 57
    - Do not include additional prose in the final response.
    - If no matching reaction exists, return: ANSWER: 0
"""


async def run_agent_verbose(agent: CodeActAgent, ctx: Context, query: str):
    handler = agent.run(user_input=query, ctx=ctx)
    async for _event in handler.stream_events():
        pass
    return await handler


def build_code_executor(lines: list[str]) -> SimpleCodeExecutor:
    return SimpleCodeExecutor(
        locals={
            "lines": lines,
        },
        globals={
            "__builtins__": __builtins__,
            "np": __import__("numpy"),
            "rdkit": __import__("rdkit"),
        },
    )


async def main() -> None:
    maybe_init_tracing()
    lines = load_lines()
    rng = random.Random(SEED)
    retriever = build_retriever(name=RETRIEVER_NAME, lines=lines, rng=rng)
    run_session_id = f"codeact-task2-{uuid.uuid4()}"

    run = wandb.init(
        project="CodeAct-Task2",
        config={
            "MODEL_NAME": MODEL_NAME,
            "thresholds": THRESHOLDS,
            "dataset_path": DATASET_PATH,
            "workflow_timeout_s": WORKFLOW_TIMEOUT_S,
            "seed": SEED,
            "context_size": CONTEXT_SIZE,
            "retriever_name": RETRIEVER_NAME,
        },
    )
    wandb.define_metric("sample_iteration")
    wandb.define_metric("sample/*", step_metric="sample_iteration")

    correct = 0
    total_cost_usd = 0.0
    samples_with_cost = 0

    for i, threshold in enumerate(THRESHOLDS):
        print(f"Question {i + 1}/{len(THRESHOLDS)} for X={threshold}")
        question = build_question(threshold)
        retrieved_context = retriever.build_context(
            query=f"delta_weight_gt_{threshold}",
            target_index=-1,
            k=CONTEXT_SIZE,
        )
        retrieved_lines = [line for line in retrieved_context.splitlines() if line.strip()]
        context_coverage = len(retrieved_lines) / len(lines) if lines else 0.0
        completion_prompt = f"""
        You are given a subset of chemical reactions in SMILES format and a question.
        <context>
        {retrieved_context}
        </context>
        <question>
        {question}
        </question>
        """
        executor = build_code_executor(lines=retrieved_lines)
        agent = CodeActAgent(
            code_execute_fn=executor.execute,
            llm=OpenRouter(
                model=MODEL_NAME,
                api_key=OPENROUTER_API_KEY,
                max_tokens=MAX_OUTPUT_TOKENS,
                additional_kwargs={"max_completion_tokens": MAX_OUTPUT_TOKENS},
            ),
            timeout=WORKFLOW_TIMEOUT_S,
        )
        ctx = Context(agent)

        with using_tracing_attributes(
            session_id=run_session_id,
            metadata={
                "sample_index": i,
                "sample_count": len(THRESHOLDS),
                "threshold": threshold,
                "agent": "codeact",
            },
            tags=["codeact", "sample", "delta_weight"],
        ):
            print(f"Prompt: {completion_prompt!r}")
            response = await run_agent_verbose(agent, ctx, completion_prompt)
            

        response_text = extract_response_text(response)
        print(f"Raw response: {response_text!r}")
        print("-" * 60)
        llm_turn_metrics = await ctx.store.get("llm_turn_metrics", default=[])
        if not llm_turn_metrics:
            estimated_prompt_tokens = count_tokens(
                [{"role": "user", "content": completion_prompt}],
                MODEL_NAME,
            )
            estimated_completion_tokens = count_tokens(
                [{"role": "assistant", "content": response_text}],
                MODEL_NAME,
            )
            llm_turn_metrics = [
                {
                    "iteration": 1,
                    "iteration_input_tokens": estimated_prompt_tokens,
                    "iteration_output_tokens": estimated_completion_tokens,
                    "iteration_total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
                }
            ]

        parsed_count = parse_count(response_text)
        ground_truth_count = count_matches(retrieved_lines, float(threshold))
        is_correct = parsed_count == ground_truth_count
        if is_correct:
            correct += 1
        else:
            print(f"Mismatch for X={threshold}")
            print(f"Predicted count: {parsed_count}")
            print(f"Ground truth count (retrieved context): {ground_truth_count}")
            print(f"Raw response: {response_text!r}")
            print("-" * 60)

        for metric in llm_turn_metrics:
            wandb.log(
                {
                    "sample_iteration": metric["iteration"],
                    f"sample/{i}/iteration_input_tokens": metric["iteration_input_tokens"],
                    f"sample/{i}/iteration_output_tokens": metric["iteration_output_tokens"],
                    f"sample/{i}/iteration_total_tokens": metric["iteration_total_tokens"],
                    **(
                        {f"sample/{i}/iteration_cost_usd": metric["iteration_cost_usd"]}
                        if "iteration_cost_usd" in metric
                        else {}
                    ),
                }
            )

        final_input_tokens = sum(
            int(metric.get("iteration_input_tokens", 0)) for metric in llm_turn_metrics
        )
        final_output_tokens = sum(
            int(metric.get("iteration_output_tokens", 0)) for metric in llm_turn_metrics
        )
        final_total_tokens = sum(
            int(metric.get("iteration_total_tokens", 0)) for metric in llm_turn_metrics
        )
        final_cost = sum(float(metric.get("iteration_cost_usd", 0.0)) for metric in llm_turn_metrics)
        has_cost = any("iteration_cost_usd" in metric for metric in llm_turn_metrics)
        if has_cost:
            total_cost_usd += final_cost
            samples_with_cost += 1

        wandb.log(
            {
                "sample_idx": i,
                f"sample/{i}/threshold_x": threshold,
                f"sample/{i}/final_total_input_tokens": final_input_tokens,
                f"sample/{i}/final_total_output_tokens": final_output_tokens,
                f"sample/{i}/final_total_tokens": final_total_tokens,
                f"sample/{i}/iterations": len(llm_turn_metrics),
                f"sample/{i}/is_correct": int(is_correct),
                f"sample/{i}/ground_truth_count": ground_truth_count,
                f"sample/{i}/prediction_count": parsed_count if parsed_count is not None else -1,
                f"sample/{i}/response_raw": response_text,
                f"sample/{i}/completion_prompt_char_count": len(completion_prompt),
                f"sample/{i}/context_char_count": len(retrieved_context),
                f"sample/{i}/context_size": CONTEXT_SIZE,
                f"sample/{i}/retrieved_line_count": len(retrieved_lines),
                f"sample/{i}/context_coverage": context_coverage,
                **({f"sample/{i}/final_total_cost_usd": final_cost} if has_cost else {}),
            }
        )
        wandb.log(
            {
                "running_accuracy": correct / (i + 1),
                "running_context_coverage": context_coverage,
            }
        )

    total = len(THRESHOLDS)
    accuracy = (correct / total) if total else 0.0
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f}")

    run.summary["correct"] = correct
    run.summary["total"] = total
    run.summary["accuracy"] = accuracy
    run.summary["samples_with_cost"] = samples_with_cost
    if samples_with_cost > 0:
        run.summary["total_cost_usd"] = total_cost_usd
        run.summary["avg_cost_per_sample_usd"] = total_cost_usd / samples_with_cost
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
