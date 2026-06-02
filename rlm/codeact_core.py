from __future__ import annotations

import ast
import asyncio
import contextlib
import io
import json
import re
import traceback
from typing import Any, Callable, Dict, Optional

from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

from rlm.codeact_helpers import extract_usage_metrics


DEFAULT_CODEACT_SYSTEM_PROMPT = """
You are a helpful assistant in a CodeAct (Code + Acting) loop that can execute Python code to help you answer questions.
You must follow this format for each step:

1. THINK: Reason about what you need to do next
2. ACT: Take an action (execute code)

AVAILABLE ACTIONS:
- Execute Python code in a fenced block:
```python
CODE...
```
- Provide final answer exactly as: ANSWER: <integer>

RULES:
- For non-final turns, start with THINK and then provide one Python code block.
- Do not simulate execution output.
- After code execution observation, either continue with THINK+code or finalize with ANSWER.
"""


INDEX_CODEACT_SYSTEM_PROMPT = """
You are a helpful assistant in a CodeAct (Code + Acting) loop that can execute Python code to help answer chemistry tasks.
You must follow this format for each step:

1. THINK: Reason about what you need to do next
2. ACT: Take an action (execute code)

AVAILABLE ACTIONS:
- Execute Python code in a fenced block:
```python
CODE...
```
- Provide final answer exactly as:
  ANSWER: <comma-separated indices in ascending order>
  or ANSWER: -1

RULES:
- Variables are persistent across turns. You don't have to declare them again.
- For non-final turns, start with THINK and then provide one Python code block.
- Do not simulate execution output.
- After code execution observation, either continue with THINK+code or finalize with ANSWER.
- Do not declare an answer without checking the code execution output.
"""


DEFAULT_FORCE_LOOP_MESSAGE = (
    "You must follow THINK -> ACT.\n"
    "Write Python code to proceed.\n"
    "Or finish with: ANSWER: <integer>"
)

INDEX_FORCE_LOOP_MESSAGE = (
    "You must follow THINK -> ACT.\n"
    "Write Python code to proceed.\n"
    "Or finish with: ANSWER: <comma-separated indices in ascending order> (or ANSWER: -1)."
)

DEFAULT_OBSERVATION_FOLLOWUP = (
    "If this is sufficient, respond now with exactly: ANSWER: <integer>. "
    "If not sufficient, continue with THINK and one Python code block."
)

INDEX_OBSERVATION_FOLLOWUP = (
    "If this is sufficient, respond now with exactly: "
    "ANSWER: <comma-separated indices in ascending order> or ANSWER: -1. "
    "If not sufficient, continue with THINK and one Python code block."
)


class SimpleCodeExecutor:
    """
    Executes Python code with persistent state.
    NOTE: not safe for production use.
    """

    def __init__(self, locals: Dict[str, Any], globals: Dict[str, Any]):
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
                tree = ast.parse(code, mode="exec")
                if not tree.body:
                    return ""

                last_node = tree.body[-1]
                if isinstance(last_node, ast.Expr):
                    prefix_body = tree.body[:-1]
                    if prefix_body:
                        prefix_module = ast.Module(body=prefix_body, type_ignores=[])
                        ast.fix_missing_locations(prefix_module)
                        exec(
                            compile(prefix_module, filename="<codeact>", mode="exec"),
                            self.namespace,
                            self.namespace,
                        )

                    expr = ast.Expression(last_node.value)
                    ast.fix_missing_locations(expr)
                    return_value = eval(
                        compile(expr, filename="<codeact>", mode="eval"),
                        self.namespace,
                        self.namespace,
                    )
                else:
                    exec(
                        compile(tree, filename="<codeact>", mode="exec"),
                        self.namespace,
                        self.namespace,
                    )

            output = stdout.getvalue()
            if stderr.getvalue():
                output += "\n" + stderr.getvalue()
        except Exception as e:
            output = f"Error: {type(e).__name__}: {str(e)}\n"
            output += traceback.format_exc()

        if return_value is not None:
            output += "\n\n" + str(return_value)
        return output


def make_simple_code_executor(
    extra_locals: Optional[dict[str, Any]] = None,
    extra_globals: Optional[dict[str, Any]] = None,
) -> SimpleCodeExecutor:
    globals_dict: dict[str, Any] = {"__builtins__": __builtins__}
    if extra_globals:
        globals_dict.update(extra_globals)
    locals_dict: dict[str, Any] = {}
    if extra_locals:
        locals_dict.update(extra_locals)
    return SimpleCodeExecutor(locals=locals_dict, globals=globals_dict)


class InputEvent(Event):
    input: list[ChatMessage]


class CodeExecutionEvent(Event):
    code: str


def _extract_finish_reason(response: Any) -> Optional[str]:
    raw = getattr(response, "raw", None)
    if raw is None:
        return None
    choices = raw.get("choices") if isinstance(raw, dict) else getattr(raw, "choices", None)
    if not choices:
        return None
    first = choices[0]
    reason = first.get("finish_reason") if isinstance(first, dict) else getattr(first, "finish_reason", None)
    return str(reason) if reason is not None else None


def _extract_answer_payload(content: str) -> str | None:
    match = re.search(r"(?im)^\s*ANSWER:\s*(.*?)\s*$", content)
    if not match:
        return None
    return match.group(1).strip()


def _has_valid_index_answer(content: str) -> bool:
    payload = _extract_answer_payload(content)
    if payload is None:
        return False
    if payload == "-1":
        return True
    return bool(re.fullmatch(r"\d+(?:\s*,\s*\d+)*", payload))


def _iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_timeout_exception(exc: BaseException) -> bool:
    timeout_type_names = {
        "TimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "APITimeoutError",
    }
    for item in _iter_exception_chain(exc):
        name = type(item).__name__
        if name in timeout_type_names:
            return True
        msg = str(item).lower()
        if "timed out" in msg or "timeout" in msg:
            return True
    return False


def _is_retryable_llm_exception(exc: BaseException) -> bool:
    if _is_timeout_exception(exc):
        return True

    retryable_type_names = {
        "JSONDecodeError",
        "APIConnectionError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "APITimeoutError",
    }
    retryable_status_codes = {408, 409, 425, 429, 500, 502, 503, 504}

    for item in _iter_exception_chain(exc):
        if isinstance(item, json.JSONDecodeError):
            return True

        name = type(item).__name__
        if name in retryable_type_names:
            return True

        status_code = getattr(item, "status_code", None)
        if isinstance(status_code, int) and status_code in retryable_status_codes:
            return True

        msg = str(item).lower()
        if any(
            token in msg
            for token in (
                "expecting value",
                "connection reset",
                "connection aborted",
                "temporarily unavailable",
                "bad gateway",
                "upstream",
                "server error",
                "try again",
            )
        ):
            return True

    return False


class CodeActAgent(Workflow):
    def __init__(
        self,
        code_execute_fn: Callable[[str], str],
        llm: LLM | None = None,
        *,
        system_prompt: str = DEFAULT_CODEACT_SYSTEM_PROMPT,
        max_iterations: int = 8,
        force_loop_message: str = DEFAULT_FORCE_LOOP_MESSAGE,
        observation_followup: str = DEFAULT_OBSERVATION_FOLLOWUP,
        observation_role: str = "user",
        llm_timeout_retries: int = 2,
        llm_timeout_retry_backoff_s: float = 2.0,
        **workflow_kwargs: Any,
    ) -> None:
        super().__init__(**workflow_kwargs)
        self.code_execute_fn = code_execute_fn
        self.llm = llm
        self.max_iterations = max_iterations
        self.force_loop_message = force_loop_message
        self.observation_followup = observation_followup
        self.observation_role = observation_role
        self.llm_timeout_retries = max(0, llm_timeout_retries)
        self.llm_timeout_retry_backoff_s = max(0.0, llm_timeout_retry_backoff_s)
        self.system_message = ChatMessage(role="system", content=system_prompt)

    async def _build_input_messages(self, ctx: Context) -> list[ChatMessage]:
        """Build model input while preserving full chat history.

        We intentionally use get_all() (not get()) so local memory does not trim
        older turns. This keeps the full multi-turn trace visible to the model and
        telemetry unless the upstream model endpoint enforces its own truncation.
        """
        memory = await ctx.store.get("memory")
        messages = memory.get_all()
        initial_user_input = await ctx.store.get("initial_user_input", default=None)
        if initial_user_input is not None:
            if not messages or not (
                messages[0].role == "user" and messages[0].content == initial_user_input
            ):
                messages = [ChatMessage(role="user", content=initial_user_input), *messages]
        return [self.system_message, *messages]

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
        await ctx.store.set("initial_user_input", user_input)
        memory.put(ChatMessage(role="user", content=user_input))
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self._build_input_messages(ctx))

    @step
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> CodeExecutionEvent | StopEvent:
        iteration = await ctx.store.get("iteration", default=0)
        iteration += 1
        await ctx.store.set("iteration", iteration)

        max_attempts = self.llm_timeout_retries + 1
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.llm.achat(ev.input)
                break
            except Exception as exc:
                if not _is_retryable_llm_exception(exc) or attempt >= max_attempts:
                    raise
                sleep_s = self.llm_timeout_retry_backoff_s * (2 ** (attempt - 1))
                print(
                    f"[LLM RETRY] Retryable {type(exc).__name__} on attempt {attempt}/{max_attempts}; "
                    f"retrying in {sleep_s:.1f}s"
                )
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
        if response is None:
            raise ValueError("LLM returned no response")
        if response.message is None:
            response.message = ChatMessage(role="assistant", content="")

        full_content = response.message.content or ""
        print(f"\n\n===== ITERATION {iteration} =====")
        print(full_content)
        print(f"---- FINISH REASON: {_extract_finish_reason(response)} ----")
        print("=" * 40 + "\n")
        content = full_content.strip()

        memory = await ctx.store.get("memory")
        memory.put(response.message)
        await ctx.store.set("memory", memory)

        llm_turn_metrics = await ctx.store.get("llm_turn_metrics", default=[])
        usage_metrics = extract_usage_metrics(response)
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

        if _has_valid_index_answer(content) or iteration > self.max_iterations:
            return StopEvent(result=response)

        code = self._parse_code(content)
        if not code:
            memory.put(ChatMessage(role="user", content=self.force_loop_message))
            await ctx.store.set("memory", memory)
            return InputEvent(input=await self._build_input_messages(ctx))
        return CodeExecutionEvent(code=code)

    @step
    async def handle_code_execution(self, ctx: Context, ev: CodeExecutionEvent) -> InputEvent:
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
                role=self.observation_role,
                content=(
                    "Code execution observation:\n"
                    f"{output}\n\n"
                    f"{self.observation_followup}"
                ),
            )
        )
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self._build_input_messages(ctx))


async def run_agent_verbose(agent: CodeActAgent, ctx: Context, query: str):
    handler = agent.run(user_input=query, ctx=ctx)
    async for _event in handler.stream_events():
        pass
    return await handler
