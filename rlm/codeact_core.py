from __future__ import annotations

import ast
import asyncio
import contextlib
import html
import importlib
import io
import json
import multiprocessing
import os
import re
import resource
import signal
import threading
import time
import traceback
from collections.abc import Callable
from types import ModuleType
from typing import Any

from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

from rlm.codeact_helpers import extract_usage_metrics

DEFAULT_CODEACT_SYSTEM_PROMPT = """
You are a helpful assistant in a CodeAct (Code + Acting) loop that can execute Python code to help you answer questions.
You must follow this format for each step:

1. THINK: Reason about what you need to do next
2. ACT: Take an action (execute code)

PRELOADED DATA:
- `lines` is a list containing exactly the retrieved reaction-context rows for
  this question.
- Use `lines` directly in Python. Do not copy the <context> text into your code.

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
- Provide the final answer as `ANSWER:` followed by the exact output format
  requested in the question. The requested output may contain one or more lines.

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
    "Or finish with ANSWER: followed by the exact output requested in the question."
)

DEFAULT_OBSERVATION_FOLLOWUP = (
    "If this is sufficient, respond now with exactly: ANSWER: <integer>. "
    "If not sufficient, continue with THINK and one Python code block."
)

INDEX_OBSERVATION_FOLLOWUP = (
    "If this is sufficient, respond now with ANSWER: followed by the exact output "
    "requested in the question. "
    "If not sufficient, continue with THINK and one Python code block."
)

FINAL_ANSWER_REQUIRED = (
    "This was the last allowed tool action. Do not write or request more code. "
    "Respond now with ANSWER: followed by the exact output requested in the "
    "question. Include no reasoning or explanation."
)
FINAL_ANSWER_ATTEMPTS = 2

PRELOADED_LINES_REMINDER = """<tool-data-reminder>
The exact retrieved context rows are already available in Python as the list
`lines`. Use `lines` directly. Never copy or redefine the <context> rows in
generated code.
</tool-data-reminder>"""

CODEACT_TOOL_TIMEOUT_ENV = "RXNHAYSTACK_CODEACT_TOOL_TIMEOUT_SECONDS"
CODEACT_TOOL_TIMEOUT_SECONDS = 60.0
CODEACT_TOOL_MEMORY_LIMIT_ENV = "RXNHAYSTACK_CODEACT_TOOL_MEMORY_LIMIT_MIB"
CODEACT_TOOL_MEMORY_LIMIT_MIB = 4096


def append_preloaded_lines_reminder(user_input: str) -> str:
    """Keep the tool-data instruction near the question in long CodeAct prompts."""

    return f"{user_input.rstrip()}\n\n{PRELOADED_LINES_REMINDER}"


def parse_code_action(response: str) -> str | None:
    """Extract executable Python from supported CodeAct response wrappers.

    The benchmark asks every model for a fenced Python block. Some Anthropic
    models instead render the same requested action using their textual
    ``execute_python`` tool-call wrapper. Accepting that wrapper keeps the
    controller provider-neutral while executing only explicitly named code
    parameters.
    """

    fenced_matches = re.findall(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fenced_matches:
        return "\n\n".join(block.strip() for block in fenced_matches if block.strip())

    anthropic_matches = re.findall(
        r"<invoke\b[^>]*\bname\s*=\s*[\"']execute_python[\"'][^>]*>"
        r".*?<parameter\b[^>]*\bname\s*=\s*[\"']code[\"'][^>]*>"
        r"(.*?)</parameter>.*?</invoke>",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if anthropic_matches:
        return "\n\n".join(
            html.unescape(block).strip() for block in anthropic_matches if block.strip()
        )
    return None


class SimpleCodeExecutor:
    """
    Executes Python code with persistent state.
    NOTE: not safe for production use.
    """

    def __init__(self, locals: dict[str, Any], globals: dict[str, Any]):
        self.namespace: dict[str, Any] = {}
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


def _isolated_executor_main(
    connection,
    extra_locals: dict[str, Any],
    extra_globals: dict[str, Any],
    module_globals: dict[str, str],
    memory_limit_mib: int,
) -> None:
    """Serve persistent code execution inside a killable process group."""

    os.setsid()
    limit_bytes = memory_limit_mib * 1024 * 1024
    _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    if hard_limit != resource.RLIM_INFINITY:
        limit_bytes = min(limit_bytes, hard_limit)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard_limit))
    for alias, module_name in module_globals.items():
        extra_globals[alias] = importlib.import_module(module_name)
    executor = SimpleCodeExecutor(locals=extra_locals, globals=extra_globals)
    try:
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "stop":
                return
            if command != "execute":
                connection.send(("error", f"Unknown executor command: {command}"))
                continue
            connection.send(("result", executor.execute(payload)))
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


class IsolatedCodeExecutor:
    """Persistent CodeAct namespace in a process that can be truly terminated."""

    def __init__(
        self,
        *,
        extra_locals: dict[str, Any],
        extra_globals: dict[str, Any],
        timeout_s: float,
        memory_limit_mib: int,
    ) -> None:
        self.extra_locals = extra_locals
        self.extra_globals: dict[str, Any] = {}
        self.module_globals: dict[str, str] = {}
        for alias, value in extra_globals.items():
            if isinstance(value, ModuleType):
                self.module_globals[alias] = value.__name__
            else:
                self.extra_globals[alias] = value
        self.timeout_s = timeout_s
        self.memory_limit_mib = memory_limit_mib
        self._context = multiprocessing.get_context("spawn")
        self._process: multiprocessing.Process | None = None
        self._connection = None
        self._lock = threading.Lock()
        self._start()

    def _start(self) -> None:
        parent_connection, child_connection = self._context.Pipe()
        process = self._context.Process(
            target=_isolated_executor_main,
            args=(
                child_connection,
                self.extra_locals,
                self.extra_globals,
                self.module_globals,
                self.memory_limit_mib,
            ),
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process
        try:
            if not parent_connection.poll(30):
                raise RuntimeError("Isolated CodeAct process did not start within 30 seconds")
            status, payload = parent_connection.recv()
            if status != "ready":
                raise RuntimeError(f"Isolated CodeAct process failed to start: {payload}")
        except (BrokenPipeError, EOFError, OSError, RuntimeError):
            exit_code = process.exitcode
            self._stop()
            raise RuntimeError(
                f"Isolated CodeAct process failed during startup (exit code {exit_code})"
            ) from None

    def _stop(self) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is None:
            return
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()

    def _reset(self) -> None:
        self._stop()
        self._start()

    def execute(self, code: str) -> str:
        with self._lock:
            if self._process is None or not self._process.is_alive():
                self._reset()
            assert self._process is not None
            assert self._connection is not None
            try:
                self._connection.send(("execute", code))
                if not self._connection.poll(self.timeout_s):
                    self._reset()
                    return (
                        format_code_execution_timeout_error(self.timeout_s)
                        + "\nThe isolated Python state was reset after the timeout."
                    )
                status, payload = self._connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                exit_code = self._process.exitcode
                self._reset()
                return (
                    "Error: RuntimeError: The isolated code process exited "
                    f"unexpectedly (exit code {exit_code}).\n"
                    "The isolated Python state was reset; use a safer approach."
                )
            if status != "result":
                return f"Error: RuntimeError: {payload}"
            return str(payload)

    def cleanup(self) -> None:
        with self._lock:
            self._stop()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


def format_code_execution_timeout_error(timeout_s: float) -> str:
    return (
        f"Error: TimeoutError: Code execution exceeded {timeout_s:.1f}s timeout.\n"
        "The code likely blocked on I/O (e.g., sys.stdin.read()) or ran too long.\n"
        "Traceback (most recent call last):\n"
        '  File "<codeact>", line ?, in <module>\n'
        f"TimeoutError: Code execution exceeded {timeout_s:.1f}s timeout"
    )


def _tool_isolation_failure(output: str) -> bool:
    return output.startswith("Error: TimeoutError: Code execution exceeded") or output.startswith(
        "Error: RuntimeError: The isolated code process exited unexpectedly"
    )


def _reported_tool_timeout(output: str) -> str | None:
    match = re.search(r"Code execution exceeded ([0-9.]+)s timeout", output)
    return match.group(1) if match else None


async def execute_code_with_timeout(
    code_execute_fn: Callable[[str], str],
    code: str,
    timeout_s: float | None,
) -> str:
    if timeout_s is None:
        return code_execute_fn(code)
    try:
        return await asyncio.wait_for(asyncio.to_thread(code_execute_fn, code), timeout=timeout_s)
    except TimeoutError:
        return format_code_execution_timeout_error(timeout_s)


def make_simple_code_executor(
    extra_locals: dict[str, Any] | None = None,
    extra_globals: dict[str, Any] | None = None,
) -> SimpleCodeExecutor | IsolatedCodeExecutor:
    globals_dict: dict[str, Any] = {"__builtins__": __builtins__}
    if extra_globals:
        globals_dict.update(extra_globals)
    locals_dict: dict[str, Any] = {}
    if extra_locals:
        locals_dict.update(extra_locals)
    if os.environ.get("RXNHAYSTACK_RUN_ID"):
        return IsolatedCodeExecutor(
            extra_locals=locals_dict,
            extra_globals=globals_dict,
            timeout_s=_positive_float_from_environment(
                CODEACT_TOOL_TIMEOUT_ENV, CODEACT_TOOL_TIMEOUT_SECONDS
            ),
            memory_limit_mib=_positive_int_from_environment(
                CODEACT_TOOL_MEMORY_LIMIT_ENV, CODEACT_TOOL_MEMORY_LIMIT_MIB
            ),
        )
    return SimpleCodeExecutor(locals=locals_dict, globals=globals_dict)


def _positive_float_from_environment(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _positive_int_from_environment(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class InputEvent(Event):
    input: list[ChatMessage]


class CodeExecutionEvent(Event):
    code: str


def _extract_finish_reason(response: Any) -> str | None:
    raw = getattr(response, "raw", None)
    if raw is None:
        return None
    choices = raw.get("choices") if isinstance(raw, dict) else getattr(raw, "choices", None)
    if not choices:
        return None
    first = choices[0]
    reason = (
        first.get("finish_reason")
        if isinstance(first, dict)
        else getattr(first, "finish_reason", None)
    )
    return str(reason) if reason is not None else None


def _extract_answer_payload(content: str) -> str | None:
    match = re.search(r"(?im)^\s*ANSWER:\s*(.*?)\s*$", content)
    if not match:
        return None
    return match.group(1).strip()


def _has_final_answer(content: str) -> bool:
    """Recognize the CodeAct protocol marker without assuming a task's shape.

    Some benchmark tasks return one index, while others return several lines of
    pairs or paths. Their task-specific parsers remain responsible for validating
    the payload.
    """

    return _extract_answer_payload(content) is not None


def _continuation_instruction(
    *, iteration: int, max_iterations: int, normal_instruction: str
) -> str:
    """Require an answer after the final permitted tool/reasoning turn."""

    if iteration >= max_iterations:
        return FINAL_ANSWER_REQUIRED
    return normal_instruction


def _is_answer_only_turn(*, iteration: int, max_iterations: int) -> bool:
    return iteration > max_iterations


def _answer_only_attempts_exhausted(*, iteration: int, max_iterations: int) -> bool:
    return iteration >= max_iterations + FINAL_ANSWER_ATTEMPTS


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
        llm_request_timeout_s: float = 300.0,
        code_execution_timeout_s: float = 300.0,
        memory_token_limit: int = 120_000,
        on_llm_start: Callable[[int], None] | None = None,
        on_llm_complete: Callable[[int, float, bool], None] | None = None,
        on_llm_usage: Callable[[int, dict[str, float | int]], None] | None = None,
        on_tool_start: Callable[[int], None] | None = None,
        on_tool_complete: Callable[[int, float, bool], None] | None = None,
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
        self.llm_request_timeout_s = (
            float(llm_request_timeout_s) if llm_request_timeout_s > 0 else None
        )
        self.code_execution_timeout_s = (
            float(code_execution_timeout_s) if code_execution_timeout_s > 0 else None
        )
        self.memory_token_limit = max(1024, memory_token_limit)
        self.on_llm_start = on_llm_start
        self.on_llm_complete = on_llm_complete
        self.on_llm_usage = on_llm_usage
        self.on_tool_start = on_tool_start
        self.on_tool_complete = on_tool_complete
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
        return parse_code_action(response)

    @step
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        memory = await ctx.store.get("memory", default=None)
        if not memory:
            try:
                memory = ChatMemoryBuffer.from_defaults(llm=self.llm)
            except ValueError as exc:
                # Some OpenAI-compatible providers use model IDs that LlamaIndex's OpenAI
                # metadata mapper does not recognize (e.g., SwissAI Qwen identifiers).
                # Fall back to an explicit token limit instead of model-name lookup.
                if "Unknown model" not in str(exc):
                    raise
                print(
                    f"[MEMORY] Unknown model metadata lookup; "
                    f"falling back to token_limit={self.memory_token_limit}"
                )
                memory = ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit)
        user_input = ev.get("user_input")
        if user_input is None:
            raise ValueError("user_input kwarg is required")
        user_input = append_preloaded_lines_reminder(user_input)
        await ctx.store.set("initial_user_input", user_input)
        memory.put(ChatMessage(role="user", content=user_input))
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self._build_input_messages(ctx))

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> CodeExecutionEvent | StopEvent:
        iteration = await ctx.store.get("iteration", default=0)
        iteration += 1
        await ctx.store.set("iteration", iteration)

        max_attempts = self.llm_timeout_retries + 1
        response = None
        llm_started = time.perf_counter()
        if self.on_llm_start is not None:
            self.on_llm_start(iteration)
        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    if self.llm_request_timeout_s is not None:
                        response = await asyncio.wait_for(
                            self.llm.achat(ev.input), timeout=self.llm_request_timeout_s
                        )
                    else:
                        response = await self.llm.achat(ev.input)
                    break
                except Exception as exc:
                    if not _is_retryable_llm_exception(exc) or attempt >= max_attempts:
                        raise
                    sleep_s = self.llm_timeout_retry_backoff_s * (2 ** (attempt - 1))
                    print(
                        f"[LLM RETRY] Retryable {type(exc).__name__} on attempt "
                        f"{attempt}/{max_attempts}; retrying in {sleep_s:.1f}s"
                    )
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
        except Exception:
            if self.on_llm_complete is not None:
                self.on_llm_complete(iteration, time.perf_counter() - llm_started, True)
            raise
        llm_duration = time.perf_counter() - llm_started
        if self.on_llm_complete is not None:
            self.on_llm_complete(iteration, llm_duration, False)
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
        if self.on_llm_usage is not None:
            self.on_llm_usage(iteration, usage_metrics)
        llm_turn_metrics.append(
            {
                "iteration": iteration,
                "iteration_input_tokens": int(usage_metrics.get("prompt_tokens", 0)),
                "iteration_output_tokens": int(usage_metrics.get("completion_tokens", 0)),
                "iteration_total_tokens": int(usage_metrics.get("total_tokens", 0)),
                "iteration_latency_seconds": llm_duration,
                **(
                    {"iteration_cost_usd": float(usage_metrics["cost_usd"])}
                    if "cost_usd" in usage_metrics
                    else {}
                ),
            }
        )
        await ctx.store.set("llm_turn_metrics", llm_turn_metrics)

        if _has_final_answer(content) or _answer_only_attempts_exhausted(
            iteration=iteration, max_iterations=self.max_iterations
        ):
            return StopEvent(result=response)

        if _is_answer_only_turn(iteration=iteration, max_iterations=self.max_iterations):
            memory.put(ChatMessage(role="user", content=FINAL_ANSWER_REQUIRED))
            await ctx.store.set("memory", memory)
            return InputEvent(input=await self._build_input_messages(ctx))

        code = self._parse_code(content)
        if not code:
            correction = _continuation_instruction(
                iteration=iteration,
                max_iterations=self.max_iterations,
                normal_instruction=self.force_loop_message,
            )
            memory.put(ChatMessage(role="user", content=correction))
            await ctx.store.set("memory", memory)
            return InputEvent(input=await self._build_input_messages(ctx))
        return CodeExecutionEvent(code=code)

    @step
    async def handle_code_execution(self, ctx: Context, ev: CodeExecutionEvent) -> InputEvent:
        print("\n[CODE]")
        print(ev.code)
        print("[END CODE]\n")
        iteration = await ctx.store.get("iteration", default=0)
        tool_started = time.perf_counter()
        if self.on_tool_start is not None:
            self.on_tool_start(iteration)
        try:
            output = await execute_code_with_timeout(
                self.code_execute_fn,
                ev.code,
                self.code_execution_timeout_s,
            )
        except Exception:
            if self.on_tool_complete is not None:
                self.on_tool_complete(iteration, time.perf_counter() - tool_started, True)
            raise
        tool_duration = time.perf_counter() - tool_started
        tool_failed = _tool_isolation_failure(output)
        if self.on_tool_complete is not None:
            self.on_tool_complete(iteration, tool_duration, tool_failed)
        tool_turn_metrics = await ctx.store.get("tool_turn_metrics", default=[])
        tool_turn_metrics.append(
            {
                "iteration": iteration,
                "tool_time_seconds": tool_duration,
                "tool_failed": tool_failed,
            }
        )
        await ctx.store.set("tool_turn_metrics", tool_turn_metrics)
        reported_timeout = _reported_tool_timeout(output)
        if reported_timeout is not None:
            print(f"[CODE EXEC TIMEOUT] isolated process stopped after {reported_timeout}s")
        print("[OUTPUT]")
        print(output)
        print("[END OUTPUT]\n")

        memory = await ctx.store.get("memory")
        followup = _continuation_instruction(
            iteration=iteration,
            max_iterations=self.max_iterations,
            normal_instruction=self.observation_followup,
        )
        memory.put(
            ChatMessage(
                role=self.observation_role,
                content=(f"Code execution observation:\n{output}\n\n{followup}"),
            )
        )
        await ctx.store.set("memory", memory)
        return InputEvent(input=await self._build_input_messages(ctx))


async def run_agent_verbose(agent: CodeActAgent, ctx: Context, query: str):
    try:
        handler = agent.run(user_input=query, ctx=ctx)
        async for _event in handler.stream_events():
            pass
        return await handler
    finally:
        executor = getattr(agent.code_execute_fn, "__self__", None)
        cleanup = getattr(executor, "cleanup", None)
        if callable(cleanup):
            cleanup()
