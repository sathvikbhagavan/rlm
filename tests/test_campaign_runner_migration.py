from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runners(method: str, *, include_tier1: bool = False) -> list[Path]:
    tiers = ("tier1", "tier2", "tier3", "tier4") if include_tier1 else ("tier2", "tier3", "tier4")
    return [
        path
        for tier in tiers
        for path in sorted((ROOT / tier).glob(f"{method}_task*.py"))
    ]


def parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }


def enclosing_function(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def test_all_campaign_runners_install_metrics_and_are_portable() -> None:
    paths = runners("llm") + runners("codeact") + runners("rlm")
    assert len(paths) == 99
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "install_campaign_metrics(wandb)" in source, path
        assert "/home/bhagavan" not in source, path
        assert "RXNHAYSTACK_MODEL" in source, path
        assert "RXNHAYSTACK_SEED" in source, path
        assert "RXNHAYSTACK_CONTEXT_SIZE" in source, path
        assert "RXNHAYSTACK_CLEANED_DATASET" in source, path


def test_every_multi_question_async_runner_uses_bounded_mapping() -> None:
    migrated = 0
    for path in runners("llm") + runners("codeact"):
        tree = ast.parse(path.read_text(encoding="utf-8"), path)
        parents = parent_map(tree)
        target = "llm.achat" if path.name.startswith("llm_") else "run_agent_verbose"
        for node in ast.walk(tree):
            if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
                continue
            if ast.unparse(node.value.func) != target:
                continue
            function = enclosing_function(node, parents)
            if function is not None and function.name == "_evaluate_question":
                migrated += 1
                assert any(
                    isinstance(call, ast.Call)
                    and ast.unparse(call.func) == "map_async_bounded"
                    for call in ast.walk(tree)
                ), path
                assert "question_parallelism_from_environment" in path.read_text(), path
    assert migrated == 36


def test_codeact_questions_own_agents_contexts_and_timing_callbacks() -> None:
    for path in runners("codeact"):
        tree = ast.parse(path.read_text(encoding="utf-8"), path)
        parents = parent_map(tree)
        agent_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "CodeActAgent"
        ]
        assert len(agent_calls) == 1, path
        call = agent_calls[0]
        assert any(
            keyword.arg is None
            and "codeact_callbacks_from_environment" in ast.unparse(keyword.value)
            for keyword in call.keywords
        ), path
        function = enclosing_function(call, parents)
        awaited = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "run_agent_verbose"
        )
        assert enclosing_function(awaited, parents) is function, path
        if function is not None and function.name == "_evaluate_question":
            names = {
                ast.unparse(node.func)
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
            }
            assert "Context" in names, path
            assert "build_benchmark_llm" in names, path


def test_every_codeact_tool_preloads_only_its_retrieved_context() -> None:
    paths = runners("codeact", include_tier1=True)
    assert len(paths) == 34
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), path)
        builders = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "build_code_executor"
        ]
        assert len(builders) == 1, path
        builder = builders[0]
        assert "lines" in {argument.arg for argument in builder.args.args}, path
        assert any(
            isinstance(node, ast.Dict)
            and any(
                isinstance(key, ast.Constant)
                and key.value == "lines"
                and isinstance(value, ast.Name)
                and value.id == "lines"
                for key, value in zip(node.keys, node.values, strict=True)
            )
            for node in ast.walk(builder)
        ), path

        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "build_code_executor"
        ]
        assert len(calls) == 1, path
        assert "retrieved_lines" in ast.unparse(calls[0]), path


def test_all_rlm_instances_receive_campaign_instrumentation() -> None:
    for path in runners("rlm"):
        tree = ast.parse(path.read_text(encoding="utf-8"), path)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "RLM"
        ]
        assert calls, path
        assert all(
            any(
                keyword.arg is None
                and "instrument_rlm_from_environment" in ast.unparse(keyword.value)
                for keyword in call.keywords
            )
            for call in calls
        ), path
