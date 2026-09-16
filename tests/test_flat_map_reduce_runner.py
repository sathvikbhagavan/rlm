from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.iclr2027 import run_flat_map_reduce as runner
from rxnhaystack.map_reduce_tasks import MapReduceQuestion


class FakeRun:
    def __init__(self) -> None:
        self.url = "https://wandb.invalid/run"
        self.summary: dict[str, object] = {}
        self.logged: list[dict[str, object]] = []

    def log(self, values: dict[str, object]) -> None:
        self.logged.append(values)


class FakeLLM:
    async def achat(self, messages):
        prompt = messages[0].content
        answer = "1" if "1 C>>C" in prompt else "-1"
        return SimpleNamespace(
            message=SimpleNamespace(content=answer),
            raw={
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                    "cost": 0.01,
                }
            },
        )


def test_runner_writes_science_usage_cost_and_time_artifacts(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "runs" / "mock-run" / "attempt-001"
    config = SimpleNamespace(
        method="map-reduce",
        task="tier1/task1",
        model="mock/model",
        run_id="mock-run",
        run_dir=run_dir,
        seed=42,
        repetition=1,
        require_dataset=lambda: SimpleNamespace(cleaned=tmp_path / "cleaned.txt"),
    )
    question = MapReduceQuestion(
        question_id="mock-question",
        benchmark_task="tier1/task1",
        question="Find index 1",
        ground_truth_indices=frozenset({1}),
    )
    fake_run = FakeRun()
    captured_metrics = []
    monkeypatch.setattr(runner.WorkerConfig, "from_environment", lambda: config)
    monkeypatch.setattr(runner, "selected_questions", lambda: {question.question_id: question})
    monkeypatch.setattr(runner, "load_lines", lambda _path: ["0 C>>C", "1 C>>C", "2 C>>C"])
    monkeypatch.setattr(runner, "build_benchmark_llm", lambda **_kwargs: FakeLLM())
    monkeypatch.setattr(runner, "benchmark_provider", lambda: "openrouter")
    monkeypatch.setattr(runner.wandb, "init", lambda **_kwargs: fake_run)
    monkeypatch.setattr(runner.wandb, "finish", lambda: None)
    monkeypatch.setattr(
        runner, "write_run_metrics", lambda metrics: captured_metrics.append(metrics)
    )
    monkeypatch.setenv(runner.QUESTION_ID_ENV, question.question_id)
    monkeypatch.setenv(runner.CHUNK_SIZE_ENV, "2")
    monkeypatch.setenv(runner.MAX_PARALLEL_ENV, "1")
    monkeypatch.setenv("RXNHAYSTACK_USD_TO_CHF", "0.8")

    asyncio.run(runner.main())

    details = json.loads((run_dir / "map-reduce-details.json").read_text(encoding="utf-8"))
    assert details["completed_chunk_count"] == 2
    assert details["predicted_indices"] == [1]
    assert details["score"]["f1"] == 1
    assert details["cost_usd"] == 0.02
    assert all(chunk["latency_seconds"] >= 0 for chunk in details["chunks"])
    assert len(captured_metrics) == 1
    metrics = captured_metrics[0]
    assert metrics.calls == 2
    assert metrics.input_tokens == 20
    assert metrics.output_tokens == 2
    assert metrics.cost_usd == 0.02
    assert metrics.cost_chf == 0.016
    assert metrics.latency_seconds >= 0
    assert (run_dir.parent / "map-reduce-checkpoint.json").is_file()
