from __future__ import annotations

import argparse
import fnmatch
import json
import sqlite3
import stat
import subprocess
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rxnhaystack.manifest import ExperimentManifest, load_manifest

MAIN_ROOT = Path("/home/amin/rlm")
CODE_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_ROOT = Path("/home/amin/rlm-qwen-docker-repair-20260920")
DEEPSEEK_ROOT = Path("/home/amin/rlm-deepseek-paid-20260919")
DATA_DIR = Path("/home/amin/datasets/rxnhaystack")
OPENROUTER_KEY = Path("/home/amin/.openrouter_api_key")
SWISSAI_KEY = Path("/home/amin/.swissai_research_api_key")
WANDB_KEY = Path("/home/amin/.wandb_api_key")
STATE_DIR = Path("/home/amin/.local/state/rxnhaystack")
STATE_PATH = STATE_DIR / "local-docker-queue.json"
LOG_PATH = STATE_DIR / "local-docker-queue.log"
OPENROUTER_RESERVE_USD = 20.0


@dataclass(frozen=True)
class Phase:
    name: str
    root: Path
    manifest_relative: Path
    selections: tuple[str, ...]
    expected_runs: int
    title: str
    cost_ceiling_chf: float
    needs_openrouter: bool = True
    needs_swissai: bool = False
    existing_dashboard_reporter: bool = False
    existing_watcher: bool = False

    @property
    def manifest_path(self) -> Path:
        return self.root / self.manifest_relative


PHASES = (
    Phase(
        name="qwen-repair",
        root=EXECUTION_ROOT,
        manifest_relative=Path("experiments/iclr2027/qwen-paid-openrouter-docker-repair.toml"),
        selections=(),
        expected_runs=6,
        title="Qwen 3.5 · RLM · Docker repairs",
        cost_ceiling_chf=25.0,
        existing_dashboard_reporter=True,
        existing_watcher=True,
    ),
    Phase(
        name="prospective-task16",
        root=EXECUTION_ROOT,
        manifest_relative=Path("experiments/iclr2027/prospective-decomposition.toml"),
        selections=(),
        expected_runs=30,
        title="Task 16 · prospective decomposition",
        cost_ceiling_chf=30.0,
        needs_swissai=True,
        existing_dashboard_reporter=True,
    ),
    Phase(
        name="gemini-repair",
        root=EXECUTION_ROOT,
        manifest_relative=Path("experiments/iclr2027/gemini-paid-openrouter-docker-repair.toml"),
        selections=(),
        expected_runs=8,
        title="Gemini 3.7 Flash · RLM · five unresolved Docker cells",
        cost_ceiling_chf=25.0,
    ),
    Phase(
        name="deepseek-docker",
        root=DEEPSEEK_ROOT,
        manifest_relative=Path("experiments/iclr2027/deepseek-paid-openrouter-continuation.toml"),
        selections=(
            "paid-openrouter-full-deepseek-v4-flash-tier4-task16-rlm-*",
            "paid-openrouter-full-deepseek-v4-flash-tier4-task17-rlm-*",
            "paid-openrouter-full-deepseek-v4-flash-tier4-task17b-rlm-*",
        ),
        expected_runs=24,
        title="DeepSeek V4 Flash · RLM · Docker remainder",
        cost_ceiling_chf=20.0,
        existing_dashboard_reporter=True,
    ),
    Phase(
        name="deepseek-rlm-x1000-docker",
        root=EXECUTION_ROOT,
        manifest_relative=Path("experiments/iclr2027/deepseek-rlm-x1000.toml"),
        selections=(
            "x1000-openrouter-full-deepseek-v4-flash-tier4-task16-rlm-*",
            "x1000-openrouter-full-deepseek-v4-flash-tier4-task17-rlm-*",
            "x1000-openrouter-full-deepseek-v4-flash-tier4-task17b-rlm-*",
        ),
        expected_runs=15,
        title="DeepSeek V4 Flash · RLM x1000 · Docker tasks",
        cost_ceiling_chf=10.0,
    ),
    Phase(
        name="glm-docker",
        root=EXECUTION_ROOT,
        manifest_relative=Path("experiments/iclr2027/glm-paid-openrouter-docker.toml"),
        selections=(),
        expected_runs=45,
        title="GLM 5.2 · RLM · Docker tasks",
        cost_ceiling_chf=100.0,
    ),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def log(message: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_state(**values: Any) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": utc_now(), **values}
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE_PATH)


def selected_runs(phase: Phase, manifest: ExperimentManifest) -> list[Any]:
    if not phase.selections:
        return list(manifest.runs)
    return [
        run
        for run in manifest.runs
        if any(fnmatch.fnmatchcase(run.run_id, pattern) for pattern in phase.selections)
    ]


def validate_phase(phase: Phase, *, allow_main_fallback: bool = False) -> ExperimentManifest:
    path = phase.manifest_path
    if allow_main_fallback and not path.is_file():
        path = CODE_ROOT / phase.manifest_relative
    manifest = load_manifest(path)
    runs = selected_runs(phase, manifest)
    if len(runs) != phase.expected_runs:
        raise RuntimeError(f"{phase.name} selects {len(runs)} runs; expected {phase.expected_runs}")
    return manifest


def ledger_counts(phase: Phase, manifest: ExperimentManifest) -> Counter[str]:
    selected = {run.run_id for run in selected_runs(phase, manifest)}
    ledger = manifest.campaign.artifact_dir / "ledger.sqlite3"
    counts: Counter[str] = Counter()
    if not ledger.is_file():
        counts["pending"] = len(selected)
        return counts
    connection = sqlite3.connect(ledger)
    try:
        observed = {
            str(run_id): str(status)
            for run_id, status in connection.execute("SELECT run_id, status FROM runs")
            if str(run_id) in selected
        }
    finally:
        connection.close()
    for run_id in selected:
        counts[observed.get(run_id, "pending")] += 1
    return counts


def verify_secret(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if not path.is_file() or mode & 0o077:
        raise RuntimeError(f"Credential must be a regular mode-0600 file: {path}")


def openrouter_balance() -> tuple[float, float]:
    key = OPENROUTER_KEY.read_text(encoding="utf-8").strip()

    def read(endpoint: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"https://openrouter.ai/api/v1/{endpoint}",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return json.load(response)["data"]

    credits = read("credits")
    key_status = read("auth/key")
    account_available = float(credits["total_credits"]) - float(credits["total_usage"])
    return account_available, float(key_status["limit_remaining"])


def wait_for_balance(phase: Phase, poll_seconds: int) -> None:
    if not phase.needs_openrouter:
        return
    while True:
        try:
            account, key = openrouter_balance()
        except Exception as error:  # Network failure is a hold, never permission to spend.
            log(f"HOLD {phase.name}: cannot verify OpenRouter balance: {error}")
        else:
            log(
                f"BALANCE {phase.name}: account=${account:.2f}, key=${key:.2f}, "
                f"reserve=${OPENROUTER_RESERVE_USD:.2f}"
            )
            if account > OPENROUTER_RESERVE_USD and key > OPENROUTER_RESERVE_USD:
                return
            write_state(
                status="waiting-for-openrouter-balance",
                phase=phase.name,
                account_available_usd=account,
                key_remaining_usd=key,
            )
        time.sleep(poll_seconds)


def run_checked(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def prepare_execution_root(commit: str) -> None:
    run_checked(["git", "fetch", "origin", "main"], cwd=EXECUTION_ROOT)
    run_checked(["git", "checkout", "--detach", commit], cwd=EXECUTION_ROOT)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=EXECUTION_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"Execution checkout has tracked edits: {dirty}")
    run_checked(["uv", "sync", "--frozen"], cwd=EXECUTION_ROOT)

    # Continue the already-successful prospective pilot from its authoritative ledger.
    source = MAIN_ROOT / "artifacts/iclr2027-task16-prospective-decomposition-v1"
    destination = EXECUTION_ROOT / "artifacts/iclr2027-task16-prospective-decomposition-v1"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.symlink_to(source, target_is_directory=True)
    elif destination.resolve() != source.resolve():
        raise RuntimeError(f"Prospective artifact destination is not authoritative: {destination}")


def runner_command(phase: Phase) -> list[str]:
    command = [
        "uv",
        "run",
        "--frozen",
        "rxnhaystack",
        "run",
        str(phase.manifest_relative),
    ]
    for pattern in phase.selections:
        command.extend(("--select", pattern))
    command.extend(
        (
            "--data-dir",
            str(DATA_DIR),
            "--max-parallel",
            "1",
            "--max-run-seconds",
            "21600",
            "--secret-file",
            f"OPENROUTER_API_KEY={OPENROUTER_KEY}",
            "--secret-file",
            f"WANDB_API_KEY={WANDB_KEY}",
        )
    )
    if phase.needs_swissai:
        command.extend(("--secret-file", f"SWISSAI_RESEARCH_API_KEY={SWISSAI_KEY}"))
    return command


def dashboard_command(phase: Phase, manifest: ExperimentManifest, *, watch: bool) -> list[str]:
    command = [
        "uv",
        "run",
        "--frozen",
        "rxnhaystack",
        "dashboard",
        "update",
        str(phase.manifest_relative),
        "--ledger-path",
        str(manifest.campaign.artifact_dir / "ledger.sqlite3"),
        "--source-id",
        f"liacpc14-{phase.name}-queue-v1",
        "--machine",
        "liacpc14",
        "--owner",
        "Amin",
        "--entity",
        "liac",
        "--project",
        "rxnhaystack-dashboard",
        "--secret-file",
        f"WANDB_API_KEY={WANDB_KEY}",
    ]
    if watch:
        command.extend(("--watch-seconds", "300", "--heartbeat-seconds", "1800"))
    return command


def watcher_command(phase: Phase, manifest: ExperimentManifest) -> list[str]:
    command = [
        "task-notify",
        "watch-ledger",
        "--watch-id",
        f"rxnhaystack-local-queue-{phase.name}-v1",
        "--title",
        phase.title,
        "--description",
        "One-attempt local Docker phase; failures are preserved and the queue advances",
        "--project",
        "RxnHaystack",
        "--machine",
        "liacpc14",
        "--chat-title",
        "RxnHaystack ICLR 2027 master chat",
        "--chat-machine",
        "liacpc14",
        "--chat-id",
        "01a07b54-b24f-7771-8741-38b4bf67c5e6",
        "--ledger",
        str(manifest.campaign.artifact_dir / "ledger.sqlite3"),
        "--log-file",
        str(LOG_PATH),
        "--cost-ceiling-chf",
        str(phase.cost_ceiling_chf),
        "--poll-seconds",
        "60",
        "--status-interval-seconds",
        "1800",
        "--stall-after-seconds",
        "7200",
        "--timeout-seconds",
        "604800",
    ]
    for pattern in phase.selections:
        command.extend(("--select", pattern))
    return command


def stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_phase_once(phase: Phase, manifest: ExperimentManifest) -> int:
    counts = ledger_counts(phase, manifest)
    if counts["pending"] == 0 and counts["running"] == 0:
        log(f"SKIP {phase.name}: already terminal {dict(counts)}")
        return 0
    wait_for_balance(phase, poll_seconds=600)
    log(f"START {phase.name}: {dict(counts)}")
    write_state(status="running", phase=phase.name, counts=dict(counts))
    phase_log = STATE_DIR / f"local-docker-queue-{phase.name}.log"
    with phase_log.open("a", encoding="utf-8") as output:
        runner = subprocess.Popen(
            runner_command(phase),
            cwd=phase.root,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ledger = manifest.campaign.artifact_dir / "ledger.sqlite3"
        for _ in range(60):
            if ledger.exists() or runner.poll() is not None:
                break
            time.sleep(1)
        watcher: subprocess.Popen[Any] | None = None
        reporter: subprocess.Popen[Any] | None = None
        if ledger.exists() and not phase.existing_watcher:
            watcher = subprocess.Popen(
                watcher_command(phase, manifest),
                cwd=phase.root,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if not phase.existing_dashboard_reporter:
                reporter = subprocess.Popen(
                    dashboard_command(phase, manifest, watch=True),
                    cwd=phase.root,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        return_code = runner.wait()
        stop_process(watcher)
        stop_process(reporter)
    if (manifest.campaign.artifact_dir / "ledger.sqlite3").exists():
        try:
            subprocess.run(
                dashboard_command(phase, manifest, watch=False),
                cwd=phase.root,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            log(f"WARN {phase.name}: final dashboard publication timed out")
    final_counts = ledger_counts(phase, manifest)
    log(f"FINISH {phase.name}: rc={return_code}, {dict(final_counts)}")
    write_state(
        status="phase-finished",
        phase=phase.name,
        return_code=return_code,
        counts=dict(final_counts),
    )
    return return_code


def wait_for_existing_qwen(poll_seconds: int) -> None:
    phase = PHASES[0]
    manifest = validate_phase(phase)
    while True:
        counts = ledger_counts(phase, manifest)
        if counts["running"] == 0:
            # Give the existing fail-closed shell time to release its broad run.
            time.sleep(min(60, poll_seconds))
            counts = ledger_counts(phase, manifest)
            if counts["running"] == 0:
                log(f"Qwen existing launcher quiescent: {dict(counts)}")
                return
        write_state(status="waiting-for-qwen", phase=phase.name, counts=dict(counts))
        time.sleep(poll_seconds)


def wait_for_phase_terminal(phase: Phase, poll_seconds: int) -> None:
    """Wait for an already-running phase without starting a duplicate runner."""
    manifest = validate_phase(phase, allow_main_fallback=True)
    while True:
        counts = ledger_counts(phase, manifest)
        if counts["running"] == 0 and counts["pending"] == 0:
            log(f"Existing {phase.name} phase terminal: {dict(counts)}")
            return
        write_state(
            status="waiting-for-existing-phase",
            phase=phase.name,
            counts=dict(counts),
        )
        time.sleep(poll_seconds)


def check_configuration() -> None:
    verify_secret(OPENROUTER_KEY)
    verify_secret(SWISSAI_KEY)
    verify_secret(WANDB_KEY)
    run_checked(["docker", "info"], cwd=MAIN_ROOT)
    for phase in PHASES:
        manifest = validate_phase(phase, allow_main_fallback=True)
        print(f"{phase.name}: {len(selected_runs(phase, manifest))} exact runs")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local Docker queue once per cell.")
    parser.add_argument("--commit", help="Exact repository commit used for queued phases")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--wait-for-prospective-terminal",
        action="store_true",
        help="Adopt the active prospective runner and wait instead of launching a duplicate",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.check:
        check_configuration()
        return 0
    if not args.commit:
        parser.error("--commit is required outside --check")
    check_configuration()
    phase_state = []
    for phase in PHASES:
        values = asdict(phase)
        values["root"] = str(values["root"])
        values["manifest_relative"] = str(values["manifest_relative"])
        phase_state.append(values)
    write_state(status="armed", commit=args.commit, phases=phase_state)

    wait_for_existing_qwen(args.poll_seconds)
    if args.wait_for_prospective_terminal:
        prospective = next(phase for phase in PHASES if phase.name == "prospective-task16")
        wait_for_phase_terminal(prospective, args.poll_seconds)
    prepare_execution_root(args.commit)
    # The controller owns any still-pending Qwen cells exactly once. Existing
    # Existing successes and failures are skipped; only never-attempted cells run.
    for phase in PHASES:
        manifest = validate_phase(phase)
        run_phase_once(phase, manifest)

    write_state(status="complete", phase=None)
    log("COMPLETE local Docker queue")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("INTERRUPTED local Docker queue")
        raise
    except Exception as error:
        write_state(status="failed", error=str(error))
        log(f"FATAL local Docker queue: {error}")
        raise
