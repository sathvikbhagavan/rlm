from __future__ import annotations

import subprocess
from pathlib import Path

from rlm.environments.docker_repl import (
    container_cgroup_path,
    docker_exec_command,
    publish_container_cgroup,
)


def test_container_cgroup_path_reads_unified_linux_cgroup(
    tmp_path: Path, monkeypatch
) -> None:
    proc_root = tmp_path / "proc"
    process_dir = proc_root / "4321"
    process_dir.mkdir(parents=True)
    (process_dir / "cgroup").write_text(
        "7:cpu,cpuacct:/legacy\n0::/system.slice/docker-example.scope\n"
    )
    monkeypatch.setattr(
        "rlm.environments.docker_repl.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="4321\n", stderr=""
        ),
    )

    assert (
        container_cgroup_path("example", proc_root=proc_root)
        == "/system.slice/docker-example.scope"
    )


def test_publish_container_cgroup_appends_without_overwriting(
    tmp_path: Path, monkeypatch
) -> None:
    registry = tmp_path / "docker-cgroups.txt"
    registry.write_text("/existing\n")
    monkeypatch.setattr(
        "rlm.environments.docker_repl.container_cgroup_path",
        lambda _container_id: "/new-container",
    )

    publish_container_cgroup("example", registry)

    assert registry.read_text() == "/existing\n/new-container\n"


def test_docker_exec_command_applies_in_container_process_group_timeout() -> None:
    assert docker_exec_command(
        "container-id", "print(42)", execution_timeout_seconds=300
    ) == [
        "docker",
        "exec",
        "container-id",
        "timeout",
        "--verbose",
        "--signal=TERM",
        "--kill-after=5s",
        "300s",
        "python",
        "-c",
        "print(42)",
    ]
