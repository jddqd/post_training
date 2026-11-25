import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from helpers import DOUBLE
from minisweagent.exceptions import Submitted

from rl_swe_agent.config import compose, mini_config
from rl_swe_agent.sandbox.sandbox import Sandbox
from rl_swe_agent.sandbox.verifier import docker_ready
from rl_swe_agent.tasks.workspace import build_workspace

ENVIRONMENT = mini_config(compose())["environment"]  # commands killed after 10 s
pytestmark = pytest.mark.skipif(not docker_ready(), reason="needs Docker and python:3.12-slim")

TASK = DOUBLE


def run(sandbox: Sandbox, command: str) -> dict[str, object]:
    return sandbox.environment.execute({"command": command})


@pytest.fixture
def sandbox(tmp_path: Path) -> Iterator[Sandbox]:
    with Sandbox(build_workspace(TASK), ENVIRONMENT, root=tmp_path) as box:
        yield box


def test_the_workspace_is_mounted_and_writable(sandbox: Sandbox) -> None:
    listing = run(sandbox, "pwd && ls")
    assert listing["output"] == "/workspace\nsolution.py\ntest_public.py\n"
    assert run(sandbox, "python test_public.py")["returncode"] == 1  # the empty body fails
    write = "printf 'def double(x):\\n    return 2 * x\\n' > solution.py"
    assert run(sandbox, write)["returncode"] == 0
    assert run(sandbox, "python test_public.py")["returncode"] == 0
    assert sandbox.solution() == TASK.reference  # what the agent wrote, read on the host


def test_no_network(sandbox: Sandbox) -> None:
    probe = (
        "python -c \"import urllib.request; urllib.request.urlopen('http://pypi.org', timeout=3)\""
    )
    assert run(sandbox, probe)["returncode"] != 0


def test_a_command_is_stopped_inside_the_container(sandbox: Sandbox) -> None:
    start = time.perf_counter()
    assert run(sandbox, "sleep 30")["returncode"] == 124  # killed by `timeout` after 10 s
    assert time.perf_counter() - start < 15
    sleeping = run(sandbox, "cat /proc/[0-9]*/comm | grep -cx sleep")
    assert sleeping["output"] == "1\n"  # only PID 1 (the container's own `sleep`)


def test_the_submit_command_ends_the_episode(sandbox: Sandbox) -> None:
    with pytest.raises(Submitted):
        run(sandbox, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")


def test_closing_removes_the_container_and_the_directory(tmp_path: Path) -> None:
    box = Sandbox(build_workspace(TASK), ENVIRONMENT, root=tmp_path)
    container = box.environment.container_id
    assert container is not None
    with box:
        pass
    assert not box.directory.exists()
    found = subprocess.run(["docker", "ps", "-aq", "--filter", f"id={container}"],
                           capture_output=True, text=True, check=False)  # fmt: skip
    assert found.stdout.strip() == ""
