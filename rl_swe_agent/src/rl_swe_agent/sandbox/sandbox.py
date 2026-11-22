"""One rollout's sandbox: the task workspace mounted in its own Docker container.

    with Sandbox(workspace, config["environment"]) as sandbox:
        sandbox.environment.execute({"command": "python test_public.py"})
        code = sandbox.solution()

The workspace is written to a fresh host directory, mounted at /workspace: what the agent writes
there is read back from the host at the end (for the verifier). Commands run through
mini-swe-agent's DockerEnvironment (`docker exec`), which DefaultAgent drives. Closing removes
the container and the directory, also when the rollout fails, and returns once they are gone
(`docker rm -f`). mini-swe-agent's own cleanup runs `timeout 60 docker stop || docker rm -f` in
the background: on Linux `docker stop` waits its 10 s grace period (`sleep`, as PID 1, ignores
SIGTERM), on macOS `timeout` is missing so it falls through to `rm -f`.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Any

from minisweagent.environments.docker import DockerEnvironment

from rl_swe_agent.tasks.workspace import SOLUTION, Workspace


class Sandbox:
    def __init__(
        self, workspace: Workspace, environment: dict[str, Any], root: Path | None = None
    ) -> None:
        self.directory = Path(
            tempfile.mkdtemp(prefix=f"rollout-{workspace.task.id[:8]}-", dir=root)
        )
        try:
            workspace.write(self.directory)
            config = dict(environment)
            config["run_args"] = [
                *config.get("run_args", []), "-v", f"{self.directory}:{config['cwd']}"
            ]  # fmt: skip
            self.environment = DockerEnvironment(**config)
        except BaseException:
            shutil.rmtree(self.directory, ignore_errors=True)
            raise

    def solution(self) -> str:
        """The agent's solution.py, read on the host."""
        return (self.directory / SOLUTION).read_text()

    def close(self) -> None:
        container, self.environment.container_id = self.environment.container_id, None
        if container is not None:  # None: its own cleanup (and __del__) has nothing left to do
            subprocess.run(
                [self.environment.config.executable, "rm", "-f", container],
                capture_output=True,
                check=False,
            )
        shutil.rmtree(self.directory, ignore_errors=True)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
