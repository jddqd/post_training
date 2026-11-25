"""Run a solution against its asserts in a throwaway container: the reward of a code task.

Model-written code never runs on the host: one `docker run` per check, with the agent's sandbox
image and options (agent.yaml: no network, limited memory and CPU), the code mounted read-only.
Each assert runs on its own (a failing one does not hide the others) and the container prints
one JSON status per test.
"""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_swe_agent.config import load_agent_yaml

# Runs inside the container: load the solution, then each test in its namespace.
RUNNER = """
import json, signal, sys
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
namespace = {"__name__": "solution"}
statuses = []
try:
    exec(open("/task/solution.py").read(), namespace)
except BaseException as error:
    print(json.dumps({"load_error": repr(error)[:300]}))
    sys.exit()
for test in json.load(open("/task/tests.json")):
    signal.alarm(PER_TEST_TIMEOUT)
    try:
        exec(test, dict(namespace))
        statuses.append("pass")
    except AssertionError:
        statuses.append("fail")
    except TimeoutError:
        statuses.append("timeout")
    except BaseException as error:
        statuses.append("error: " + type(error).__name__)
    finally:
        signal.alarm(0)
print(json.dumps({"statuses": statuses}))
"""


def sandbox_environment() -> dict[str, Any]:
    """The agent's sandbox options (agent.yaml), shared by the verifier."""
    environment: dict[str, Any] = load_agent_yaml()["environment"]
    return environment


def docker_ready(image: str | None = None) -> bool:
    """Docker reachable and the image present (it is never pulled during a run)."""
    image = image or sandbox_environment()["image"]
    if shutil.which("docker") is None:
        return False
    found = subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=False)
    return found.returncode == 0


@dataclass(frozen=True)
class Verdict:
    statuses: list[str]  # one per test: pass, fail, timeout, error: <type>
    load_error: str = ""  # the solution itself does not run (syntax error, bad import...)

    @property
    def passed(self) -> int:
        return self.statuses.count("pass")

    @property
    def reward(self) -> float:
        """1 if every test passes, else 0 (the fraction stays available for analysis)."""
        return float(bool(self.statuses) and self.passed == len(self.statuses))


def verify(
    code: str,
    tests: list[str],
    environment: dict[str, Any] | None = None,
    per_test_timeout: int = 2,
    total_timeout: int = 30,
) -> Verdict:
    """`environment`: the sandbox options (default: agent.yaml's), for the image and limits.

    A test of these tasks runs in milliseconds: one taking 2 s is an endless loop or a far too
    slow solution, so it fails. The total (10 tests x 2 s, plus the start) bounds the worst case:
    with 5 s per test, one solution that hangs held the end of a rollout step for 60 s.
    """
    environment = environment or sandbox_environment()
    with tempfile.TemporaryDirectory() as tmp:
        task = Path(tmp)
        (task / "solution.py").write_text(code)
        (task / "tests.json").write_text(json.dumps(tests))
        (task / "runner.py").write_text(RUNNER.replace("PER_TEST_TIMEOUT", str(per_test_timeout)))
        command = [
            "docker", "run", *environment["run_args"], "-v", f"{task}:/task:ro",
            environment["image"],
            # killed inside the container: a killed `docker run` client would leave it running
            "timeout", str(total_timeout), "python", "-I", "/task/runner.py",
        ]  # fmt: skip
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=total_timeout + 30, check=False
            )
        except subprocess.TimeoutExpired:
            return Verdict(statuses=["timeout"] * len(tests))
    lines = result.stdout.strip().splitlines()
    if result.returncode == 124:  # `timeout` killed the runner
        return Verdict(statuses=["timeout"] * len(tests))
    if not lines:
        return Verdict(statuses=[], load_error=f"no output: {result.stderr.strip()[-300:]}")
    report = json.loads(lines[-1])
    if "load_error" in report:
        return Verdict(statuses=["error: load"] * len(tests), load_error=report["load_error"])
    return Verdict(statuses=report["statuses"])
