"""The mini-swe-agent configuration of the code tasks (agent.yaml) and its task variables."""

from pathlib import Path
from typing import Any

import yaml

from rl_swe_agent.tasks.opencodeinstruct import CodeTask

CONFIG = Path(__file__).with_name("agent.yaml")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    """agent.yaml as is (agent, environment, model): use rl_swe_agent.config.mini_config to run."""
    config: dict[str, Any] = yaml.safe_load(path.read_text())
    return config


def task_variables(task: CodeTask) -> dict[str, str]:
    """What agent.run() passes to the templates (`task` is mini-swe-agent's own name)."""
    return {"task": task.problem.strip(), "signature": task.signature, "function": task.function}
