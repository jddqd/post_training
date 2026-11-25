"""One episode: a task, a model, mini-swe-agent's DefaultAgent in a fresh sandbox, then the reward.

    episode = run_episode(task, model, mini_config(config))
    episode.reward, episode.exit_status, episode.messages

DefaultAgent drives the loop (query the model, run the command, add the observation) and ends it:
Submitted (the submit command), LimitsExceeded (step_limit), RepeatedFormatError,
ContextLimitExceeded (the MLX model, at max_context_tokens). The solution
is then checked by the hidden tests in a separate container (sandbox/verifier.py).

The reward is given only to a submitted solution: an episode stopped by the step limit gets 0,
even when its solution passes (its verdict is kept for analysis). Knowing when to stop is part
of the task.
"""

import time
from dataclasses import asdict, dataclass
from typing import Any

from minisweagent.agents.default import DefaultAgent

from rl_swe_agent.agent.model import TextModel
from rl_swe_agent.sandbox.sandbox import Sandbox
from rl_swe_agent.sandbox.verifier import Verdict, verify
from rl_swe_agent.tasks.opencodeinstruct import CodeTask
from rl_swe_agent.tasks.workspace import build_workspace

SUBMITTED = "Submitted"


def task_variables(task: CodeTask) -> dict[str, str]:
    """What agent.run() passes to the templates (`task` is mini-swe-agent's own name)."""
    return {"task": task.problem.strip(), "signature": task.signature, "function": task.function}


@dataclass
class Episode:
    task_id: str
    messages: list[dict[str, Any]]  # DefaultAgent's: system, task, then responses and observations
    exit_status: str  # Submitted, LimitsExceeded, RepeatedFormatError
    verdict: Verdict  # the hidden tests on the final solution, submitted or not
    model_calls: int
    seconds: float
    ids: list[int]  # the episode as tokens (empty for a model that only returns text)
    mask: list[int]
    logprobs: list[float]

    @property
    def submitted(self) -> bool:
        return self.exit_status == SUBMITTED

    @property
    def reward(self) -> float:
        return self.verdict.reward if self.submitted else 0.0

    @property
    def format_errors(self) -> int:
        return sum(m.get("extra", {}).get("interrupt_type") == "FormatError" for m in self.messages)

    def to_json(self) -> dict[str, Any]:
        """One line of episodes.jsonl: the fields, and the derived values for reading."""
        return {**asdict(self), "reward": self.reward, "passed": self.verdict.passed}

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> "Episode":
        fields = {k: v for k, v in record.items() if k not in ("reward", "passed")}
        return cls(**{**fields, "verdict": Verdict(**fields["verdict"])})


def run_episode(task: CodeTask, model: TextModel, mini: dict[str, Any]) -> Episode:
    """`mini`: rl_swe_agent.config.mini_config(config)."""
    start = time.time()
    with Sandbox(build_workspace(task), mini["environment"]) as sandbox:
        agent = DefaultAgent(model, sandbox.environment, **mini["agent"])
        info = agent.run(**task_variables(task), **mini["template_vars"])
        solution = sandbox.solution()
    return Episode(
        task_id=task.id,
        messages=agent.messages,
        exit_status=info["exit_status"],
        verdict=verify(solution, task.tests, mini["environment"]),
        model_calls=agent.n_calls,
        seconds=time.time() - start,
        ids=model.ids,
        mask=model.mask,
        logprobs=model.logprobs,
    )
