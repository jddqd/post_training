"""One episode: a task, a model, mini-swe-agent's DefaultAgent in a fresh sandbox, then the reward.

    episode = run_episode(task, model, load_config())
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
from dataclasses import dataclass
from typing import Any

from minisweagent.agents.default import DefaultAgent

from rl_swe_agent.agent.config import task_variables
from rl_swe_agent.agent.model import TextModel
from rl_swe_agent.sandbox.sandbox import Sandbox
from rl_swe_agent.sandbox.verifier import Verdict, verify
from rl_swe_agent.tasks.opencodeinstruct import CodeTask
from rl_swe_agent.tasks.workspace import build_workspace

SUBMITTED = "Submitted"


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


def run_episode(task: CodeTask, model: TextModel, config: dict[str, Any]) -> Episode:
    start = time.time()
    with Sandbox(build_workspace(task), config["environment"]) as sandbox:
        agent = DefaultAgent(model, sandbox.environment, **config["agent"])
        info = agent.run(**task_variables(task))
        solution = sandbox.solution()
    return Episode(
        task_id=task.id,
        messages=agent.messages,
        exit_status=info["exit_status"],
        verdict=verify(solution, task.tests),
        model_calls=agent.n_calls,
        seconds=time.time() - start,
        ids=model.ids,
        mask=model.mask,
        logprobs=model.logprobs,
    )
