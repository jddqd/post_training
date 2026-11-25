"""Dynamic batching of a step's episodes, with scripted models in Docker."""

import itertools
import threading

import pytest
from helpers import DOUBLE

from rl_swe_agent.agent.model import ScriptedModel, TextModel
from rl_swe_agent.agent.toolcall import write_tool_call as call
from rl_swe_agent.config import compose, mini_config
from rl_swe_agent.rollout.runner import run_episodes, step_stats
from rl_swe_agent.sandbox.verifier import docker_ready

pytestmark = pytest.mark.skipif(not docker_ready(), reason="needs Docker and the sandbox image")
SUBMIT = call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")


def test_a_slot_takes_the_next_episode_without_waiting_for_the_others() -> None:
    config = compose(["rollout.group_size=2", "rollout.max_in_flight=2"])
    templates = mini_config(config)["model"]
    created, lock = itertools.count(), threading.Lock()

    def policy() -> TextModel:
        with lock:
            n = next(created)
        slow = [call("sleep 3"), SUBMIT]  # the first episode holds its slot for 3 s
        return ScriptedModel(templates, slow if n == 0 else [SUBMIT])

    groups = run_episodes([DOUBLE, DOUBLE], policy, config)
    episodes = [e for group in groups for e in group]
    assert [len(group) for group in groups] == [2, 2]
    assert all(e.exit_status == "Submitted" for e in episodes)

    slow, *others = sorted(episodes, key=lambda e: e.seconds, reverse=True)
    assert slow.seconds > 3
    # The three quick episodes ran one after the other in the second slot, during the slow one.
    assert all(slow.started < e.started < slow.started + slow.seconds for e in others[1:])
    intervals = [(e.started, e.started + e.seconds) for e in episodes]
    for start, _ in intervals:
        running = sum(s <= start < end for s, end in intervals)
        assert running <= 2

    timeline = step_stats(episodes, slow.seconds, max_in_flight=2)
    assert 0 < timeline["occupancy"] <= 1
