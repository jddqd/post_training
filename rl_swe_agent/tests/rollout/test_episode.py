"""Scripted models through the real loop: DefaultAgent, the Docker sandbox and the verifier."""

import subprocess

import pytest

from rl_swe_agent.agent.config import load_config
from rl_swe_agent.agent.model import ScriptedModel
from rl_swe_agent.rollout.episode import Episode, run_episode
from rl_swe_agent.sandbox.verifier import docker_ready
from rl_swe_agent.tasks.opencodeinstruct import CodeTask

pytestmark = pytest.mark.skipif(not docker_ready(), reason="needs Docker and python:3.12-slim")

CONFIG = load_config()
TASK = CodeTask(
    id="episode-test",
    problem="Double a number.",
    reference="def double(x):\n    return 2 * x\n",
    tests=[f"assert double({x}) == {2 * x}" for x in (1, 0, -1, 3, 10, -7, 100, 5, 2, 8)],
    function="double",
    signature="def double(x):",
    score=1.0,
    domain="generic",
)
WRITE = "cat <<'EOF' > solution.py\ndef double(x):\n    return 2 * x\nEOF"
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def call(command: str) -> str:
    return (
        f"<tool_call>\n<function=bash>\n<parameter=command>\n{command}\n"
        "</parameter>\n</function>\n</tool_call>"
    )


def episode(*responses: str) -> Episode:
    return run_episode(TASK, ScriptedModel(CONFIG["model"], list(responses)), CONFIG)


def roles(episode: Episode) -> list[str]:
    return [message["role"] for message in episode.messages]


def test_oracle_writes_tests_and_submits() -> None:
    result = episode(call(WRITE), call("python test_public.py"), call(SUBMIT))
    assert result.exit_status == "Submitted"
    assert result.reward == 1.0
    assert result.model_calls == 3
    assert roles(result) == ["system", "user"] + ["assistant", "tool"] * 2 + ["assistant", "exit"]
    assert "OK: all public tests passed" in result.messages[5]["content"]


def test_submitting_the_stub_gets_nothing() -> None:
    result = episode(call(SUBMIT))
    assert result.submitted
    assert result.verdict.passed == 0
    assert result.reward == 0.0


def test_rewriting_the_public_tests_does_not_fool_the_verifier() -> None:
    cheat = "echo 'print(\"OK: all public tests passed\")' > test_public.py"
    result = episode(call(cheat), call("python test_public.py"), call(SUBMIT))
    assert "OK: all public tests passed" in result.messages[5]["content"]
    assert result.reward == 0.0


def test_three_format_errors_in_a_row_end_the_episode() -> None:
    result = episode("Let me think.", "Still thinking.", "Done, I think.")
    assert result.exit_status == "RepeatedFormatError"
    assert result.reward == 0.0
    # Each faulty response stays in the conversation, followed by its error.
    assert roles(result) == ["system", "user"] + ["assistant", "user"] * 3 + ["exit"]
    assert result.messages[3]["content"].startswith("Format error: No tool calls found")


def test_a_format_error_can_be_recovered() -> None:
    result = episode("Let me think.", call(WRITE), call(SUBMIT))
    assert result.exit_status == "Submitted"
    assert result.reward == 1.0


def test_a_hanging_command_is_stopped_and_the_episode_goes_on() -> None:
    result = episode(call('python -c "while True: pass"'), call(WRITE), call(SUBMIT))
    assert (
        "<warning>The command was stopped after 10 seconds.</warning>"
        in result.messages[3]["content"]
    )
    assert result.reward == 1.0
    assert result.seconds < 20


def test_not_submitting_gets_nothing_even_with_a_correct_solution() -> None:
    step_limit = CONFIG["agent"]["step_limit"]
    result = episode(call(WRITE), *[call("ls")] * step_limit)
    assert result.exit_status == "LimitsExceeded"
    assert result.model_calls == step_limit
    assert result.verdict.passed == 10
    assert result.reward == 0.0


def containers() -> set[str]:
    listed = subprocess.run(["docker", "ps", "-aq"], capture_output=True, text=True, check=True)
    return set(listed.stdout.split())


def test_no_container_is_left_behind() -> None:
    before = containers()  # others may run meanwhile: only this episode's must be gone
    episode(call(SUBMIT))
    assert containers() <= before
