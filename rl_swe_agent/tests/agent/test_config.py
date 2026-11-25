from jinja2 import StrictUndefined, Template
from minisweagent.agents.default import AgentConfig
from minisweagent.environments.docker import DockerEnvironmentConfig

from rl_swe_agent.agent.config import load_config, task_variables
from rl_swe_agent.config import compose, mini_config
from rl_swe_agent.tasks.opencodeinstruct import CodeTask

TASK = CodeTask(
    id="t",
    problem="  Return the sum of a list.\n",
    reference="def total(xs):\n    return sum(xs)",
    tests=["assert total([1]) == 1"] * 10,
    function="total",
    signature="def total(xs):",
    score=1.0,
    domain="generic",
)


def render(template: str, **variables: object) -> str:
    # As DefaultAgent does: a missing variable is an error, not an empty string.
    return str(Template(template, undefined=StrictUndefined).render(**variables))


def test_agent_yaml_holds_no_hyperparameter() -> None:
    config = load_config()
    assert set(config) == {"agent", "environment", "model"}
    assert "step_limit" not in config["agent"]
    assert config["environment"]["interpreter"] == ["bash", "-lc"]


def test_with_the_hyperparameters_it_is_a_valid_mini_swe_agent_config() -> None:
    mini = mini_config(compose(["rollout.step_limit=7", "rollout.command_timeout=3"]))
    agent = AgentConfig(**mini["agent"])
    assert agent.step_limit == 7
    environment = DockerEnvironmentConfig(**mini["environment"])
    assert environment.interpreter == ["timeout", "-k", "2", "3", "bash", "-lc"]
    assert environment.timeout == 13
    assert environment.cwd == "/workspace"
    assert "--network" in environment.run_args
    assert mini["template_vars"] == {"command_timeout": 3}


def test_the_task_message_names_the_function_and_the_submit_command() -> None:
    config = load_config()
    message = render(config["agent"]["instance_template"], step_limit=15, **task_variables(TASK))
    assert "<problem>\nReturn the sum of a list.\n</problem>" in message
    assert "def total(xs):" in message
    assert "from solution import total; print(total(<edge-case input>))" in message
    assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in message
    assert "You have 15 responses in total" in message


def test_long_outputs_are_cut_in_the_middle() -> None:
    template = load_config()["model"]["observation_template"]
    short = render(template, output={"returncode": 1, "output": "boom\n", "exception_info": ""})
    assert short == "<returncode>1</returncode>\n<output>\nboom\n</output>"
    long = render(template, output={"returncode": 0, "output": "a" * 5000, "exception_info": ""})
    assert "<elided_chars>2000</elided_chars>" in long
    assert len(long) < 3200


def test_format_errors_explain_the_tool() -> None:
    template = load_config()["model"]["format_error_template"]
    message = render(template, error="No tool calls found in the response.")
    assert message.startswith("Format error: No tool calls found in the response.")
