import pytest
from minisweagent.exceptions import FormatError

from rl_swe_agent.agent.config import load_config
from rl_swe_agent.agent.model import ScriptedModel

CONFIG = load_config()["model"]


def call(command: str) -> str:
    return (
        f"<tool_call>\n<function=bash>\n<parameter=command>\n{command}\n"
        "</parameter>\n</function>\n</tool_call>"
    )


def test_a_response_carries_its_action_and_raw_text() -> None:
    model = ScriptedModel(CONFIG, ["Listing.\n" + call("ls"), call("pwd")])
    first = model.query([])
    assert first["role"] == "assistant"
    assert first["content"] == "Listing.\n" + call("ls")
    assert first["extra"]["actions"] == [{"command": "ls", "tool_call_id": "call_0"}]
    assert model.query([])["extra"]["actions"][0]["tool_call_id"] == "call_1"


def test_a_format_error_keeps_the_response_before_the_error() -> None:
    model = ScriptedModel(CONFIG, ["The answer is 42."])
    with pytest.raises(FormatError) as raised:
        model.query([])
    response, error = raised.value.messages
    assert response == {
        "role": "assistant",
        "content": "The answer is 42.",
        "extra": response["extra"],
    }
    assert "actions" not in response["extra"]
    assert error["role"] == "user"
    assert error["content"].startswith("Format error: No tool calls found")


def test_observations_are_tool_messages_in_the_template() -> None:
    model = ScriptedModel(CONFIG, [call("echo hi")])
    message = model.query([])
    output = {"returncode": 0, "output": "hi\n", "exception_info": ""}
    (observation,) = model.format_observation_messages(message, [output])
    assert observation["role"] == "tool"
    assert observation["tool_call_id"] == "call_0"
    assert observation["content"] == "<returncode>0</returncode>\n<output>\nhi\n</output>"
