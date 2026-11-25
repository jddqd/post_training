import pytest
from minisweagent.exceptions import FormatError

from rl_swe_agent.agent.toolcall import ToolCall, find_tool_calls, parse_actions
from rl_swe_agent.agent.toolcall import write_tool_call as call
from rl_swe_agent.config import load_agent_yaml

TEMPLATE = load_agent_yaml()["model"]["format_error_template"]


def error(text: str) -> str:
    with pytest.raises(FormatError) as raised:
        parse_actions(text, format_error_template=TEMPLATE)
    message = raised.value.messages[0]
    assert message["role"] == "user"
    assert message["extra"]["interrupt_type"] == "FormatError"
    content: str = message["content"]
    assert content.startswith("Format error: ")
    return content


def test_one_call_is_one_action() -> None:
    actions = parse_actions(call("ls"), format_error_template=TEMPLATE, call_id="call_3")
    assert actions == [{"command": "ls", "tool_call_id": "call_3"}]


def test_reasoning_before_and_text_after_are_allowed() -> None:
    text = f"I will run the tests first.\n\n{call('python test_public.py')}\nDone."
    assert parse_actions(text, format_error_template=TEMPLATE)[0]["command"] == (
        "python test_public.py"
    )


def test_commands_are_raw_text() -> None:
    command = (
        "cat <<'EOF' > solution.py\ndef f(a, b):\n    if a < b and b > 0:\n"
        '        return "x & y"\n\n    return None\nEOF\n'
    )
    # Only the newlines around the value, written by the template, are removed.
    assert find_tool_calls(call(command)) == [ToolCall("bash", {"command": command})]


def test_no_call() -> None:
    assert "No tool calls found" in error("I think the answer is 42.")


def test_several_calls() -> None:
    assert "2 tool calls found" in error(call("ls") + "\n" + call("pwd"))


def test_unknown_tool() -> None:
    assert "Unknown tool 'python'" in error(call("print(1)", function="python"))


def test_missing_command() -> None:
    assert "Missing 'command'" in error(call("ls", parameter="cmd"))


def test_cut_inside_the_call() -> None:
    assert "cut off" in error("Writing it.\n<tool_call>\n<function=bash>\n<parameter=command>\ncat")


def test_malformed_tags() -> None:
    assert "Malformed" in error("<tool_call>\n<function=bash>\nls\n</function>\n</tool_call>")
    unclosed = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</function>\n</tool_call>"
    assert "Malformed" in error(unclosed)
