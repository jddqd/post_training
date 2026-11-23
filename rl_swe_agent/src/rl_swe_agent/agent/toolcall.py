"""Qwen3.5's XML tool calls, from the raw generated text to mini-swe-agent actions.

With an API, the server extracts the tool calls; with local MLX generation we get the text:

    I will write the function.
    <tool_call>
    <function=bash>
    <parameter=command>
    python test_public.py
    </parameter>
    </function>
    </tool_call>

Each call is handed to mini-swe-agent's own `parse_toolcall_actions` (unknown tool, missing
`command`: its errors). On top of it: a response cut inside a call (max_tokens), malformed tags,
and more than one call are format errors too, as DefaultAgent expects: a `FormatError` whose
message, rendered with the config's `format_error_template`, is shown to the model.
Text before the call (reasoning) and after it is allowed. The raw text is never rewritten.
"""

import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall import parse_toolcall_actions

OPEN, CLOSE = "<tool_call>", "</tool_call>"
CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FUNCTION = re.compile(r"\s*<function=([^>\n]+)>(.*)</function>\s*", re.DOTALL)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)

TRUNCATED = (
    "The response was cut off inside the tool call (too many tokens). "
    "Write shorter commands, for example a smaller file or one change at a time."
)
MALFORMED = (
    "Malformed tool call. The format is <tool_call>, <function=bash>, "
    "<parameter=command>, the command, </parameter>, </function>, </tool_call>, one per line."
)
SEVERAL = "{n} tool calls found in the response. Call the `bash` tool exactly once per response."


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, str]


def find_tool_calls(text: str) -> list[ToolCall]:
    """Every complete call in the text. ValueError on a malformed one."""
    calls = []
    for body in CALL.findall(text):
        function = FUNCTION.fullmatch(body)
        if function is None or PARAMETER.sub("", function.group(2)).strip():
            raise ValueError(body)
        arguments = {
            key.strip(): strip_newline(value) for key, value in PARAMETER.findall(function.group(2))
        }
        calls.append(ToolCall(function.group(1).strip(), arguments))
    return calls


def strip_newline(value: str) -> str:
    """The template writes each value between newlines; only those two are removed."""
    value = value.removeprefix("\n")
    return value.removesuffix("\n")


def truncated(text: str) -> bool:
    """A call was opened and never closed: generation stopped inside it."""
    return text.rfind(OPEN) > text.rfind(CLOSE)


def format_error(template: str, error: str) -> FormatError:
    """The same message as mini-swe-agent's own format errors."""
    content = Template(template, undefined=StrictUndefined).render(
        error=error, actions=[], has_tool_calls=True
    )
    return FormatError(
        {"role": "user", "content": content, "extra": {"interrupt_type": "FormatError"}}
    )


def parse_actions(
    text: str, *, format_error_template: str, call_id: str = "call_0"
) -> list[dict[str, Any]]:
    """The single bash action of a response: [{"command", "tool_call_id"}]. Else FormatError."""
    if truncated(text):
        raise format_error(format_error_template, TRUNCATED)
    try:
        calls = find_tool_calls(text)
    except ValueError:
        raise format_error(format_error_template, MALFORMED) from None
    if len(calls) > 1:
        raise format_error(format_error_template, SEVERAL.format(n=len(calls)))
    # mini-swe-agent's checks (no call, unknown tool, missing command), on API-shaped calls.
    api_calls = [
        SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=call.name, arguments=json.dumps(call.arguments)),
        )
        for call in calls
    ]
    actions: list[dict[str, Any]] = parse_toolcall_actions(
        api_calls, format_error_template=format_error_template
    )
    return actions
