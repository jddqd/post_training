"""Convert Team-ACE/ToolACE rows to normalized conversations.

A raw row has `system` (str) and `conversations` (a list of {"from", "value"} turns, with
"from" in user / assistant / tool). The tools are a JSON list inside the system prompt, after
`TOOLS_MARKER`, already close to BFCL's function-doc format (`"type": "dict"`). Assistant tool
calls are Python-like text:

    [Market Trends API(trend_type="MARKET_INDEXES", country="us"), Other Tool(n=3)]

What we keep and why (see xlam.py for the target format):
- The system prompt is dropped: the tools go through the chat template, exactly as BFCL renders
  them (ToolACE's prompt and call syntax are not what the model is evaluated on).
- Tool calls become structured `tool_calls`, rendered by the chat template as <tool_call> JSON.
- Multi-turn conversations, tool responses and text answers are kept: they teach what xLAM
  lacks (answering from a tool result, saying that no function applies, asking for a missing
  parameter).
- Rows whose system prompt uses another tool format (ToolACE's format augmentation: other key
  names, custom call syntaxes, some in Chinese) are rejected: they would teach call formats the
  model never has to produce.
"""

import ast
import json
from typing import Any

from post_training.data.conversation import Conversation
from post_training.data.xlam import BFCL_PYTHON_HINT, JsonSchema, RejectedError, validate_calls

TOOLS_MARKER = "Here is a list of functions in JSON format that you can invoke:\n"

# ToolACE type names -> BFCL's vocabulary (JSON-Schema-like with Python names).
_TYPES = {
    "string": "string",
    "str": "string",
    "integer": "integer",
    "int": "integer",
    "float": "float",
    "number": "float",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "dict",
    "object": "dict",
    "tuple": "tuple",
    "any": "any",
}

_ROLES = {"user": "user", "assistant": "assistant", "tool": "tool"}


def normalize_type(raw: Any) -> str:
    """`"int"` -> `"integer"`; a nullable union `["integer", "null"]` -> `"integer"`."""
    if isinstance(raw, list):
        non_null = [t for t in raw if t != "null"]
        if len(non_null) != 1:
            raise RejectedError("union_type", json.dumps(raw))
        raw = non_null[0]
    if not isinstance(raw, str) or raw not in _TYPES:
        raise RejectedError("unsupported_type", json.dumps(raw))
    return _TYPES[raw]


def normalize_schema(schema: JsonSchema) -> JsonSchema:
    """A parameter schema with BFCL type names, recursively (array items, dict properties)."""
    if not isinstance(schema, dict) or "type" not in schema:
        raise RejectedError("invalid_schema", json.dumps(schema)[:200])
    result = {**schema, "type": normalize_type(schema["type"])}
    if isinstance(schema.get("items"), dict):
        result["items"] = normalize_schema(schema["items"])
    if isinstance(schema.get("properties"), dict):
        result["properties"] = {k: normalize_schema(v) for k, v in schema["properties"].items()}
    return result


def convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """ToolACE tool -> BFCL function doc (name, description + Python hint, dict parameters)."""
    parameters = tool.get("parameters") or {}
    if normalize_type(parameters.get("type", "dict")) != "dict":
        raise RejectedError("invalid_schema", f"{tool['name']}: parameters are not a dict")
    properties = {
        name: normalize_schema(schema)
        for name, schema in (parameters.get("properties") or {}).items()
    }
    required = parameters.get("required") or []
    if unknown := set(required) - set(properties):
        raise RejectedError("invalid_schema", f"{tool['name']}: required {sorted(unknown)}")
    return {
        "name": tool["name"],
        "description": (tool.get("description") or "") + BFCL_PYTHON_HINT,
        # Only the keys BFCL uses ("required": null at the top level, "optional" are dropped).
        "parameters": {"type": "dict", "properties": properties, "required": required},
    }


def parse_tools(system: str) -> list[dict[str, Any]]:
    if TOOLS_MARKER not in system:
        raise RejectedError("nonstandard_format", system[:100])
    try:
        raw_tools, _ = json.JSONDecoder().raw_decode(system.split(TOOLS_MARKER, 1)[1])
    except json.JSONDecodeError as error:
        raise RejectedError("invalid_json", str(error)) from error
    tools = [convert_tool(tool) for tool in raw_tools]
    names = [tool["name"] for tool in tools]
    if len(set(names)) != len(names):
        # Two functions with one name: the expected call would be ambiguous.
        raise RejectedError("duplicate_tool_name", ", ".join(sorted(names)))
    return tools


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on `sep` outside brackets and string literals (Python quoting, with escapes)."""
    parts, depth, quote, current, i = [], 0, "", "", 0
    while i < len(text):
        char = text[i]
        if quote:
            if char == "\\" and i + 1 < len(text):
                current += char
                i += 1
                char = text[i]
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise RejectedError("invalid_call", f"unbalanced brackets: {text[:100]}")
        if char == sep and depth == 0 and not quote:
            parts.append(current)
            current = ""
        else:
            current += char
        i += 1
    if quote or depth:
        raise RejectedError("invalid_call", f"unbalanced brackets or quotes: {text[:100]}")
    parts.append(current)
    return parts


def _literal(node: ast.expr) -> Any:
    """A JSON value from a Python literal (also accepts JSON's true / false / null)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[node.id]
    if isinstance(node, ast.List | ast.Tuple):
        return [_literal(element) for element in node.elts]
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise RejectedError("invalid_call", "dict unpacking")
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values, strict=True)}  # type: ignore[arg-type]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        value = _literal(node.operand)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return -value if isinstance(node.op, ast.USub) else value
    raise RejectedError("invalid_call", f"not a literal: {ast.dump(node)[:100]}")


def _parse_arguments(source: str) -> dict[str, Any]:
    """`a=1, from="x"` -> {"a": 1, "from": "x"}: keyword arguments with literal values.

    Split by hand rather than parsed as a Python call: argument names can be Python keywords
    (`from`, `in`, `class`), which `ast` rejects.
    """
    arguments: dict[str, Any] = {}
    if not source.strip():
        return arguments
    for part in _split_top_level(source, ","):
        name, sep, value = part.partition("=")
        name = name.strip()
        # Any name is accepted here ("$top", "Battle ID" exist): validate_calls then checks it
        # against the tool's parameters.
        if not sep or not name or any(quote in name for quote in "\"'"):
            raise RejectedError("invalid_call", f"not a keyword argument: {part.strip()[:100]}")
        if name in arguments:
            raise RejectedError("invalid_call", f"repeated argument {name!r}")
        try:
            arguments[name] = _literal(ast.parse(value.strip(), mode="eval").body)
        except SyntaxError as error:
            raise RejectedError("invalid_call", f"{error.msg}: {value.strip()[:100]}") from error
    return arguments


def parse_calls(text: str, names: list[str]) -> list[dict[str, Any]]:
    """`[Name A(x=1), Name B(y="s")]` -> [{"name": "Name A", "arguments": {"x": 1}}, ...].

    Function names may contain spaces, quotes or even parentheses (not valid Python), so each
    call must start with one of the known tool `names` (the longest that matches) followed by
    "(", then the arguments up to the matching ")".
    """
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise RejectedError("invalid_call", text[:100])
    by_length = sorted(names, key=len, reverse=True)  # "Get User Info" before "Get User"
    calls = [
        {"name": name, "arguments": _parse_arguments(arguments)}
        for name, arguments in _split_call_list(text[1:-1], by_length)
    ]
    if not calls:
        raise RejectedError("invalid_call", "empty call list")
    return calls


def _closing_paren(text: str, start: int) -> int:
    """Index of the ")" matching the "(" at `start`, ignoring brackets in string literals."""
    depth, quote, i = 0, "", start
    while i < len(text):
        char = text[i]
        if quote:
            if char == "\\":
                i += 1  # skip the escaped character
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise RejectedError("invalid_call", f"unbalanced parentheses: {text[start:][:100]}")


def _split_call_list(body: str, names: list[str]) -> list[tuple[str, str]]:
    """`A(x=1), B(y=2)` -> [("A", "x=1"), ("B", "y=2")]. The name is matched first, so that a
    parenthesis inside a function name is not taken for the start of the arguments."""
    calls: list[tuple[str, str]] = []
    i = 0
    while True:
        while i < len(body) and body[i].isspace():
            i += 1
        if i == len(body):
            return calls
        name = next((n for n in names if body.startswith(n + "(", i)), None)
        if name is None:
            raise RejectedError("unknown_function", body[i:][:100])
        open_paren = i + len(name)
        close = _closing_paren(body, open_paren)
        calls.append((name, body[open_paren + 1 : close]))
        i = close + 1
        while i < len(body) and body[i].isspace():
            i += 1
        if i < len(body):
            if body[i] != ",":
                raise RejectedError("invalid_call", f"expected ',' in {body[close:][:100]!r}")
            i += 1


def is_call(value: str) -> bool:
    """Assistant turns starting with "[" are calls (text answers never do in ToolACE)."""
    return value.lstrip().startswith("[")


def convert_row(row: dict[str, Any]) -> Conversation:
    """One ToolACE row -> a (possibly multi-turn) conversation with structured tool calls."""
    tools = parse_tools(row["system"])
    messages: list[dict[str, Any]] = []
    for turn in row["conversations"]:
        role = _ROLES.get(turn["from"])
        if role is None:
            raise RejectedError("invalid_role", turn["from"])
        value = turn["value"]
        if role == "assistant" and is_call(value):
            calls = parse_calls(value, [tool["name"] for tool in tools])
            validate_calls(calls, tools)
            tool_calls = [{"type": "function", "function": call} for call in calls]
            messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
            continue
        if role == "tool" and not (
            messages and (messages[-1].get("tool_calls") or messages[-1]["role"] == "tool")
        ):
            raise RejectedError("invalid_turn_order", "tool response without a call")
        messages.append({"role": role, "content": value})
    if not messages or messages[0]["role"] != "user":
        raise RejectedError("invalid_turn_order", "does not start with a user message")
    if messages[-1]["role"] != "assistant":
        # A trailing user or tool message would never be trained: the row ends unanswered.
        raise RejectedError("invalid_turn_order", f"ends with a {messages[-1]['role']} message")
    return Conversation(messages=messages, tools=tools)
