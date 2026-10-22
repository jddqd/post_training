"""Convert Salesforce/xlam-function-calling-60k rows to normalized conversations.

A raw row has `query` (str) plus `tools` and `answers` (JSON strings). Tool parameters use
Python type strings (`"str, optional"`, `"List[int]"`, ...). We convert tools to the exact
function-doc format BFCL puts in the prompt, so the model is trained on what it is evaluated on:

    {"name": ..., "description": "... Note that the provided function is in Python 3 syntax.",
     "parameters": {"type": "dict", "properties": {...}, "required": [...]}}

BFCL's schema is JSON-Schema-like but with Python type names: `dict` (not `object`), `float`
(not `number`), `tuple`, `any`. Tools are not wrapped in `{"type": "function", ...}`.
"""

import json
from typing import Any

from post_training.data.conversation import Conversation

JsonSchema = dict[str, Any]

# Appended by BFCL to every function description of the Python and live categories.
BFCL_PYTHON_HINT = " Note that the provided function is in Python 3 syntax."


class RejectedError(ValueError):
    """A row we cannot convert faithfully. `reason` is a short, stable label for the funnel."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# Every base type found in xLAM except `Callable[[float], float]` (593 rows), which has no
# equivalent in BFCL's schema vocabulary: rows using it are rejected.
_SIMPLE_TYPES: dict[str, JsonSchema] = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "float"},
    "bool": {"type": "boolean"},
    "list": {"type": "array"},
    "List": {"type": "array"},
    "Dict": {"type": "dict"},
    "set": {"type": "array"},  # BFCL has no set type
}


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on `sep` outside brackets.

    'Tuple[int, int], optional' -> ['Tuple[int, int]', 'optional']
    """
    parts: list[str] = []
    depth, current = 0, ""
    for char in text:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        if char == sep and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    parts.append(current.strip())
    return parts


def type_to_schema(type_name: str) -> JsonSchema:
    """Python type name -> BFCL schema, recursively for generics ('List[List[int]]')."""
    type_name = type_name.strip()
    if type_name in _SIMPLE_TYPES:
        return dict(_SIMPLE_TYPES[type_name])  # copy: callers add "description"/"default"

    generic, _, rest = type_name.partition("[")
    if not rest.endswith("]"):
        raise RejectedError("unsupported_type", type_name)
    args = [type_to_schema(arg) for arg in split_top_level(rest[:-1])]

    match generic:
        case "List" if len(args) == 1:
            return {"type": "array", "items": args[0]}
        case "Tuple" if all(arg == args[0] for arg in args):
            # Homogeneous tuple, e.g. a (lat, lon) pair: {"type": "tuple", "items": ...} in BFCL.
            return {"type": "tuple", "items": args[0]}
        case "Union" if sorted(arg["type"] for arg in args) == ["float", "integer"]:
            return {"type": "float"}  # an int is a valid float argument
        case _:
            raise RejectedError("unsupported_type", type_name)


def parse_type(type_string: str) -> tuple[JsonSchema, bool]:
    """'List[int], optional' -> ({'type': 'array', 'items': {'type': 'integer'}}, True).

    xLAM encodes optionality in the type string: 'str, optional', "str, default='London'".
    """
    base, *modifiers = split_top_level(type_string)
    optional = any("optional" in m or "default" in m for m in modifiers)
    return type_to_schema(base), optional


_PYTHON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "float": (int, float),
    "boolean": bool,
    "array": list,
    "tuple": list,  # JSON has no tuples: a tuple value is a JSON array
    "dict": dict,
}


def _matches(schema: JsonSchema, value: Any) -> bool:
    expected = _PYTHON_TYPES.get(schema.get("type", ""))
    if expected is None:  # e.g. "any": don't judge
        return True
    if isinstance(value, bool) and schema["type"] in ("integer", "float"):
        return False  # bool is a subclass of int in Python, but not a number for BFCL
    return isinstance(value, expected)


def coerce_default(schema: JsonSchema, value: Any) -> tuple[bool, Any]:
    """Return (keep, value): a default must match its declared type.

    xLAM has ~49k defaults of the wrong type (`"250"` for an integer) and ~16k empty strings.
    A string that parses to the right type is converted; anything else is dropped (the
    parameter stays optional, it just has no default).
    """
    if _matches(schema, value):
        return True, value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False, None
        if _matches(schema, parsed):
            return True, parsed
    return False, None


def convert_parameters(parameters: dict[str, Any]) -> JsonSchema:
    """xLAM `{name: {type, description, default}}` -> BFCL `dict` schema with `required`.

    A parameter is optional if its type says so *or* it has a default value: like in Python,
    a parameter with a default can be omitted. (Without the second rule, 1 896 rows would have
    calls "missing" a required argument; with it, only 275 truly inconsistent rows remain.)
    """
    properties: dict[str, JsonSchema] = {}
    required: list[str] = []
    for name, spec in parameters.items():
        schema, optional = parse_type(str(spec.get("type", "")))
        if description := spec.get("description"):
            schema["description"] = description
        if "default" in spec:
            optional = True
            keep, default = coerce_default(schema, spec["default"])
            if keep:
                schema["default"] = default
        properties[name] = schema
        if not optional:
            required.append(name)
    return {"type": "dict", "properties": properties, "required": required}


def validate_calls(calls: list[dict[str, Any]], tools: list[dict[str, Any]]) -> None:
    """Never train the model to call a function or pass an argument that doesn't exist."""
    if not calls:
        raise RejectedError("no_call")
    schemas = {tool["name"]: tool["parameters"] for tool in tools}
    for call in calls:
        schema = schemas.get(call["name"])
        if schema is None:
            raise RejectedError("unknown_function", call["name"])
        arguments = set(call["arguments"])
        if unknown := arguments - set(schema["properties"]):
            raise RejectedError("unknown_argument", f"{call['name']}: {sorted(unknown)}")
        if missing := set(schema["required"]) - arguments:
            raise RejectedError("missing_required_argument", f"{call['name']}: {sorted(missing)}")


def convert_row(row: dict[str, Any]) -> Conversation:
    """One xLAM row -> a single-turn conversation (user query, assistant tool calls)."""
    try:
        raw_tools = json.loads(row["tools"])
        calls = json.loads(row["answers"])
    except json.JSONDecodeError as error:
        raise RejectedError("invalid_json", str(error)) from error

    tools = [
        {
            "name": tool["name"],
            "description": tool.get("description", "") + BFCL_PYTHON_HINT,
            "parameters": convert_parameters(tool.get("parameters", {})),
        }
        for tool in raw_tools
    ]
    validate_calls(calls, tools)

    return Conversation(
        messages=[
            {"role": "user", "content": row["query"]},
            {
                "role": "assistant",
                "content": "",
                # Tool calls keep the Hugging Face message format: the chat template reads
                # `tool_call.function.name/arguments` and renders them as <tool_call> blocks.
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in calls
                ],
            },
        ],
        tools=tools,
    )
