import json
from typing import Any

import pytest

from post_training.data.xlam import (
    RejectedError,
    coerce_default,
    convert_parameters,
    convert_row,
    parse_type,
    split_top_level,
)


def make_row(tools: list[dict[str, Any]], answers: list[dict[str, Any]]) -> dict[str, Any]:
    return {"query": "Q ?", "tools": json.dumps(tools), "answers": json.dumps(answers)}


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Weather of a city",
    "parameters": {
        "city": {"type": "str", "description": "City name"},
        "unit": {"type": "str, optional", "description": "C or F"},
    },
}


# ---------------------------------------------------------------- type conversion


def test_split_top_level_ignores_separators_inside_brackets() -> None:
    assert split_top_level("Tuple[int, int], optional") == ["Tuple[int, int]", "optional"]


@pytest.mark.parametrize(
    ("type_string", "schema"),
    [
        ("str", {"type": "string"}),
        ("int", {"type": "integer"}),
        ("float", {"type": "number"}),
        ("bool", {"type": "boolean"}),
        ("list", {"type": "array"}),
        ("List", {"type": "array"}),
        ("Dict", {"type": "object"}),
        ("set", {"type": "array", "uniqueItems": True}),
        ("List[str]", {"type": "array", "items": {"type": "string"}}),
        (
            "List[List[int]]",
            {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        ),
        ("List[Union[int, float]]", {"type": "array", "items": {"type": "number"}}),
        (
            "Tuple[float, float]",
            {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        ),
    ],
)
def test_python_types_map_to_json_schema(type_string: str, schema: dict[str, Any]) -> None:
    assert parse_type(type_string) == (schema, False)


@pytest.mark.parametrize(
    "type_string", ["str, optional", "str, default='elonmusk'", "str, optional, default 'London'"]
)
def test_optional_is_read_from_the_type_string(type_string: str) -> None:
    assert parse_type(type_string) == ({"type": "string"}, True)


@pytest.mark.parametrize("type_string", ["Callable[[float], float]", "", "Tuple[int, str]"])
def test_types_without_a_faithful_json_schema_are_rejected(type_string: str) -> None:
    with pytest.raises(RejectedError) as error:
        parse_type(type_string)
    assert error.value.reason == "unsupported_type"


def test_parameters_with_a_default_are_not_required() -> None:
    schema = convert_parameters(
        {
            "city": {"type": "str", "description": "City"},
            "unit": {"type": "str", "default": "C"},  # no "optional" but a default
            "days": {"type": "int, optional"},
        }
    )

    assert schema == {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City"},
            "unit": {"type": "string", "default": "C"},
            "days": {"type": "integer"},
        },
        "required": ["city"],
    }


@pytest.mark.parametrize(
    ("schema", "value", "expected"),
    [
        ({"type": "integer"}, 5, (True, 5)),
        ({"type": "integer"}, "250", (True, 250)),  # parsable string -> converted
        ({"type": "integer"}, "", (False, None)),  # empty string -> dropped
        ({"type": "integer"}, 0.7, (False, None)),  # wrong type, not a string -> dropped
        ({"type": "integer"}, True, (False, None)),  # bool is not an integer in JSON Schema
        ({"type": "number"}, 3, (True, 3)),  # an integer is a valid number
        ({"type": "string"}, "Paris", (True, "Paris")),
        ({"type": "array"}, "[1, 2]", (True, [1, 2])),
        ({"anyOf": [{"type": "string"}]}, 1, (True, 1)),  # no single type: kept as is
    ],
)
def test_defaults_must_match_their_type(
    schema: dict[str, Any], value: Any, expected: tuple[bool, Any]
) -> None:
    assert coerce_default(schema, value) == expected


def test_a_dropped_default_still_makes_the_parameter_optional() -> None:
    schema = convert_parameters({"limit": {"type": "int", "default": ""}})

    assert schema["properties"]["limit"] == {"type": "integer"}
    assert schema["required"] == []


# ---------------------------------------------------------------- row conversion


def test_converts_a_row_to_a_chat_conversation() -> None:
    row = make_row([WEATHER_TOOL], [{"name": "get_weather", "arguments": {"city": "Paris"}}])

    conversation = convert_row(row)

    assert conversation.messages == [
        {"role": "user", "content": "Q ?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                }
            ],
        },
    ]
    [tool] = conversation.tools
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["parameters"]["required"] == ["city"]


@pytest.mark.parametrize(
    ("answers", "reason"),
    [
        ([], "no_call"),
        ([{"name": "get_time", "arguments": {}}], "unknown_function"),
        ([{"name": "get_weather", "arguments": {"city": "P", "lang": "fr"}}], "unknown_argument"),
        ([{"name": "get_weather", "arguments": {"unit": "C"}}], "missing_required_argument"),
    ],
)
def test_invalid_calls_are_rejected(answers: list[dict[str, Any]], reason: str) -> None:
    with pytest.raises(RejectedError) as error:
        convert_row(make_row([WEATHER_TOOL], answers))
    assert error.value.reason == reason


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(RejectedError) as error:
        convert_row({"query": "Q", "tools": "[{", "answers": "[]"})
    assert error.value.reason == "invalid_json"
