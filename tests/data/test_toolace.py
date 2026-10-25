import json
from pathlib import Path
from typing import Any

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from post_training.config import PrepareDataConfig
from post_training.data.prepare import prepare
from post_training.data.toolace import (
    TOOLS_MARKER,
    convert_row,
    normalize_schema,
    normalize_type,
    parse_calls,
)
from post_training.data.xlam import BFCL_PYTHON_HINT, RejectedError

TRENDS = {
    "name": "Market Trends API",
    "description": "Latest market trends.",
    "parameters": {
        "type": "dict",
        "properties": {
            "trend_type": {"description": "Trend type.", "type": "string"},
            "count": {"description": "How many.", "type": "int"},
            "from": {"description": "Start date.", "type": ["string", "null"]},
        },
        "required": ["trend_type"],
    },
    "required": None,
}
AGENT = {
    "name": "Get Agent's Listings (v2)",
    "description": "Listings of an agent.",
    "parameters": {"type": "dict", "properties": {"agent": {"type": "string"}}, "required": []},
    "required": None,
}


def system(tools: list[dict[str, Any]]) -> str:
    return (
        f"You are an expert in composing functions.\n{TOOLS_MARKER}{json.dumps(tools)}. \nUse it."
    )


def row(turns: list[tuple[str, str]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "system": system(tools or [TRENDS, AGENT]),
        "conversations": [{"from": role, "value": value} for role, value in turns],
    }


def test_types_are_mapped_to_bfcl_names() -> None:
    assert normalize_type("int") == "integer"
    assert normalize_type("object") == "dict"
    assert normalize_type(["integer", "null"]) == "integer"  # nullable = optional, same type
    with pytest.raises(RejectedError, match="union_type"):
        normalize_type(["string", "integer"])
    with pytest.raises(RejectedError, match="unsupported_type"):
        normalize_type("datetime")


def test_nested_schemas_are_normalized() -> None:
    schema = {"type": "array", "items": {"type": "dict", "properties": {"n": {"type": "int"}}}}

    assert normalize_schema(schema) == {
        "type": "array",
        "items": {"type": "dict", "properties": {"n": {"type": "integer"}}},
    }


def test_calls_with_unusual_names_and_python_keywords_as_arguments() -> None:
    names = ["Market Trends API", "Get Agent's Listings (v2)", "Get Agent"]
    text = (
        '[Market Trends API(trend_type="GAINERS", count=-3, from=null), '
        "Get Agent's Listings (v2)(agent='O\\'Neil, (Jr)'), "
        'Market Trends API(trend_type="x", tags=["a", "b"], opts={"k": True})]'
    )

    assert parse_calls(text, names) == [
        {
            "name": "Market Trends API",
            "arguments": {"trend_type": "GAINERS", "count": -3, "from": None},
        },
        {"name": "Get Agent's Listings (v2)", "arguments": {"agent": "O'Neil, (Jr)"}},
        {
            "name": "Market Trends API",
            "arguments": {"trend_type": "x", "tags": ["a", "b"], "opts": {"k": True}},
        },
    ]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ('[Unknown(x="1")]', "unknown_function"),
        ('[Market Trends API("GAINERS")]', "invalid_call"),  # positional argument
        ("[Market Trends API(trend_type=GAINERS)]", "invalid_call"),  # not a literal
        ('[Market Trends API(trend_type="a"', "invalid_call"),  # unbalanced
        ('[Market Trends API(trend_type="a") Get Agent(agent="b")]', "invalid_call"),  # no comma
        ("[]", "invalid_call"),
    ],
)
def test_malformed_calls_are_rejected(text: str, reason: str) -> None:
    with pytest.raises(RejectedError, match=reason):
        parse_calls(text, ["Market Trends API", "Get Agent"])


def test_multi_turn_row_becomes_a_structured_conversation() -> None:
    conversation = convert_row(
        row(
            [
                ("user", "Top gainers?"),
                ("assistant", '[Market Trends API(trend_type="GAINERS")]'),
                ("tool", '[{"name": "Market Trends API", "results": {"top": "NVDA"}}]'),
                ("assistant", "NVDA is the top gainer."),
                ("user", "Thanks! What can you do?"),
                ("assistant", "I can fetch market trends and agent listings."),
            ]
        )
    )

    assert [m["role"] for m in conversation.messages] == [
        "user", "assistant", "tool", "assistant", "user", "assistant",
    ]  # fmt: skip
    assert conversation.messages[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "Market Trends API", "arguments": {"trend_type": "GAINERS"}},
            }
        ],
    }
    trends = conversation.tools[0]
    assert trends["description"] == "Latest market trends." + BFCL_PYTHON_HINT
    assert trends["parameters"]["properties"]["count"]["type"] == "integer"
    assert trends["parameters"]["properties"]["from"]["type"] == "string"
    assert set(trends) == {"name", "description", "parameters"}  # top-level "required" dropped


def answer(call: str) -> dict[str, Any]:
    return row([("user", "Q"), ("assistant", call)])


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (
            {"system": "Here are the tools you can use:\n{}", "conversations": []},
            "nonstandard_format",
        ),
        (answer("[Market Trends API(count=1)]"), "missing_required_argument"),
        (answer('[Market Trends API(trend_type="a", x=1)]'), "unknown_argument"),
        (row([("user", "Q"), ("tool", "{}"), ("assistant", "A")]), "invalid_turn_order"),
        (row([("user", "Q"), ("assistant", "A"), ("user", "Q2")]), "invalid_turn_order"),
        (row([("user", "Q"), ("assistant", "A")], tools=[TRENDS, TRENDS]), "duplicate_tool_name"),
    ],
)
def test_rows_that_cannot_be_converted_faithfully_are_rejected(
    data: dict[str, Any], reason: str
) -> None:
    with pytest.raises(RejectedError, match=reason):
        convert_row(data)


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    return tokenizer


def test_prepare_runs_the_shared_pipeline_on_toolace_rows(
    tok: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    rows = [
        *[
            row(
                [("user", f"Trends {i}?"), ("assistant", f'[Market Trends API(trend_type="T{i}")]')]
            )
            for i in range(10)
        ],
        row([("user", "Weather?"), ("assistant", "None of the functions can do that.")]),
        row([("user", "Q"), ("assistant", "[broken")]),
    ]
    config = PrepareDataConfig.model_validate(
        {"source": "toolace", "tokenizer": "Qwen/Qwen3-0.6B-Base", "output_dir": tmp_path}
    )

    data = prepare(rows, tok, config)

    assert dict(data.funnel)["valid"] == 11
    assert data.rejected_counts() == {"invalid_call": 1}
    assert len(data.train) + len(data.val) == 11
