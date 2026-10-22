import random
from typing import Any

from post_training.data.conversation import Conversation
from post_training.data.preprocess import (
    category,
    clean,
    dedup_key,
    normalize_text,
    shuffle_tools,
    stratified_split,
)


def tool(name: str) -> dict[str, Any]:
    return {"name": name, "description": "", "parameters": {"type": "dict", "properties": {}}}


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def conversation(query: str, calls: list[dict[str, Any]], tools: list[str]) -> Conversation:
    return Conversation(
        messages=[
            {"role": "user", "content": query},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ],
        tools=[tool(name) for name in tools],
    )


def test_normalize_text_unifies_unicode_and_strips() -> None:
    decomposed = "été"  # 'e' + combining accent, twice
    assert normalize_text(f"  {decomposed} \n") == "été"


def test_clean_normalizes_string_contents_only() -> None:
    cleaned = clean(conversation("  Météo ?  ", [call("f")], ["f"]))

    assert cleaned.messages[0]["content"] == "Météo ?"
    assert cleaned.messages[1]["tool_calls"] == [call("f")]


def test_dedup_key_ignores_tools_but_not_arguments() -> None:
    a = conversation("Q", [call("f", x=1)], ["f", "g"])
    same_with_other_tools = conversation("Q", [call("f", x=1)], ["f"])
    other_argument = conversation("Q", [call("f", x=2)], ["f", "g"])

    assert dedup_key(a) == dedup_key(same_with_other_tools)
    assert dedup_key(a) != dedup_key(other_argument)


def test_shuffle_tools_is_deterministic_and_keeps_every_tool() -> None:
    c = conversation("Q", [call("a")], ["a", "b", "c", "d", "e"])

    first = shuffle_tools(c, random.Random(0))
    second = shuffle_tools(c, random.Random(0))

    assert first.tools == second.tools
    assert sorted(t["name"] for t in first.tools) == ["a", "b", "c", "d", "e"]
    assert c.tools == [tool(n) for n in "abcde"]  # input untouched


def test_category_follows_bfcl_definitions() -> None:
    assert category(conversation("Q", [call("f")], ["f"])) == "simple"
    assert category(conversation("Q", [call("f")], ["f", "g"])) == "multiple"
    assert category(conversation("Q", [call("f"), call("f")], ["f"])) == "parallel"
    assert category(conversation("Q", [call("f"), call("g")], ["f", "g"])) == "parallel_multiple"


def test_stratified_split_keeps_proportions_and_is_deterministic() -> None:
    items = [("a", i) for i in range(900)] + [("b", i) for i in range(100)]

    train, val = stratified_split(items, key=lambda x: x[0], val_fraction=0.1, seed=0)

    assert len(val) == 100
    assert sum(1 for name, _ in val if name == "b") == 10  # 10% of each group
    assert set(train).isdisjoint(val)
    assert len(train) + len(val) == len(items)
    assert stratified_split(items, key=lambda x: x[0], val_fraction=0.1, seed=0) == (train, val)
