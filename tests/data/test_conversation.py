import json
from pathlib import Path
from typing import Any

import pytest

from post_training.data.conversation import load_jsonl

VALID: dict[str, Any] = {
    "messages": [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Salut !"},
    ],
}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_loads_conversations_and_defaults_tools_to_empty(tmp_path: Path) -> None:
    with_tools = {**VALID, "tools": [{"type": "function", "function": {"name": "f"}}]}
    path = write_jsonl(tmp_path / "train.jsonl", [VALID, with_tools])

    conversations = load_jsonl(path)

    assert len(conversations) == 2
    assert conversations[0].tools == []
    assert conversations[1].tools[0]["function"]["name"] == "f"


def test_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(VALID) + "\n\n" + json.dumps(VALID) + "\n")

    assert len(load_jsonl(path)) == 2


@pytest.mark.parametrize(
    ("record", "error"),
    [
        ({"messages": []}, "at least 1 item"),
        ({"messages": [{"role": "user", "content": "?"}]}, "no assistant message"),
        ({"messages": [{"role": "robot", "content": "?"}]}, "role must be one of"),
        ({**VALID, "tool": []}, "Extra inputs are not permitted"),  # typo for "tools"
    ],
)
def test_invalid_record_reports_its_line_number(
    tmp_path: Path, record: dict[str, Any], error: str
) -> None:
    path = write_jsonl(tmp_path / "train.jsonl", [VALID, record])

    with pytest.raises(ValueError, match=rf"train\.jsonl:2:(.|\n)*{error}"):
        load_jsonl(path)
