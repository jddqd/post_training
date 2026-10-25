import json
from pathlib import Path
from typing import Any

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from post_training.config import PrepareDataConfig
from post_training.data.conversation import load_jsonl, save_jsonl
from post_training.data.decontaminate import BenchmarkText, Decontaminator
from post_training.data.prepare import prepare

TOOL = {
    "name": "get_weather",
    "description": "Weather of a city",
    "parameters": {"city": {"type": "str", "description": "City"}},
}
CALLABLE_TOOL = {"name": "integrate", "parameters": {"f": {"type": "Callable[[float], float]"}}}


def row(query: str, city: str, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "query": query,
        "tools": json.dumps(tools or [TOOL]),
        "answers": json.dumps([{"name": "get_weather", "arguments": {"city": city}}]),
    }


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    return tokenizer


def config(tmp_path: Path, **overrides: Any) -> PrepareDataConfig:
    return PrepareDataConfig.model_validate(
        {"tokenizer": "Qwen/Qwen3-0.6B-Base", "output_dir": tmp_path, **overrides}
    )


def test_funnel_and_rejections(tok: PreTrainedTokenizerBase, tmp_path: Path) -> None:
    rows = [
        *[row(f"Weather in city {i}?", f"City{i}") for i in range(20)],
        row("Weather in city 0?  ", "City0"),  # duplicate once cleaned (trailing spaces)
        row("Integrate", "X", tools=[TOOL, CALLABLE_TOOL]),  # unsupported type
    ]

    data = prepare(rows, tok, config(tmp_path, val_fraction=0.1))

    assert data.funnel == [
        ("source", 22),
        ("valid", 21),
        ("deduplicated", 20),
        ("length_ok", 20),
        ("train", 18),
        ("val", 2),
    ]
    assert data.rejected_counts() == {"unsupported_type": 1, "duplicate": 1}
    assert len(data.train_lengths) == len(data.train)


def test_too_long_rows_are_dropped_not_truncated(
    tok: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    rows = [row("Short?", "Paris"), row("Long " * 300 + "?", "Paris")]

    data = prepare(rows, tok, config(tmp_path, max_length=300))

    assert data.rejected_counts() == {"too_long": 1}
    assert max(data.train_lengths) <= 300


def test_output_round_trips_through_jsonl(tok: PreTrainedTokenizerBase, tmp_path: Path) -> None:
    data = prepare([row(f"Q{i}?", f"C{i}") for i in range(5)], tok, config(tmp_path))
    path = tmp_path / "train.jsonl"

    save_jsonl(path, data.train)

    assert load_jsonl(path) == data.train


def test_rows_overlapping_the_benchmark_are_dropped(
    tok: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    rows = [row("Weather in Paris?", "Paris"), row("Weather in Lyon?", "Lyon")]
    decontaminator = Decontaminator([BenchmarkText("simple_python_1", "weather in LYON")])

    data = prepare(rows, tok, config(tmp_path, val_fraction=0.1), decontaminator)

    assert ("decontaminated", 1) in data.funnel
    [rejected] = data.rejected
    assert rejected.reason == "contaminated"
    assert rejected.detail == "simple_python_1 (100%)"
