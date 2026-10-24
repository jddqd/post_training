import json

import yaml

from post_training.data.hub import CARD_EXAMPLE, dataset_card
from post_training.data.xlam import convert_row

REPORT = {
    "config": {
        "max_length": 2048,
        "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
        "val_fraction": 0.03,
        "seed": 0,
    },
    "source": {"dataset": "Salesforce/xlam-function-calling-60k", "revision": "abc123"},
    "funnel": {"source": 100, "valid": 90, "train": 87, "val": 3},
    "rejected": {"duplicate": 10},
    "categories": {"train": {"simple": 87}, "val": {"simple": 3}},
    "train_tokens": {"mean": 500.0, "p50": 480, "p99": 1200, "max": 2000, "total": 43500},
}


def test_card_metadata_declares_license_source_and_splits() -> None:
    card = dataset_card(REPORT)

    header = yaml.safe_load(card.split("---")[1])
    assert header["license"] == "cc-by-4.0"  # same license as the source (CC-BY requires it)
    assert header["source_datasets"] == ["Salesforce/xlam-function-calling-60k"]
    splits = {f["split"]: f["path"] for f in header["configs"][0]["data_files"]}
    assert splits == {"train": "train.jsonl", "validation": "val.jsonl"}


def test_card_body_cites_the_pinned_source_revision_and_funnel() -> None:
    card = dataset_card(REPORT)

    assert "revision `abc123`" in card
    assert "| train | 87 |" in card
    assert "| `duplicate` | 10 |" in card


def test_card_example_is_exactly_what_the_conversion_produces() -> None:
    # The card's example must not drift from the real data format (it once still showed the
    # OpenAI format after the switch to BFCL function docs).
    xlam_row = {
        "query": "What is the weather in Paris?",
        "tools": json.dumps(
            [
                {
                    "name": "get_weather",
                    "description": "Get the current weather of a city.",
                    "parameters": {
                        "city": {"type": "str", "description": "The city name."},
                        "unit": {
                            "type": "str, optional",
                            "description": "celsius or fahrenheit.",
                            "default": "celsius",
                        },
                    },
                }
            ]
        ),
        "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}]),
    }

    assert convert_row(xlam_row).model_dump() == CARD_EXAMPLE
    assert json.dumps(CARD_EXAMPLE, indent=2) in dataset_card(REPORT)
