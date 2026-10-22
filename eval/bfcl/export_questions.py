"""Export every user message of the BFCL benchmark, for decontaminating the training data.

The training environment never imports BFCL: it reads this plain JSONL file instead.

Usage:
    uv run --project eval/bfcl python eval/bfcl/export_questions.py [--output <file>]
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument

REPO_ROOT = Path(__file__).resolve().parents[2]


def benchmark_questions() -> list[dict[str, Any]]:
    """One record per user message, across all categories (multi-turn included)."""
    records = []
    for category in parse_test_category_argument(["all"]):
        for entry in load_dataset_entry(category):
            for turn in entry.get("question", []):
                for message in turn:
                    text = message.get("content")
                    if message.get("role") == "user" and isinstance(text, str) and text.strip():
                        records.append({"id": entry["id"], "category": category, "text": text})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data" / "bfcl" / "questions.jsonl"
    )
    args = parser.parse_args()

    records = benchmark_questions()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    categories = len({r["category"] for r in records})
    print(
        f"{len(records)} user messages from {categories} categories "
        f"(bfcl-eval {version('bfcl-eval')}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
