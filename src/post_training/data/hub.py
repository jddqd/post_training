"""Publish a prepared dataset to the Hugging Face Hub, with a dataset card."""

import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

FUNNEL_LABELS = {
    "source": "source rows",
    "valid": "valid (converted + calls checked)",
    "deduplicated": "after exact dedup (question + answer)",
    "decontaminated": "after decontamination against BFCL",
    "length_ok": "within max length",
    "train": "train",
    "val": "validation",
}


def dataset_card(report: dict[str, Any]) -> str:
    """README.md for the Hub: metadata header (license, splits) + how the data was made."""
    source = report["source"]
    config = report["config"]
    funnel = "\n".join(
        f"| {FUNNEL_LABELS.get(step, step)} | {count:,} |"
        for step, count in report["funnel"].items()
    )
    rejected = "\n".join(
        f"| `{reason}` | {count:,} |" for reason, count in report["rejected"].items()
    )
    categories = "\n".join(
        f"| {name} | {count:,} | {report['categories']['val'].get(name, 0):,} |"
        for name, count in report["categories"]["train"].items()
    )
    tokens = report["train_tokens"]
    decontamination = ""
    if report.get("decontamination"):
        decontamination = (
            f"\n   Decontaminated against the BFCL questions: dropped rows whose user message is a "
            f"BFCL question or shares {config['decontamination_ngram']}-grams covering at least "
            f"{config['decontamination_min_coverage']:.0%} of the shorter text."
        )
    example = {
        "messages": [
            {"role": "user", "content": "..."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "...", "arguments": {}}}],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ],
    }
    return f"""---
license: cc-by-4.0
language:
- en
task_categories:
- text-generation
tags:
- function-calling
- tool-use
- sft
source_datasets:
- {source["dataset"]}
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
---

# xLAM function calling, prepared for SFT

Derived from [{source["dataset"]}](https://huggingface.co/datasets/{source["dataset"]})
(revision `{source["revision"]}`), licensed CC-BY-4.0 by Salesforce. Please cite the original
APIGen / xLAM work when using this data.

Each line is a single-turn conversation: messages in the Hugging Face chat format, tools in the
exact function-doc format the [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) evaluation
puts in the prompt. Usable with `tokenizer.apply_chat_template(messages, tools=tools)`:

```json
{json.dumps(example)}
```

## Processing

1. Tools converted to BFCL's function-doc format: no `{{"type": "function"}}` wrapper, BFCL's
   Python-3 hint appended to descriptions, parameters as a `dict` schema with `properties` /
   `required` and BFCL type names (`string`, `integer`, `float`, `boolean`, `array`, `tuple`,
   `dict`). Rows using types without an equivalent (`Callable`) are dropped. A parameter is
   optional if its type says so or it has a default. Defaults of the wrong type are converted
   when possible (`"250"` → `250`), otherwise removed.
2. Calls validated against the tools: no unknown function, no unknown argument, no missing
   required argument.
3. Unicode NFC normalization and whitespace stripping.
4. Exact deduplication on (question, answer).{decontamination}
5. Tool order shuffled (in the source, the called tool is listed first in 49% of rows).
6. Rows longer than {config["max_length"]} tokens with the `{config["tokenizer"]}` chat template
   dropped (never truncated).
7. Train / validation split ({config["val_fraction"]:.0%}) stratified by BFCL-style category,
   seed {config["seed"]}.

| step | rows |
|---|---|
{funnel}

| rejection reason | rows |
|---|---|
{rejected}

`rejected.jsonl` contains every dropped source row with its reason.

## Statistics

| category | train | validation |
|---|---|---|
{categories}

Train tokens (`{config["tokenizer"]}` chat template): mean {tokens["mean"]}, median
{tokens["p50"]}, p99 {tokens["p99"]}, max {tokens["max"]}, total {tokens["total"]:,}.
"""


def push_to_hub(folder: Path, repo_id: str, private: bool, commit_message: str) -> str:
    """Create the dataset repo if needed and upload the folder. Returns the repo URL."""
    api = HfApi()
    # exist_ok: re-running the pipeline updates the same repo (visibility is left unchanged).
    url = api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=folder, repo_id=repo_id, repo_type="dataset", commit_message=commit_message
    )
    return str(url)
