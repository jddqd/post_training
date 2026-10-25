"""Publish a prepared dataset to the Hugging Face Hub, with a dataset card."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from post_training.data.xlam import BFCL_PYTHON_HINT

FUNNEL_LABELS = {
    "source": "source rows",
    "valid": "valid (converted + calls checked)",
    "deduplicated": "after exact dedup (question + answer)",
    "decontaminated": "after decontamination against BFCL",
    "length_ok": "within max length",
    "train": "train",
    "val": "validation",
}

# One row in the exact format of train.jsonl / val.jsonl (checked against the real format in
# tests): BFCL function docs, no {"type": "function"} wrapper, HF chat-format tool calls.
CARD_EXAMPLE: dict[str, Any] = {
    "messages": [
        {"role": "user", "content": "What is the weather in Paris?"},
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
    ],
    "tools": [
        {
            "name": "get_weather",
            "description": "Get the current weather of a city." + BFCL_PYTHON_HINT,
            "parameters": {
                "type": "dict",
                "properties": {
                    "city": {"type": "string", "description": "The city name."},
                    "unit": {
                        "type": "string",
                        "description": "celsius or fahrenheit.",
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        }
    ],
}


@dataclass(frozen=True)
class SourceCard:
    """What the dataset card says about one source dataset."""

    title: str
    license: str  # HF license id, the same as the source (both licenses require it)
    credit: str  # license holder and what to cite
    shape: str  # what a line holds
    conversion: str  # processing step 1, in Markdown (list item continuation indented 3)
    tool_order_bias: str  # how often the source lists the called tool first


SOURCE_CARDS = {
    "xlam": SourceCard(
        title="xLAM function calling, prepared for SFT",
        license="cc-by-4.0",
        credit="licensed CC-BY-4.0 by Salesforce. Please cite the original\nAPIGen / xLAM work",
        shape="Each line is a single-turn conversation",
        conversion='Tools converted to BFCL\'s function-doc format: no `{"type": "function"}` '
        """wrapper, BFCL's
   Python-3 hint appended to descriptions, parameters as a `dict` schema with `properties` /
   `required` and BFCL type names (`string`, `integer`, `float`, `boolean`, `array`, `tuple`,
   `dict`). Rows using types without an equivalent (`Callable`) are dropped. A parameter is
   optional if its type says so or it has a default. Defaults of the wrong type are converted
   when possible (`"250"` → `250`), otherwise removed.
2. Calls validated against the tools: no unknown function, no unknown argument, no missing
   required argument.""",
        tool_order_bias="49% of rows",
    ),
    "toolace": SourceCard(
        title="ToolACE, prepared for SFT",
        license="apache-2.0",
        credit="licensed Apache-2.0 by Team-ACE. Please cite the original\nToolACE work",
        shape=(
            "Each line is a conversation, single- or multi-turn (tool responses, text answers\n"
            "when no function applies or a parameter is missing)"
        ),
        conversion="""Tools parsed from the system prompt and normalized to BFCL's
   function-doc format (the system prompt itself is dropped: the chat template renders the
   tools as BFCL does): BFCL's Python-3 hint appended to descriptions, BFCL type names
   (`int` → `integer`, nullable unions such as `["integer", "null"]` → `integer`). Rows using
   another tool format (ToolACE's format augmentation: other key names, custom call
   syntaxes), non-nullable unions, or two tools with the same name are dropped.
2. Python-like calls (`[Tool Name(arg="value")]`) parsed into structured `tool_calls` and
   validated against the tools: no unknown function, no unknown argument, no missing
   required argument. Conversations must start with a user message and end with an
   assistant answer.""",
        tool_order_bias="48% of rows with several tools, 29% expected by chance",
    ),
}


def dataset_card(report: dict[str, Any]) -> str:
    """README.md for the Hub: metadata header (license, splits) + how the data was made."""
    source = report["source"]
    config = report["config"]
    card = SOURCE_CARDS[config.get("source", "xlam")]
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
    return f"""---
license: {card.license}
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

# {card.title}

Derived from [{source["dataset"]}](https://huggingface.co/datasets/{source["dataset"]})
(revision `{source["revision"]}`), {card.credit} when using this data.

{card.shape}: messages in the Hugging Face chat format, tools in the
exact function-doc format the [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) evaluation
puts in the prompt. Usable with `tokenizer.apply_chat_template(messages, tools=tools)`:

```json
{json.dumps(CARD_EXAMPLE, indent=2)}
```

## Processing

1. {card.conversion}
3. Unicode NFC normalization and whitespace stripping.
4. Exact deduplication on (question, answer).{decontamination}
5. Tool order shuffled (in the source, the called tool is listed first in {card.tool_order_bias}).
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


def combined_card(report: dict[str, Any]) -> str:
    """README.md for a combination of prepared datasets (scripts/combine_datasets.py)."""
    sources = report["sources"]
    licenses = sorted({SOURCE_CARDS[s["source"]].license for s in sources})
    license_header = (
        f"license: {licenses[0]}"
        if len(licenses) == 1
        else "license: other\nlicense_name: " + "-and-".join(licenses)
    )
    source_datasets = "\n".join(f"- {s['dataset']}" for s in sources)

    def credit(source: dict[str, Any]) -> str:
        prepared = source.get("hub_repo")
        link = f" ([{prepared}](https://huggingface.co/datasets/{prepared}))" if prepared else ""
        dataset = source["dataset"]
        return (
            f"- **{source['name']}**{link}: from [{dataset}](https://huggingface.co/datasets/"
            f"{dataset}) (revision `{source['revision']}`), "
            f"{SOURCE_CARDS[source['source']].credit.replace(chr(10), ' ')}."
        )

    credits = "\n".join(credit(source) for source in sources)
    kept = "\n".join(
        f"| {s['name']} | {s['kept']['train']:,} | {s['kept']['val']:,} | "
        f"{s['train_tokens']['total']:,} |"
        for s in sources
    )
    dropped = "\n".join(f"| {reason} | {count:,} |" for reason, count in report["dropped"].items())
    categories = "\n".join(
        f"| {name} | {count:,} | {report['categories']['val'].get(name, 0):,} |"
        for name, count in report["categories"]["train"].items()
    )
    return f"""---
{license_header}
language:
- en
task_categories:
- text-generation
tags:
- function-calling
- tool-use
- sft
source_datasets:
{source_datasets}
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
---

# {" + ".join(s["name"] for s in sources)}: function calling, prepared for SFT

The union of these prepared datasets, same format and processing (see each one's card):

{credits}

Each row keeps the license of its source dataset: cite the original works when using this data.

Each line is a conversation: messages in the Hugging Face chat format, tools in the exact
function-doc format the [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) evaluation puts
in the prompt. Usable with `tokenizer.apply_chat_template(messages, tools=tools)`:

```json
{json.dumps(CARD_EXAMPLE, indent=2)}
```

## Combination

Each source keeps its own train / validation split (the validation set is the union of the
sources' validation sets). Exact duplicates across sources are kept once (in the first source
listed), and a validation conversation found in another source's train set is dropped. Rows are
shuffled (seed {report["config"]["seed"]}).

| source | train | validation | train tokens |
|---|---|---|---|
{kept}

| dropped | rows |
|---|---|
{dropped or "| none | 0 |"}

## Statistics

| category | train | validation |
|---|---|---|
{categories}
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
