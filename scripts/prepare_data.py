"""Prepare a source dataset into normalized train/val JSONL files.

Usage:
    uv run python scripts/prepare_data.py --config configs/data/xlam.yaml [section.key=value ...]

Writes to `output_dir`: train.jsonl, val.jsonl, rejected.jsonl (every dropped row with its
reason, for inspection), report.json (funnel, stats, pinned source revision, git state) and a
README.md dataset card. If `hub_repo` is set, the folder is then pushed to the HF Hub.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from pydantic import ValidationError
from transformers import AutoTokenizer

from post_training.config import PrepareDataConfig, load_yaml_config
from post_training.data.conversation import save_jsonl
from post_training.data.hub import dataset_card, push_to_hub
from post_training.data.prepare import PreparedData, prepare_xlam
from post_training.data.preprocess import category
from post_training.tracking import git_state


def report(data: PreparedData, config: PrepareDataConfig, revision: str) -> dict[str, object]:
    lengths = sorted(data.train_lengths)
    percentiles = statistics.quantiles(lengths, n=100)
    return {
        "config": config.model_dump(mode="json"),
        "source": {"dataset": config.hf_dataset, "revision": revision},
        "git": git_state(),
        "funnel": dict(data.funnel),
        "rejected": dict(data.rejected_counts().most_common()),
        "categories": {
            "train": dict(Counter(category(c) for c in data.train).most_common()),
            "val": dict(Counter(category(c) for c in data.val).most_common()),
        },
        "train_tokens": {
            "mean": round(statistics.mean(lengths), 1),
            "p50": percentiles[49],
            "p90": percentiles[89],
            "p99": percentiles[98],
            "max": lengths[-1],
            "total": sum(lengths),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="key=value, e.g. max_length=1024")
    args = parser.parse_args()

    try:
        config = load_yaml_config(PrepareDataConfig, args.config, args.overrides)
    except (ValidationError, ValueError) as error:
        sys.exit(f"Invalid config: {error}")

    # Pin the exact dataset version: "latest" today is not "latest" next month.
    revision = HfApi().dataset_info(config.hf_dataset, revision=config.revision).sha
    assert revision is not None
    print(f"source: {config.hf_dataset}@{revision[:8]}")
    rows = load_dataset(config.hf_dataset, split="train", revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

    data = prepare_xlam(rows, tokenizer, config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(config.output_dir / "train.jsonl", data.train)
    save_jsonl(config.output_dir / "val.jsonl", data.val)
    with (config.output_dir / "rejected.jsonl").open("w") as f:
        for r in data.rejected:
            record = {"reason": r.reason, "detail": r.detail, "row": dict(r.row)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = report(data, config, revision)
    (config.output_dir / "report.json").write_text(json.dumps(summary, indent=2))
    (config.output_dir / "README.md").write_text(dataset_card(summary))

    print("\nfunnel:")
    for step, count in data.funnel:
        print(f"  {step:<14} {count:>7,}")
    print("\nrejected:")
    for reason, count in data.rejected_counts().most_common():
        print(f"  {reason:<28} {count:>6,}")
    print(f"\ntrain tokens: {summary['train_tokens']}")
    print(f"written to {config.output_dir}/")

    if config.hub_repo:
        git = summary["git"]
        assert isinstance(git, dict)
        commit = (git["commit"] or "unknown")[:8] + ("-dirty" if git["dirty"] else "")
        url = push_to_hub(
            config.output_dir,
            config.hub_repo,
            config.hub_private,
            commit_message=f"Prepare from {config.hf_dataset}@{revision[:8]} (code {commit})",
        )
        visibility = "private" if config.hub_private else "public"
        print(f"pushed to {url} ({visibility} on creation)")


if __name__ == "__main__":
    main()
