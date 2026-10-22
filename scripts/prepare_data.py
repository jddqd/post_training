"""Prepare a source dataset into normalized train/val JSONL files.

Usage:
    uv run python scripts/prepare_data.py --config configs/data/xlam.yaml [section.key=value ...]

Writes to `output_dir`: train.jsonl, val.jsonl, rejected.jsonl (every dropped row with its
reason, for inspection), report.json (funnel, stats, pinned source revision, git state) and a
README.md dataset card. With `--push`, the folder is then pushed to the HF Hub repo `hub_repo`
(publishing is never implicit: commit the code first so the dataset references a clean commit).
"""

import argparse
import hashlib
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
from post_training.data.decontaminate import Decontaminator, load_benchmark_texts
from post_training.data.hub import dataset_card, push_to_hub
from post_training.data.prepare import PreparedData, prepare_xlam
from post_training.data.preprocess import category
from post_training.tracking import git_state


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(data: PreparedData, config: PrepareDataConfig, revision: str) -> dict[str, object]:
    lengths = sorted(data.train_lengths)
    # quantiles() needs at least 2 points (tiny datasets, e.g. in tests).
    percentiles = statistics.quantiles(lengths, n=100) if len(lengths) > 1 else lengths * 99
    return {
        "config": config.model_dump(mode="json"),
        "source": {"dataset": config.hf_dataset, "revision": revision},
        # The hash pins the exact benchmark questions used (they change between BFCL versions).
        "decontamination": None
        if config.decontamination_file is None
        else {
            "file": str(config.decontamination_file),
            "sha256": file_sha256(config.decontamination_file),
        },
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
    parser.add_argument("--push", action="store_true", help="Publish to the HF repo `hub_repo`")
    args = parser.parse_args()

    try:
        config = load_yaml_config(PrepareDataConfig, args.config, args.overrides)
    except (ValidationError, ValueError) as error:
        sys.exit(f"Invalid config: {error}")
    if args.push and not config.hub_repo:
        sys.exit("--push needs `hub_repo` in the config")

    # Pin the exact dataset version: "latest" today is not "latest" next month.
    revision = HfApi().dataset_info(config.hf_dataset, revision=config.revision).sha
    assert revision is not None
    print(f"source: {config.hf_dataset}@{revision[:8]}")
    rows = load_dataset(config.hf_dataset, split="train", revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

    decontaminator = None
    if config.decontamination_file is not None:
        benchmark = load_benchmark_texts(config.decontamination_file)
        decontaminator = Decontaminator(
            benchmark, config.decontamination_ngram, config.decontamination_min_coverage
        )
        print(f"decontaminating against {len(benchmark):,} benchmark questions")

    data = prepare_xlam(rows, tokenizer, config, decontaminator)

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

    if args.push and config.hub_repo:
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
