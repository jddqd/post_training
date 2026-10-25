"""Combine prepared datasets (prepare_data.py outputs) into one, optionally pushed to the HF Hub.

Usage:
    uv run python scripts/combine_datasets.py --config configs/data/xlam_toolace.yaml [--push]

Writes to `output_dir`: train.jsonl, val.jsonl, report.json (rows kept per source, drops,
categories, each source's pinned revision, git state) and a README.md dataset card.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from post_training.config import CombineDataConfig, load_yaml_config
from post_training.data.combine import Source, combine
from post_training.data.conversation import load_jsonl, save_jsonl
from post_training.data.hub import combined_card, push_to_hub
from post_training.data.preprocess import category
from post_training.tracking import git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="key=value, e.g. seed=1")
    parser.add_argument("--push", action="store_true", help="Publish to the HF repo `hub_repo`")
    args = parser.parse_args()

    try:
        config = load_yaml_config(CombineDataConfig, args.config, args.overrides)
    except (ValidationError, ValueError) as error:
        sys.exit(f"Invalid config: {error}")
    if args.push and not config.hub_repo:
        sys.exit("--push needs `hub_repo` in the config")

    sources, source_reports = [], []
    for source in config.sources:
        source_reports.append(json.loads((source.dir / "report.json").read_text()))
        sources.append(
            Source(
                source.name,
                train=load_jsonl(source.dir / "train.jsonl"),
                val=load_jsonl(source.dir / "val.jsonl"),
            )
        )
        print(f"{source.name}: {len(sources[-1].train):,} train, {len(sources[-1].val):,} val")

    data = combine(sources, config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(config.output_dir / "train.jsonl", data.train)
    save_jsonl(config.output_dir / "val.jsonl", data.val)
    report = {
        "config": config.model_dump(mode="json"),
        "sources": [
            {
                "name": source.name,
                "source": source_report["config"]["source"],
                "dataset": source_report["source"]["dataset"],
                "revision": source_report["source"]["revision"],
                "hub_repo": source_report["config"].get("hub_repo"),
                "kept": data.kept[source.name],
                "train_tokens": source_report["train_tokens"],
            }
            for source, source_report in zip(config.sources, source_reports, strict=True)
        ],
        "git": git_state(),
        "dropped": dict(data.dropped.most_common()),
        "categories": {
            "train": dict(Counter(category(c) for c in data.train).most_common()),
            "val": dict(Counter(category(c) for c in data.val).most_common()),
        },
    }
    (config.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    (config.output_dir / "README.md").write_text(combined_card(report))

    print(f"\ncombined: {len(data.train):,} train, {len(data.val):,} val")
    for reason, count in data.dropped.most_common():
        print(f"  dropped {reason}: {count:,}")
    print(f"written to {config.output_dir}/")

    if args.push and config.hub_repo:
        git = report["git"]
        assert isinstance(git, dict)
        commit = (git["commit"] or "unknown")[:8] + ("-dirty" if git["dirty"] else "")
        names = " + ".join(source.name for source in config.sources)
        url = push_to_hub(
            config.output_dir,
            config.hub_repo,
            config.hub_private,
            commit_message=f"Combine {names} (code {commit})",
        )
        visibility = "private" if config.hub_private else "public"
        print(f"pushed to {url} ({visibility} on creation)")


if __name__ == "__main__":
    main()
