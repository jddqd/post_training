"""Build the RL task set from OpenCodeInstruct and publish it on the HF Hub.

Funnel: one shard → rows whose tests all call one function defined in the reference (to_task) →
dataset score > --min-score → a random sample of --size tasks → train / val / test.

    uv run python scripts/build_dataset.py            # writes data/tasks/
    uv run python scripts/build_dataset.py --push     # and publishes it (needs `hf auth login`)
"""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi

from rl_swe_agent.opencodeinstruct import REPO, SHARD, TASKS_REPO, download_shard, load_tasks

CARD = """---
license: cc-by-4.0
pretty_name: OpenCodeInstruct RL tasks
task_categories:
- text-generation
tags:
- code
- reinforcement-learning
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/val.jsonl
  - split: test
    path: data/test.jsonl
---

# OpenCodeInstruct RL tasks

Code tasks for RL with a verifiable reward, derived from
[{source}](https://huggingface.co/datasets/{source}) (CC BY 4.0, NVIDIA): write one Python
function, checked by the row's `assert` unit tests.

## Fields

| Field | Content |
|---|---|
| `id` | Row id in {source} |
| `problem` | The problem statement (`input`) |
| `function`, `signature` | The function every test calls, and its `def` line in the reference |
| `tests` | The `assert` statements, one per element |
| `reference` | The dataset's solution (code only) |
| `score` | `average_test_score`: fraction of tests the reference passed in NVIDIA's run |
| `domain` | `generic` or `algorithmic` |

## How it was built

{funnel}

Caveat: `score` comes from NVIDIA's execution environment. Re-running the references shows that
some tasks need files or third-party packages: filter by re-running the reference in your own
sandbox before training.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=25_000)
    parser.add_argument("--eval-size", type=int, default=500, help="for val and for test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    shard = download_shard(args.shard, args.data_dir / "raw" / "opencodeinstruct")
    rows = pq.ParquetFile(shard).metadata.num_rows
    tasks = load_tasks(shard)
    eligible = [t for t in tasks if t.score > args.min_score]
    sample = random.Random(args.seed).sample(eligible, args.size)
    splits = {
        "test": sample[: args.eval_size],
        "val": sample[args.eval_size : 2 * args.eval_size],
        "train": sample[2 * args.eval_size :],
    }

    out = args.data_dir / "tasks"
    (out / "data").mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        with (out / "data" / f"{name}.jsonl").open("w") as f:
            f.writelines(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in split)
    funnel = "\n".join(
        [
            f"- `{SHARD.format(index=args.shard)}` of {REPO}: {rows:,} rows",
            f"- tests all call one function defined in the reference: {len(tasks):,}",
            f"- dataset score > {args.min_score}: {len(eligible):,}",
            f"- random sample (seed {args.seed}): {args.size:,}",
            "- splits: " + ", ".join(f"{n} {len(s):,}" for n, s in splits.items()),
        ]
    )
    (out / "README.md").write_text(CARD.format(source=REPO, funnel=funnel))
    print(funnel)

    if args.push:
        api = HfApi()
        api.create_repo(TASKS_REPO, repo_type="dataset", exist_ok=True)
        api.upload_folder(repo_id=TASKS_REPO, repo_type="dataset", folder_path=out)
        print(f"pushed: https://huggingface.co/datasets/{TASKS_REPO}")


if __name__ == "__main__":
    main()
