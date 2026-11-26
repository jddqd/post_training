"""Build the RL task set from OpenCodeInstruct and publish it on the HF Hub.

Funnel: one shard → rows whose tests all call one function defined in the reference (to_task) →
two pools: score 1 (the reference passes all its tests) and --min-score < score < 1 → a random
sample of --size tasks, --imperfect-share of them from the second pool, in every split →
train / val / test.

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

from rl_swe_agent.tasks.opencodeinstruct import (
    REPO,
    SHARD,
    TASKS_REPO,
    CodeTask,
    download_shard,
    load_tasks,
)

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

{share_perfect:.0%} of each split are tasks whose reference passes all its tests (`score` 1), and
{share_imperfect:.0%} tasks whose reference fails some (`score` < 1): there the tests, or the
reference, are partly wrong, and even a correct solution may not pass them all.

{funnel}

Caveat: `score` comes from NVIDIA's execution environment. Re-running the references shows that
some tasks need files or third-party packages: filter by re-running the reference in your own
sandbox before training.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=0.2, help="floor of the score < 1 pool")
    parser.add_argument("--imperfect-share", type=float, default=0.3, help="tasks with score < 1")
    parser.add_argument("--size", type=int, default=25_000)
    parser.add_argument("--eval-size", type=int, default=500, help="for val and for test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    shard = download_shard(args.shard, args.data_dir / "raw" / "opencodeinstruct")
    rows = pq.ParquetFile(shard).metadata.num_rows
    tasks = load_tasks(shard)
    perfect = [t for t in tasks if t.score >= 1]
    imperfect = [t for t in tasks if args.min_score < t.score < 1]
    rng = random.Random(args.seed)
    n_imperfect = round(args.size * args.imperfect_share)
    pools = {
        "perfect": rng.sample(perfect, args.size - n_imperfect),
        "imperfect": rng.sample(imperfect, n_imperfect),
    }
    sizes = {"test": args.eval_size, "val": args.eval_size}
    sizes["train"] = args.size - 2 * args.eval_size
    splits: dict[str, list[CodeTask]] = {}
    for name, size in sizes.items():  # the same share of score < 1 tasks in every split
        k = round(size * args.imperfect_share) if name != "train" else len(pools["imperfect"])
        split = [pools["imperfect"].pop() for _ in range(k)]
        split += [pools["perfect"].pop() for _ in range(size - k)]
        rng.shuffle(split)
        splits[name] = split

    out = args.data_dir / "tasks"
    (out / "data").mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        with (out / "data" / f"{name}.jsonl").open("w") as f:
            f.writelines(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in split)
    funnel = "\n".join(
        [
            f"- `{SHARD.format(index=args.shard)}` of {REPO}: {rows:,} rows",
            f"- tests all call one function defined in the reference: {len(tasks):,}",
            f"- score 1: {len(perfect):,}; {args.min_score} < score < 1: {len(imperfect):,}",
            f"- random sample (seed {args.seed}): {args.size:,}, "
            f"{args.imperfect_share:.0%} with score < 1 in every split",
            "- splits: " + ", ".join(f"{n} {len(s):,}" for n, s in splits.items()),
        ]
    )
    (out / "README.md").write_text(
        CARD.format(
            source=REPO,
            funnel=funnel,
            share_perfect=1 - args.imperfect_share,
            share_imperfect=args.imperfect_share,
        )
    )
    print(funnel)

    if args.push:
        api = HfApi()
        api.create_repo(TASKS_REPO, repo_type="dataset", exist_ok=True)
        api.upload_folder(repo_id=TASKS_REPO, repo_type="dataset", folder_path=out)
        print(f"pushed: https://huggingface.co/datasets/{TASKS_REPO}")


if __name__ == "__main__":
    main()
