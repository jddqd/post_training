"""Rollouts: GRPO groups of the agent on tasks, saved in the run's directory.

    uv run python scripts/rollout.py                                # configs/config.yaml
    uv run python scripts/rollout.py model=qwen3_5_2b tasks.n=10 rollout.group_size=4
    uv run python scripts/inspect_rollouts.py outputs/<date>/<time>

The run directory (outputs/<date>/<time>/) holds the resolved config (.hydra/), episodes.jsonl
(one line per episode, written group by group) and summary.json. Docker must be reachable.
"""

import json
import time
from pathlib import Path

import hydra
import mlx.core as mx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from rl_swe_agent.agent.mlx_model import load_policy
from rl_swe_agent.config import validate
from rl_swe_agent.rollout.runner import (
    group_stats,
    read_episodes,
    run_group,
    select_tasks,
    summarize,
    write_episodes,
)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config = validate(cfg)  # before the model and the tasks are loaded
    run = Path(HydraConfig.get().runtime.output_dir)
    mx.random.seed(config.seed)
    tasks = select_tasks(config.tasks)
    model, tokenizer = load_policy(config.model)
    # Keeps the weights in memory (no paging) when memory gets tight, as mlx-lm's generate does.
    mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
    print(f"{len(tasks)} tasks x {config.rollout.group_size} rollouts, {config.model.name} → {run}")

    start = time.time()
    for k, task in enumerate(tasks, 1):
        episodes = run_group(task, model, tokenizer, config)
        write_episodes(run / "episodes.jsonl", episodes)
        stats = group_stats(episodes)
        rewards = "".join(str(int(r)) for r in stats["rewards"])
        print(
            f"[{k}/{len(tasks)}] {task.function[:30]:30s} rewards {rewards} "
            f"tests {stats['mean_passed']:4.1f}/10 {'useful' if stats['useful'] else '      '} "
            f"{sum(e.seconds for e in episodes):5.0f} s (total {time.time() - start:5.0f} s)",
            flush=True,
        )
    summary = summarize(list(read_episodes(run / "episodes.jsonl")))
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
