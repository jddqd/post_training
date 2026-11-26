"""Rollouts: GRPO groups of the agent on tasks, saved in the run's directory.

    uv run python scripts/rollout.py                                # configs/config.yaml
    uv run python scripts/rollout.py model=qwen3_5_2b tasks.n=8 rollout.tasks_per_step=4
    uv run python scripts/inspect_rollouts.py outputs/<date>/<time>
    uv run python scripts/plot_timeline.py outputs/<date>/<time>    # timeline.html

The run directory (outputs/<date>/<time>/) holds the resolved config (.hydra/), episodes.jsonl
(one line per episode, written group by group, with its timeline), resources.jsonl (memory over
time), batch.jsonl (the size of each batched decoding step), memory_trace.jsonl (MLX's memory
and what each batch step processed) and summary.json. Docker must be
reachable.
"""

import json
import logging
import time
from pathlib import Path

import hydra
import mlx.core as mx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from rl_swe_agent.agent.mlx_model import load_policy
from rl_swe_agent.config import validate
from rl_swe_agent.rollout.monitor import ResourceMonitor
from rl_swe_agent.rollout.runner import (
    group_stats,
    read_episodes,
    run_step,
    select_tasks,
    step_stats,
    summarize,
    write_episodes,
)


def peak_memory(trace: Path) -> float:
    """GB: the highest MLX memory of a batch step (the trace resets the peak at each step; there
    is none when generating one sequence at a time)."""
    if not trace.exists():
        return mx.get_peak_memory() / 1e9
    with trace.open() as f:
        return max((json.loads(line)["peak_gb"] for line in f), default=0.0)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config = validate(cfg)  # before the model and the tasks are loaded
    logging.getLogger("minisweagent").setLevel(logging.WARNING)  # one line per container
    run = Path(HydraConfig.get().runtime.output_dir)
    mx.random.seed(config.seed)
    tasks = select_tasks(config.tasks)
    model, tokenizer = load_policy(config.model)
    print(f"{len(tasks)} tasks x {config.rollout.group_size} rollouts, {config.model.name} → {run}")

    size = config.rollout.tasks_per_step
    for step, k in enumerate(range(0, len(tasks), size)):
        batch = tasks[k : k + size]
        start = time.time()
        batch_log: list[tuple[float, int]] = []
        with ResourceMonitor() as monitor:
            groups = run_step(
                batch,
                model,
                tokenizer,
                config,
                batch_log=batch_log,
                trace=run / "memory_trace.jsonl",
            )
        with (run / "batch.jsonl").open("a") as f:
            f.writelines(
                json.dumps({"step": step, "time": t, "sequences": n}) + "\n" for t, n in batch_log
            )
        with (run / "resources.jsonl").open("a") as f:
            f.writelines(json.dumps({"step": step, **sample}) + "\n" for sample in monitor.samples)
        timeline = step_stats(
            [e for g in groups for e in g], time.time() - start, config.rollout.max_in_flight
        )
        for i, (task, episodes) in enumerate(zip(batch, groups, strict=True), k + 1):
            write_episodes(run / "episodes.jsonl", episodes)
            stats = group_stats(episodes)
            rewards = "".join(str(int(r)) for r in stats["rewards"])
            print(
                f"[{i}/{len(tasks)}] {task.function[:30]:30s} rewards {rewards} "
                f"tests {stats['mean_passed']:4.1f}/10 {'useful' if stats['useful'] else ''}",
                flush=True,
            )
        print(
            f"  step {step}: {timeline['seconds']:.0f} s, slots busy {timeline['occupancy']:.0%}, "
            f"tail {timeline['tail_seconds']:.0f} s, "
            f"peak MLX memory {peak_memory(run / 'memory_trace.jsonl'):.1f} GB",
            flush=True,
        )
    summary = summarize(list(read_episodes(run / "episodes.jsonl")))
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
