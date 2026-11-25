"""GRPO groups: the rollouts of a task, saved as one JSON line per episode, and their statistics.

    tasks = select_tasks(config.tasks)
    episodes = run_group(task, model, tokenizer, config)  # config.rollout.group_size episodes

v1 runs the episodes one after the other (each in its own fresh sandbox).
"""

import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from statistics import mean
from typing import Any

from rl_swe_agent.agent.mlx_model import MLXModel
from rl_swe_agent.config import Config, TasksConfig, mini_config
from rl_swe_agent.rollout.episode import Episode, run_episode
from rl_swe_agent.tasks.opencodeinstruct import CodeTask, download_split, read_tasks


def select_tasks(config: TasksConfig) -> list[CodeTask]:
    """`n` tasks of the split, drawn with `seed` (the split is downloaded if missing)."""
    if not config.path.exists():
        download_split(config.split, config.root)
    tasks = read_tasks(config.path)
    if config.n is None or config.n >= len(tasks):
        return tasks
    return random.Random(config.seed).sample(tasks, config.n)


def run_group(task: CodeTask, model: Any, tokenizer: Any, config: Config) -> list[Episode]:
    if config.rollout.tasks_per_batch != 1:
        raise NotImplementedError("several tasks per batch need batched generation")
    mini = mini_config(config)
    episodes = []
    for _ in range(config.rollout.group_size):
        policy = MLXModel(model, tokenizer, mini["model"], config.sampling, config.rollout)
        episodes.append(run_episode(task, policy, mini))
    return episodes


def write_episodes(path: Path, episodes: Iterable[Episode]) -> None:
    """Appends: a run interrupted midway keeps its finished groups."""
    with path.open("a") as f:
        for episode in episodes:
            f.write(json.dumps(episode.to_json()) + "\n")


def read_episodes(path: Path) -> Iterator[Episode]:
    with path.open() as f:
        for line in f:
            yield Episode.from_json(json.loads(line))


def groups(episodes: Iterable[Episode]) -> dict[str, list[Episode]]:
    by_task: dict[str, list[Episode]] = {}
    for episode in episodes:
        by_task.setdefault(episode.task_id, []).append(episode)
    return by_task


def group_stats(group: list[Episode]) -> dict[str, Any]:
    rewards = [e.reward for e in group]
    return {
        "rewards": rewards,
        "mean_reward": mean(rewards),
        # GRPO learns only from groups whose rewards differ: an advantage is reward - group mean.
        "useful": len(set(rewards)) > 1,
        "mean_passed": mean(e.verdict.passed for e in group),
    }


def summarize(episodes: list[Episode]) -> dict[str, Any]:
    """pass@1, pass@k, share of useful groups, turns, ends, format errors, tokens, time."""
    by_task = groups(episodes)
    stats = [group_stats(g) for g in by_task.values()]
    generated = [sum(e.mask) for e in episodes]
    return {
        "tasks": len(by_task),
        "episodes": len(episodes),
        "pass@1": mean(e.reward for e in episodes),
        "pass@k": mean(any(r > 0 for r in s["rewards"]) for s in stats),
        "useful_groups": mean(s["useful"] for s in stats),
        "mean_passed_tests": mean(e.verdict.passed for e in episodes),
        "mean_model_calls": mean(e.model_calls for e in episodes),
        "exit_status": dict(Counter(e.exit_status for e in episodes)),
        "format_errors_per_episode": mean(e.format_errors for e in episodes),
        "mean_generated_tokens": mean(generated),
        "mean_episode_tokens": mean(len(e.ids) for e in episodes),
        "mean_seconds": mean(e.seconds for e in episodes),
    }
