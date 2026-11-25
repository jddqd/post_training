"""Rollout steps: the GRPO groups collected for one update of the weights, saved as JSON lines.

    tasks = select_tasks(config.tasks)
    groups = run_step(tasks[:4], model, tokenizer, config)  # a group per task

Dynamic batching: the episodes of the step (tasks x group_size) wait in a queue and at most
max_in_flight run at once, each in its thread and fresh sandbox, their turns generated as one
batch. When an episode ends, the next one starts: the GPU stays busy until the end of the step,
where at worst the last long episodes run alone. Every token comes from the same weights
(on-policy); each records their version, for a later off-policy variant.
"""

import json
import random
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any

from rl_swe_agent.agent.mlx_model import MLXModel, make_generator
from rl_swe_agent.agent.model import TextModel
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


def run_episodes(
    tasks: list[CodeTask], policy: Callable[[], TextModel], config: Config
) -> list[list[Episode]]:
    """One group of group_size episodes per task, max_in_flight at once; `policy()` makes the
    model of one episode."""
    mini = mini_config(config)
    jobs = [task for task in tasks for _ in range(config.rollout.group_size)]

    def episode(task: CodeTask) -> Episode:
        return run_episode(task, policy(), mini)

    if config.rollout.max_in_flight == 1:
        episodes = [episode(task) for task in jobs]
    else:  # a worker thread takes the next queued episode when its own ends
        with ThreadPoolExecutor(config.rollout.max_in_flight, "episode") as pool:
            episodes = list(pool.map(episode, jobs))
    size = config.rollout.group_size
    return [episodes[i : i + size] for i in range(0, len(episodes), size)]


def run_step(
    tasks: list[CodeTask],
    model: Any,
    tokenizer: Any,
    config: Config,
    version: int = 0,
    batch_log: list[tuple[float, int]] | None = None,
) -> list[list[Episode]]:
    """The groups of `tasks` sampled by the MLX policy, whose weights are at `version`.
    `batch_log` receives (time, sequences) for each batched decoding step."""
    templates = mini_config(config)["model"]
    in_flight = config.rollout.max_in_flight
    with make_generator(
        model, tokenizer, config.sampling, in_flight, version, config.rollout.prefill_batch_size
    ) as generator:
        groups = run_episodes(
            tasks, lambda: MLXModel(generator, tokenizer, templates, config.rollout), config
        )
    if batch_log is not None:
        batch_log += getattr(generator, "steps", [])
    return groups


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


def step_stats(episodes: list[Episode], seconds: float, max_in_flight: int) -> dict[str, float]:
    """How busy the step kept its slots: occupancy = episode time / (slots x step time), and the
    tail: from the last episode's start to the end, when fewer and fewer episodes run."""
    first = min(e.started for e in episodes)
    busy = sum(e.seconds for e in episodes)
    return {
        "seconds": seconds,
        "occupancy": busy / (max_in_flight * seconds),
        "tail_seconds": first + seconds - max(e.started for e in episodes),
    }
