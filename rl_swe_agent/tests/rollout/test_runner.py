import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rl_swe_agent.config import TasksConfig
from rl_swe_agent.rollout.episode import Episode
from rl_swe_agent.rollout.runner import (
    group_stats,
    read_episodes,
    select_tasks,
    summarize,
    write_episodes,
)
from rl_swe_agent.sandbox.verifier import Verdict
from rl_swe_agent.tasks.opencodeinstruct import CodeTask


def episode(task_id: str, passed: int, exit_status: str = "Submitted") -> Episode:
    return Episode(
        task_id=task_id,
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "hm"},
            {"role": "user", "content": "Format error: No tool calls found."},
        ],
        exit_status=exit_status,
        verdict=Verdict(["pass"] * passed + ["fail"] * (10 - passed)),
        model_calls=3,
        seconds=2.0,
        ids=[1, 2, 3, 4],
        mask=[0, 1, 1, 0],
        logprobs=[0.0, -0.5, -0.1, 0.0],
    )


def test_episodes_round_trip_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    written = [episode("a", 10), episode("a", 3, "LimitsExceeded")]
    write_episodes(path, written[:1])
    write_episodes(path, written[1:])  # appended, group after group
    assert list(read_episodes(path)) == written
    line = json.loads(path.read_text().splitlines()[0])
    assert line["reward"] == 1.0
    assert line["passed"] == 10


def test_a_group_is_useful_only_if_its_rewards_differ() -> None:
    assert group_stats([episode("a", 10), episode("a", 2)])["useful"]
    assert not group_stats([episode("a", 2), episode("a", 5)])["useful"]  # all 0
    assert not group_stats([episode("a", 10), episode("a", 10)])["useful"]  # all 1


def test_summary() -> None:
    unsubmitted = episode("b", 10, "LimitsExceeded")  # correct, but earns nothing
    episodes = [episode("a", 10), episode("a", 0), unsubmitted, episode("b", 4)]
    summary = summarize(episodes)
    assert summary["tasks"] == 2
    assert summary["pass@1"] == 0.25
    assert summary["pass@k"] == 0.5
    assert summary["useful_groups"] == 0.5
    assert summary["exit_status"] == {"Submitted": 3, "LimitsExceeded": 1}
    assert summary["format_errors_per_episode"] == 1
    assert summary["mean_generated_tokens"] == 2


@pytest.fixture
def split(tmp_path: Path) -> Path:
    task = CodeTask("t", "p", "r", ["assert f() == 1"] * 10, "f", "def f():", 1.0, "generic")
    rows = [json.dumps(asdict(task) | {"id": f"t{i}"}) for i in range(20)]
    (tmp_path / "val.jsonl").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_tasks_are_drawn_with_a_seed(split: Path) -> None:
    config = TasksConfig(data_dir=split, split="val", n=5, seed=1)
    drawn = [t.id for t in select_tasks(config)]
    assert len(drawn) == 5
    assert drawn == [t.id for t in select_tasks(config)]
    assert len(select_tasks(TasksConfig(data_dir=split, split="val"))) == 20
