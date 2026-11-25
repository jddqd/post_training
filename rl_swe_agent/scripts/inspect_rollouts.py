"""Read a rollout run: groups per task, the summary, or one episode's conversation.

uv run python scripts/inspect_rollouts.py outputs/<date>/<time>
uv run python scripts/inspect_rollouts.py outputs/<date>/<time> --task <id> --rollout 0
"""

import argparse
import json
from pathlib import Path

from rl_swe_agent.rollout.episode import Episode
from rl_swe_agent.rollout.runner import group_stats, groups, read_episodes, summarize


def show_groups(episodes: list[Episode]) -> None:
    print(f"{'task':10s} {'rewards':10s} {'tests':>6s} {'calls':>6s}  ends")
    for task_id, group in groups(episodes).items():
        stats = group_stats(group)
        rewards = "".join(str(int(r)) for r in stats["rewards"])
        calls = sum(e.model_calls for e in group) / len(group)
        ends = ",".join(sorted({e.exit_status for e in group}))
        useful = "useful" if stats["useful"] else ""
        print(
            f"{task_id[:10]:10s} {rewards:10s} {stats['mean_passed']:6.1f} {calls:6.1f}  "
            f"{ends} {useful}"
        )


def show_episode(episode: Episode) -> None:
    print(
        f"task {episode.task_id}: {episode.exit_status}, reward {episode.reward}, "
        f"hidden tests {episode.verdict.passed}/10, {episode.model_calls} calls, "
        f"{len(episode.ids)} tokens ({sum(episode.mask)} generated), {episode.seconds:.0f} s\n"
    )
    for message in episode.messages[1:]:  # the system prompt is the same everywhere
        print(f"===== {message['role']}\n{message['content']}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="run directory, or its episodes.jsonl")
    parser.add_argument("--task", help="prefix of a task id: print one of its episodes")
    parser.add_argument("--rollout", type=int, default=0, help="which episode of the group")
    args = parser.parse_args()
    path = args.run / "episodes.jsonl" if args.run.is_dir() else args.run
    episodes = list(read_episodes(path))
    if args.task:
        group = next(g for t, g in groups(episodes).items() if t.startswith(args.task))
        show_episode(group[args.rollout])
        return
    show_groups(episodes)
    print(json.dumps(summarize(episodes), indent=2))


if __name__ == "__main__":
    main()
