"""Read a rollout run: groups per task, the summary, or one episode's conversation.

uv run python scripts/inspect_rollouts.py outputs/<date>/<time>
uv run python scripts/inspect_rollouts.py outputs/<date>/<time> --task <id> --rollout 0
uv run python scripts/inspect_rollouts.py outputs/<date>/<time> --sample 4   # trajectories/*.md
"""

import argparse
import json
import random
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


def to_markdown(episode: Episode, index: int) -> str:
    """One episode as a readable page: the task, each response and what the sandbox answered."""
    verdict = episode.verdict
    lines = [
        f"# {episode.task_id} — rollout {index}",
        "",
        f"- end: **{episode.exit_status}**, reward **{episode.reward:g}**",
        f"- hidden tests: {verdict.passed}/{len(verdict.statuses)} ({', '.join(verdict.statuses)})"
        + (f", load error: `{verdict.load_error}`" if verdict.load_error else ""),
        f"- {episode.model_calls} model calls, {episode.format_errors} format errors, "
        f"{len(episode.ids)} tokens ({sum(episode.mask)} generated), {episode.seconds:.0f} s",
        "",
    ]
    turn = 0
    for message in episode.messages[1:]:  # the system prompt is the same everywhere
        role, content = message["role"], message["content"]
        if role == "assistant":
            turn += 1
            title = f"## Turn {turn} — model"
        elif turn == 0:
            title = "## Task"
        elif message.get("extra", {}).get("interrupt_type") == "FormatError":
            title = "### Format error"
        else:
            title = "### Observation"
        fence = "````" if "```" in content else "```"
        lines += [title, "", fence, content.rstrip(), fence, ""]
    return "\n".join(lines)


def write_sample(episodes: list[Episode], out: Path, n: int, seed: int) -> None:
    by_task = groups(episodes)
    pairs = [(task_id, i) for task_id, group in by_task.items() for i in range(len(group))]
    out.mkdir(exist_ok=True)
    for task_id, i in random.Random(seed).sample(pairs, n):
        episode = by_task[task_id][i]
        path = out / f"{task_id[:10]}_{i}_{episode.exit_status}_r{episode.reward:g}.md"
        path.write_text(to_markdown(episode, i))
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="run directory, or its episodes.jsonl")
    parser.add_argument("--task", help="prefix of a task id: print one of its episodes")
    parser.add_argument("--rollout", type=int, default=0, help="which episode of the group")
    parser.add_argument("--sample", type=int, help="write this many random episodes as markdown")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = args.run / "episodes.jsonl" if args.run.is_dir() else args.run
    episodes = list(read_episodes(path))
    if args.sample:
        write_sample(episodes, path.parent / "trajectories", args.sample, args.seed)
        return
    if args.task:
        group = next(g for t, g in groups(episodes).items() if t.startswith(args.task))
        show_episode(group[args.rollout])
        return
    show_groups(episodes)
    print(json.dumps(summarize(episodes), indent=2))


if __name__ == "__main__":
    main()
