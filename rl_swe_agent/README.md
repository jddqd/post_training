# rl_swe_agent

RL (GRPO + LoRA in MLX, on the Mac) of a code agent working inside the
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) harness, from Qwen3.5.
Tasks: [jddqd/opencodeinstruct-rl-tasks-mix70](https://huggingface.co/datasets/jddqd/opencodeinstruct-rl-tasks-mix70)
(write one Python function, checked by 10 asserts).

## Layout: one package per stage

```
configs/          hyperparameters (Hydra): model/, sampling/, rollout/, tasks/
src/rl_swe_agent/
├── config.py     validates them (Pydantic), loads agent.yaml and injects them into it
├── tasks/        data → tasks
│   ├── opencodeinstruct.py   OpenCodeInstruct rows as tasks, the published task set
│   └── workspace.py          a task as the agent's project: solution.py + test_public.py (3 asserts)
├── sandbox/      where model-written code runs (Docker)
│   ├── sandbox.py            one rollout's container, the workspace mounted at /workspace
│   └── verifier.py           the reward: all 10 asserts in a fresh container (same image and limits)
├── agent/        the mini-swe-agent harness
│   ├── agent.yaml            prompts, observation format, sandbox options (mini-swe-agent's format)
│   ├── toolcall.py           Qwen3.5's XML tool calls → mini-swe-agent actions, format errors
│   ├── model.py              mini-swe-agent's Model for a text policy; ScriptedModel for tests
│   ├── mlx_model.py          the policy: Qwen3.5 in MLX, token in / token out (ids, mask, logprobs)
│   └── generation.py         its tokens: alone, or batched with the other episodes (one MLX thread)
└── rollout/      from a task to saved trajectories
    ├── episode.py            DefaultAgent in a fresh sandbox, then the hidden tests
    │                         (the reward goes to a submitted solution only)
    ├── runner.py             rollout steps: dynamic batching (a thread per episode), jsonl, stats
    └── monitor.py            memory over time during a step (resources.jsonl)
scripts/
├── build_dataset.py          builds and publishes the task set
├── rollout.py                rollouts of a set of tasks (Hydra), saved in outputs/<date>/<time>/
├── inspect_rollouts.py       reads a run: groups, summary, one episode's conversation
└── plot_timeline.py          a run's timeline.html: episodes per worker over time, concurrency, memory
```

Next stage: the GRPO loop (`training/`).

## Usage

```bash
uv sync
# Docker (OrbStack) must be reachable, with the image used by the sandbox and the verifier:
docker pull python:3.12-slim
uv run pytest          # the Docker and model tests are skipped when not available

# the task set is downloaded from the Hub on first use (or rebuilt: scripts/build_dataset.py)
uv run python scripts/rollout.py model=qwen3_5_2b tasks.n=10                            # 1 task x 8 per batch
uv run python scripts/rollout.py model=qwen3_5_2b tasks.n=10 rollout.tasks_per_batch=2  # 2 tasks x 8
uv run python scripts/rollout.py --multirun model=qwen3_5_0_8b,qwen3_5_2b
uv run python scripts/inspect_rollouts.py outputs/<date>/<time>
uv run python scripts/plot_timeline.py outputs/<date>/<time>      # then open timeline.html
```

Every value in `configs/` can be overridden on the command line (`group.key=value`, or a whole
group: `sampling=qwen`). The config is validated before the model is loaded: an unknown key or
an inconsistent value fails at once. Each run keeps its resolved config in `.hydra/`.
