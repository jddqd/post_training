# rl_swe_agent

RL (GRPO + LoRA in MLX, on the Mac) of a code agent working inside the
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) harness, from Qwen3.5.
Tasks: [jddqd/opencodeinstruct-rl-tasks](https://huggingface.co/datasets/jddqd/opencodeinstruct-rl-tasks)
(write one Python function, checked by 10 asserts).

## Layout: one package per stage

```
configs/          hyperparameters (Hydra): model/, sampling/, rollout/, tasks/
src/rl_swe_agent/
├── config.py     validates them (Pydantic) and injects them into agent.yaml
├── tasks/        data → tasks
│   ├── opencodeinstruct.py   OpenCodeInstruct rows as tasks, the published task set
│   └── workspace.py          a task as the agent's project: solution.py + test_public.py (3 asserts)
├── sandbox/      where model-written code runs (Docker)
│   ├── sandbox.py            one rollout's container, the workspace mounted at /workspace
│   └── verifier.py           the reward: all 10 asserts in a fresh container
├── agent/        the mini-swe-agent harness
│   ├── agent.yaml            prompts, observation format, sandbox options (mini-swe-agent's format)
│   ├── config.py             loads it, task variables for the templates
│   ├── toolcall.py           Qwen3.5's XML tool calls → mini-swe-agent actions, format errors
│   ├── model.py              mini-swe-agent's Model for a text policy; ScriptedModel for tests
│   └── mlx_model.py          the policy: Qwen3.5 in MLX, token in / token out (ids, mask, logprobs)
└── rollout/      from a task to saved trajectories
    ├── episode.py            DefaultAgent in a fresh sandbox, then the hidden tests
    │                         (the reward goes to a submitted solution only)
    └── runner.py             GRPO groups, episodes.jsonl, statistics (pass@1, useful groups...)
scripts/
├── build_dataset.py          builds and publishes the task set
├── try_models.py             single-turn baseline of Qwen3.5 (before the agent loop)
├── rollout.py                rollouts of a set of tasks (Hydra), saved in outputs/<date>/<time>/
└── inspect_rollouts.py       reads a run: groups, summary, one episode's conversation
```

Next stages: batched generation of a group, the GRPO loop (`training/`).

## Usage

```bash
uv sync
# Docker (OrbStack) must be reachable, with the image used by the sandbox and the verifier:
docker pull python:3.12-slim
uv run pytest          # the Docker and model tests are skipped when not available

uv run python scripts/build_dataset.py                  # data/tasks/data/{train,val,test}.jsonl
uv run python scripts/rollout.py model=qwen3_5_2b tasks.n=10 rollout.group_size=8
uv run python scripts/rollout.py --multirun model=qwen3_5_0_8b,qwen3_5_2b
uv run python scripts/inspect_rollouts.py outputs/<date>/<time>
```

Every value in `configs/` can be overridden on the command line (`group.key=value`, or a whole
group: `sampling=qwen`). The config is validated before the model is loaded: an unknown key or
an inconsistent value fails at once. Each run keeps its resolved config in `.hydra/`.
