# rl_swe_agent

RL (GRPO + LoRA in MLX, on the Mac) of a code agent working inside the
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) harness, from Qwen3.5.
Tasks: [jddqd/opencodeinstruct-rl-tasks](https://huggingface.co/datasets/jddqd/opencodeinstruct-rl-tasks)
(write one Python function, checked by 10 asserts).

## Layout: one package per stage

```
src/rl_swe_agent/
├── tasks/        data → tasks
│   ├── opencodeinstruct.py   OpenCodeInstruct rows as tasks, the published task set
│   └── workspace.py          a task as the agent's project: solution.py + test_public.py (3 asserts)
├── sandbox/      where model-written code runs (Docker)
│   ├── sandbox.py            one rollout's container, the workspace mounted at /workspace
│   └── verifier.py           the reward: all 10 asserts in a fresh container
├── agent/        the mini-swe-agent harness
│   ├── agent.yaml            prompts, observation format, sandbox options, limits
│   ├── config.py             loads it, task variables for the templates
│   ├── toolcall.py           Qwen3.5's XML tool calls → mini-swe-agent actions, format errors
│   └── model.py              mini-swe-agent's Model for a text policy; ScriptedModel for tests
└── rollout/      one episode, from the sandbox to the reward
    └── episode.py            DefaultAgent in a fresh sandbox, then the hidden tests
                              (the reward goes to a submitted solution only)
scripts/
├── build_dataset.py          builds and publishes the task set
└── try_models.py             single-turn baseline of Qwen3.5 (before the agent loop)
```

Next stages: the MLX model (`agent/`), the group runner and saved trajectories
(`rollout/`), the GRPO loop (`training/`).

## Usage

```bash
uv sync
# Docker (OrbStack) must be reachable, with the image used by the sandbox and the verifier:
docker pull python:3.12-slim
uv run pytest          # the Docker tests are skipped when Docker is not available
```
