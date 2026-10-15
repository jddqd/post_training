# post-training

Pipeline to post-train LLMs (SFT / preference tuning) and evaluate them on the
[Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html).

Development and small experiments run locally on macOS (MPS); full training runs on a
CUDA GPU on [RunPod](https://www.runpod.io/).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Python 3.12 (uv installs it automatically from `.python-version`)

## Setup

```bash
make install          # uv sync + pre-commit hooks
cp .env.example .env  # then fill in HF_TOKEN, WANDB_API_KEY, ...
```

## Development

| Command          | What it does                              |
| ---------------- | ----------------------------------------- |
| `make fmt`       | Format and auto-fix with ruff             |
| `make lint`      | Lint and format check (no changes)        |
| `make typecheck` | Run mypy (strict)                         |
| `make test`      | Run pytest with coverage                  |
| `make check`     | lint + typecheck + test (same as CI)      |
| `make clean`     | Remove caches and build artifacts         |

Tests that need a CUDA GPU are marked `@pytest.mark.gpu` and skipped in CI
(`pytest -m "not gpu"`). Long tests are marked `@pytest.mark.slow`.

Add dependencies with `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (tooling).

## Project layout

```
.
├── configs/              # training / eval configs
├── docs/LEARNING.md      # roadmap + concepts learned (learning journal)
├── scripts/              # entrypoints and RunPod helpers
├── src/post_training/    # library code
├── tests/
├── data/                 # datasets (git-ignored)
└── outputs/              # checkpoints, logs (git-ignored)
```

## Running on RunPod

_To be documented once the training stack is in place._ CUDA-only dependencies will be
installed with `uv sync --extra gpu`.
