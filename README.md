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
| `make data`      | Prepare xLAM → `data/xlam/` (~30 s)       |
| `make clean`     | Remove caches and build artifacts         |

Tests that need a CUDA GPU are marked `@pytest.mark.gpu` and skipped in CI
(`pytest -m "not gpu"`). Long tests are marked `@pytest.mark.slow`.

Add dependencies with `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (tooling).

## Preparing data

```bash
make data   # = uv run python scripts/prepare_data.py --config configs/data/xlam.yaml
```

Converts [xLAM-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) into
normalized conversations (tool parameters converted to JSON Schema), validates, cleans,
deduplicates, shuffles tool order, filters by token length and splits train/val (stratified by
BFCL category). Writes to `data/xlam/`: `train.jsonl`, `val.jsonl`, `rejected.jsonl` (every
dropped row with its reason), `report.json` (funnel, stats, pinned dataset revision) and a
`README.md` dataset card, then pushes the folder to the private HF dataset repo
[`jddqd/xlam-function-calling-60k-sft`](https://huggingface.co/datasets/jddqd/xlam-function-calling-60k-sft)
(`hub_repo` in the config; remove it to skip the push).

## Running an experiment

Training reads a JSONL file of normalized conversations, one per line, in the Hugging Face chat
format (see `src/post_training/data/conversation.py`):

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "tools": []}
```

An experiment is a YAML file in `configs/`, validated with pydantic before anything is loaded.
Override any value from the command line with `section.key=value`:

```bash
uv run python scripts/train.py --config configs/<experiment>.yaml train.lr=5e-5 train.max_steps=20
```

Each run writes to `outputs/<name>-<timestamp>/`: the resolved `config.yaml`, `metadata.json`
(command, overrides, git commit, dirty flag) and `history.jsonl` (one line per step).

## Project layout

```
.
├── configs/              # experiment configs (YAML, see src/post_training/config.py)
├── scripts/              # entrypoints and RunPod helpers
├── src/post_training/    # library code
├── tests/
├── data/                 # datasets (git-ignored)
└── outputs/              # checkpoints, logs (git-ignored)
```

## Running on RunPod

_To be documented once the training stack is in place._ CUDA-only dependencies will be
installed with `uv sync --extra gpu`.
