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
| `make data`      | Prepare xLAM → `data/xlam/` (~1 min)      |
| `make data-push` | Same, then publish to the HF repo         |
| `make clean`     | Remove caches and build artifacts         |

Tests that need a CUDA GPU are marked `@pytest.mark.gpu` and skipped in CI
(`pytest -m "not gpu"`). Long tests are marked `@pytest.mark.slow`.

Add dependencies with `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (tooling).

## Preparing data

```bash
make data   # = make bfcl-questions + uv run python scripts/prepare_data.py --config configs/data/xlam.yaml
```

Converts [xLAM-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) into
normalized conversations (tools in BFCL's function-doc format), validates, cleans,
deduplicates, **decontaminates against BFCL** (drops questions that are BFCL questions or share
8-grams covering ≥ 50% of the shorter text; `make bfcl-questions` exports them to
`data/bfcl/questions.jsonl`), shuffles tool order, filters by token length and splits train/val
(stratified by BFCL category). Writes to `data/xlam/`: `train.jsonl`, `val.jsonl`, `rejected.jsonl` (every
dropped row with its reason), `report.json` (funnel, stats, pinned dataset revision) and a
`README.md` dataset card. `make data-push` also publishes the folder to the private HF dataset
repo [`jddqd/xlam-function-calling-60k-sft`](https://huggingface.co/datasets/jddqd/xlam-function-calling-60k-sft)
(`hub_repo` in the config); commit the code first so the dataset references a clean commit.

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

## Training

Two entry points read the same experiment config and produce the same run directory and
checkpoint (`outputs/<name>-<timestamp>/checkpoint`, ready for BFCL):

```bash
# Our hand-written loop (src/post_training/training/loop.py): the reference
uv run python scripts/train.py --config configs/sft_qwen3_base_xlam.yaml

# TRL's SFTTrainer on the same tokens and labels, with every option set for one NVIDIA GPU
# (bf16, tf32, fused AdamW, length-grouped batches, resumable checkpoints, wandb)
uv run python scripts/train_trl.py --config configs/sft_qwen3_base_xlam_gpu.yaml
```

`configs/sft_qwen3_base_xlam_gpu.yaml` documents every option and why its value was chosen; it
needs CUDA (`tf32`, fused AdamW are checked at startup, before anything is loaded).

Both support gradient accumulation (normalized by token count), gradient checkpointing, bf16
mixed precision (fp32 master weights) and Weights & Biases logging (`wandb.enabled=true`, after
`wandb login`; `wandb.mode=offline` to log locally). `tests/training/test_trl_parity.py` checks
that TRL trains on our labels with the same loss as our loop.

## Evaluating with BFCL

The [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) is
used as a pinned dependency (`bfcl-eval`) in its own environment, `eval/bfcl/` (its dependencies
are heavy and must not leak into the training environment). Training prompts are rendered exactly
like BFCL's Qwen handler renders them; `eval/bfcl/tests` checks it against BFCL's own code.

```bash
# GPU machine: baseline of the untrained base model; BFCL starts vLLM itself
# (install the `vllm` CLI separately, e.g. `uv tool install vllm`)
uv run --project eval/bfcl python eval/bfcl/run_bfcl.py --name base --model-path Qwen/Qwen3-0.6B-Base

# Evaluate a fine-tuned checkpoint with the same handler (prompt format and parsing)
uv run --project eval/bfcl python eval/bfcl/run_bfcl.py --name sft-1 --model-path <checkpoint dir>

# Laptop (no CUDA): small subset against a local transformers server
uv run --project eval/bfcl python eval/bfcl/serve_local.py --model Qwen/Qwen3-0.6B-Base  # terminal 1
uv run --project eval/bfcl python eval/bfcl/run_bfcl.py --name smoke --limit 5 \
    --model-path Qwen/Qwen3-0.6B-Base --server-url http://localhost:1053/v1             # terminal 2
```

Each run writes BFCL's raw results, scores and a `summary.json` to `outputs/bfcl/<name>-<timestamp>/`.
`--model-path` is required: BFCL's registry name (`--model`) only selects the prompt format and
output parsing; without explicit weights it would silently evaluate the Instruct model. The run
fails if any generation errored (BFCL would count a failed generation as a correct "no call").
With `--limit`, only per-category accuracies are meaningful (BFCL's aggregates count the
categories that were not run as 0).

## Project layout

```
.
├── configs/              # experiment configs (YAML, see src/post_training/config.py)
├── eval/bfcl/            # BFCL evaluation (isolated uv project)
├── scripts/              # entrypoints and RunPod helpers
├── src/post_training/    # library code
├── tests/
├── data/                 # datasets (git-ignored)
└── outputs/              # checkpoints, logs (git-ignored)
```

## Running on RunPod

_To be documented once the training stack is in place._ CUDA-only dependencies will be
installed with `uv sync --extra gpu`.
