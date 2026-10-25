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
| `make data`      | Prepare xLAM + ToolACE, combine (~2 min)  |
| `make data-push` | Same, then publish to the HF Hub          |
| `make clean`     | Remove caches and build artifacts         |

Tests that need a CUDA GPU are marked `@pytest.mark.gpu` and skipped in CI
(`pytest -m "not gpu"`). Long tests are marked `@pytest.mark.slow`.

Add dependencies with `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (tooling).

## Preparing data

```bash
make data   # = make bfcl-questions + prepare xLAM, prepare ToolACE, combine them
```

Each source is converted by its own converter (`src/post_training/data/{xlam,toolace}.py`) into
normalized conversations (tools in BFCL's function-doc format); the rest of the pipeline is
shared (`scripts/prepare_data.py`, `configs/data/<source>.yaml`): validation of every call
against the tools, cleaning, deduplication, **decontamination against BFCL** (drops questions
that are BFCL questions or share 8-grams covering ≥ 50% of the shorter text;
`make bfcl-questions` exports them to `data/bfcl/questions.jsonl`), tool order shuffling, a
token-length filter and a train/val split stratified by category. Each output folder holds
`train.jsonl`, `val.jsonl`, `rejected.jsonl` (every dropped row with its reason), `report.json`
(funnel, stats, pinned dataset revision) and a `README.md` dataset card.

| Dataset | Source | Content | HF repo |
|---|---|---|---|
| `data/xlam/` | [xLAM-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | single-turn tool calls | [`jddqd/xlam-function-calling-60k-sft`](https://huggingface.co/datasets/jddqd/xlam-function-calling-60k-sft) |
| `data/toolace/` | [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | + text answers (no function applies, missing parameter), multi-turn, answers from tool responses | [`jddqd/toolace-sft`](https://huggingface.co/datasets/jddqd/toolace-sft) |
| `data/xlam_toolace/` | both (`scripts/combine_datasets.py`) | union, each source's split kept, cross-source dedup | [`jddqd/xlam-toolace-sft`](https://huggingface.co/datasets/jddqd/xlam-toolace-sft) |

`make data-push` also publishes the folders to the HF Hub (`hub_repo` in each config); commit
the code first so the datasets reference a clean commit.

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

# Same experiment with packing: ~15 conversations per 8 192-token sequence (needs flash-attn)
uv run python scripts/train_trl.py --config configs/sft_qwen3_base_xlam_gpu_packed.yaml
```

`configs/sft_qwen3_base_xlam_gpu.yaml` documents every option and why its value was chosen; it
needs CUDA (`tf32`, fused AdamW are checked at startup, before anything is loaded). The packed
variant keeps conversations whole (best-fit decreasing), isolates their attention with
flash-attention's varlen kernels and evaluates on unpacked rows, so its validation loss is
comparable to the unpacked run's.

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

Any NVIDIA GPU from Ampere on (bf16, tf32, flash-attention): an A40 (48 GB) or an A100 fits
the GPU configs; 24 GB is tight with 16k tokens per forward. Plan ~50 GB of disk: resumable
checkpoints hold the optimizer state (~7 GB each, 2 kept), plus the final weights and caches.

```bash
git clone https://github.com/jddqd/post_training.git && cd post_training
bash scripts/runpod_setup.sh  # environments, vLLM, data from the HF Hub, sanity checks
wandb login

# Train (TRL + packing, or our loop), then evaluate the checkpoint on BFCL
uv run python scripts/train_trl.py --config configs/sft_qwen3_base_xlam_gpu_packed.yaml
# same on xLAM + ToolACE (private dataset: the setup asks for a Hugging Face token)
uv run python scripts/train_trl.py --config configs/sft_qwen3_base_xlam_toolace_gpu_packed.yaml
uv run --project eval/bfcl python eval/bfcl/run_bfcl.py --name sft \
    --model-path outputs/<run>/checkpoint
```

Three environments, so that no dependency can change another's versions:

| Environment | Contents |
|---|---|
| `.venv` (`uv sync --extra gpu`) | training; the `gpu` extra adds `kernels`, through which transformers loads prebuilt flash-attention kernels from the Hub (no flash-attn build) |
| `eval/bfcl/.venv` | BFCL client and grader (pinned `bfcl-eval`) |
| `uv tool install vllm` | the inference server BFCL starts (`vllm serve`); it pins its own torch / transformers |
