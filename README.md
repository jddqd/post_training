# post-training

Pipeline to post-train LLMs (SFT / preference tuning) and evaluate them on the
[Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html).

Code, tests and data preparation run locally (macOS); training and evaluation run on a CUDA
GPU on [RunPod](https://www.runpod.io/), with TRL's `SFTTrainer` and vLLM.

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
(command, overrides, git commit, dirty flag), `trainer_log.jsonl` (the Trainer's training and
validation logs) and `checkpoint/` (the final model, ready for BFCL).

## Training

Training uses TRL's `SFTTrainer` (`scripts/train.py`), on our own tokens and labels (loss on the
assistant turns only; see `src/post_training/data/masking.py`):

```bash
uv run python scripts/train.py --config configs/<experiment>.yaml [section.key=value ...]
```

| Config | Data | Batching | Use |
|---|---|---|---|
| `sft_qwen3_base_xlam_gpu_packed.yaml` | xLAM | packing: 8k-token sequences, ~15 conversations each | reference run |
| `sft_qwen3_base_xlam_toolace_gpu_packed.yaml` | xLAM + ToolACE | same | main run (adds text answers and multi-turn) |
| `sft_qwen3_base_xlam_gpu.yaml` | xLAM | length-grouped batches of 8 conversations | fallback when packing is unavailable |

All three target one NVIDIA GPU (bf16, tf32, fused AdamW, wandb, resumable checkpoints every 200
updates) and need CUDA: `tf32` and fused AdamW are checked at startup, before anything is loaded.
`configs/sft_qwen3_base_xlam_gpu.yaml` documents every option and why its value was chosen.
Packing keeps conversations whole (best-fit decreasing), isolates their attention with
flash-attention's varlen kernels and evaluates on unpacked rows, so the validation loss stays
comparable across configs. The packed compute settings (8k tokens per forward x 4 accumulation
steps, no gradient checkpointing: 9 340 tokens/s, 29 GiB on an A40) come from a throughput
benchmark; the packed configs run about 1.8x faster than the length-grouped one.

Gradient accumulation is normalized by token count, mixed precision keeps fp32 master weights,
and Weights & Biases logging is on with `wandb.enabled=true` (after `wandb login`;
`wandb.mode=offline` to log locally). `tests/training/test_trl_trainer.py` checks that the
Trainer receives our labels untouched and computes the SFT loss as defined (cross-entropy of
each trained token given the previous ones).

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
# (32 concurrent requests by default, batched by vLLM: --num-threads)

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
categories that were not run as 0). `run_bfcl.py` sets `VLLM_USE_FLASHINFER_SAMPLER=0`: vLLM's
FlashInfer sampler is compiled at startup, which needs `nvcc` and `ninja` (absent from PyTorch
images).

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

Any NVIDIA GPU from Ampere on (bf16, tf32, flash-attention) with 40 GB or more: the packed
configs peak at 29 GiB (A40 48 GB, A100). Plan ~50 GB of disk: resumable checkpoints hold the
optimizer state (~7 GB each, 2 kept), plus the final weights and caches. Pass the wandb key as
a RunPod secret (`WANDB_API_KEY={{ RUNPOD_SECRET_wandb_api_key }}` in the pod's environment):
the setup makes it visible to SSH sessions.

The GitHub repository is private: send the code from your machine as a git bundle (full
history, so runs record the right commit; no credentials on the pod):

```bash
# On your machine
git bundle create /tmp/post_training.bundle main
scp -P <port> /tmp/post_training.bundle root@<pod ip>:/workspace/
# On the pod
cd /workspace && git clone -b main post_training.bundle post_training && cd post_training
bash scripts/runpod_setup.sh  # environments, vLLM, data from the HF Hub, sanity checks

# Train (TRL + packing), then evaluate the checkpoint on BFCL
uv run python scripts/train.py --config configs/sft_qwen3_base_xlam_gpu_packed.yaml
# same on xLAM + ToolACE
uv run python scripts/train.py --config configs/sft_qwen3_base_xlam_toolace_gpu_packed.yaml
uv run --project eval/bfcl python eval/bfcl/run_bfcl.py --name sft \
    --model-path outputs/<run>/checkpoint

# Pod stopped mid-run? Resume from the last checkpoint (every trl.save_steps updates): same
# run directory, data order and wandb run
uv run python scripts/train.py --resume outputs/<run>
```

Run long jobs inside `tmux` so they survive a closed SSH session. The setup also runs the GPU
test `tests/training/test_gpu_packing.py`: packed conversations must not attend to each other
(it fails if the flash-attention varlen kernels are not used).

Three environments, so that no dependency can change another's versions:

| Environment | Contents |
|---|---|
| `.venv` (`uv sync --extra gpu`) | training; the `gpu` extra adds `kernels`, through which transformers loads prebuilt flash-attention kernels from the Hub (no flash-attn build) |
| `eval/bfcl/.venv` | BFCL client and grader (pinned `bfcl-eval`) |
| `uv tool install vllm` | the inference server BFCL starts (`vllm serve`); it pins its own torch / transformers |
