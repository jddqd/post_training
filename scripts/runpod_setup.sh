#!/usr/bin/env bash
# Set up a fresh RunPod (Linux + NVIDIA GPU) pod for training and BFCL evaluation.
#
#   git clone https://github.com/jddqd/post_training.git && cd post_training
#   bash scripts/runpod_setup.sh
#
# Three separate environments, so that no dependency can change another one's versions:
#   .venv              training (torch, transformers, trl) + the `gpu` extra (flash-attn kernels)
#   eval/bfcl/.venv    BFCL client and grader (pinned bfcl-eval)
#   vllm (uv tool)     inference server, started by BFCL as the `vllm serve` command
set -euo pipefail

VLLM_VERSION=0.32.0 # pins torch 2.13 / transformers < 5.20: hence its own environment

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --locked --extra gpu
uv sync --locked --project eval/bfcl
uv tool install "vllm==${VLLM_VERSION}"

# Data: the published dataset (the same files `make data` produces, see its README / report.json).
uv run hf download jddqd/xlam-function-calling-60k-sft --repo-type dataset --local-dir data/xlam

# Checks: GPU visible, bf16 / tf32 capable, flash-attention kernels loadable, vllm on PATH.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run python - <<'EOF'
import torch
from transformers import AutoModelForCausalLM

assert torch.cuda.is_available(), "no CUDA GPU"
major, _ = torch.cuda.get_device_capability()
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name()}")
assert major >= 8, "Ampere or newer needed (bf16, tf32, flash-attention)"
# Downloads kernels-community/flash-attn2 through `kernels` (no flash-attn package installed).
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B-Base", dtype=torch.bfloat16, attn_implementation="flash_attention_2"
).cuda()
print("attention:", model.config._attn_implementation)
EOF
vllm --version
wc -l data/xlam/train.jsonl data/xlam/val.jsonl # 55 627 / 1 721

echo
echo "Ready. Before training: wandb login (once per pod)."
