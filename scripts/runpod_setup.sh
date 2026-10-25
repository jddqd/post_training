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

# uv and the tools it installs (vllm) live in ~/.local/bin: on PATH now, and in new shells
# (BFCL starts `vllm serve` from the shell run_bfcl.py is launched in).
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv tool update-shell

# Pod environment variables (e.g. WANDB_API_KEY from a RunPod secret) are given to the
# container's main process, not to SSH sessions: every shell re-reads it from there (the value
# itself is never written to disk). Login shells (tmux "bash -lc") read ~/.profile.
WANDB_LINE='export WANDB_API_KEY="$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^WANDB_API_KEY=//p")"  # RunPod secret, not inherited by SSH sessions'
if tr "\0" "\n" < /proc/1/environ | grep -q "^WANDB_API_KEY="; then
    for rc in ~/.bashrc ~/.profile; do
        grep -q "RunPod secret, not inherited" "$rc" 2>/dev/null || echo "$WANDB_LINE" >> "$rc"
    done
fi

uv sync --locked --extra gpu
uv sync --locked --project eval/bfcl
uv tool install "vllm==${VLLM_VERSION}"

# Data: the published (public) datasets, the same files `make data` produces (see their
# README / report.json).
uv run hf download jddqd/xlam-function-calling-60k-sft --repo-type dataset --local-dir data/xlam
uv run hf download jddqd/xlam-toolace-sft --repo-type dataset --local-dir data/xlam_toolace

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
# Packed conversations must not attend to each other (flash-attention varlen kernels):
# otherwise packing silently trains on cross-conversation context.
uv run pytest -m gpu --no-cov -q
wc -l data/xlam/{train,val}.jsonl data/xlam_toolace/{train,val}.jsonl # 55 627 / 1 721, 65 973 / 2 041
df -h . # checkpoints: ~7 GB each with the optimizer state (2 kept) + final weights + caches

echo
echo "Ready. Before training: wandb login (once per pod), then open a new shell (PATH)."
echo "Long runs: inside tmux (survives a closed SSH session); resume a stopped run with"
echo "  uv run python scripts/train_trl.py --resume outputs/<run dir>"
