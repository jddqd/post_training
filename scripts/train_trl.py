"""Run an SFT experiment with TRL's SFTTrainer instead of our hand-written loop.

Same config, data, masking, run directory and checkpoint as scripts/train.py. Only the training
loop changes: TRL brings what transformers' Trainer offers on GPU (fused / 8-bit optimizers,
multi-GPU, length-grouped sampling, padding-free batches, resumable checkpoints).

Our token ids and labels are passed as is (a `labels` column is left untouched by TRL): TRL's own
assistant masking needs `{% generation %}` tags that Qwen3's template lacks, and would not apply
our BFCL template patches.

Usage:
    uv run python scripts/train_trl.py --config configs/<experiment>.yaml [section.key=value ...]
"""

import argparse
import io
import json
import sys
from pathlib import Path

import torch
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Submodule: `trl` exposes it through a lazy import, which mypy cannot follow.
from trl.trainer.sft_trainer import SFTTrainer

from post_training.config import load_config
from post_training.data.masking import CHATML, tokenize_file
from post_training.tracking import create_run_dir, save_run_metadata, start_wandb
from post_training.training.checkpoint import save_checkpoint
from post_training.training.trl_trainer import sft_config, to_dataset


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)  # follow long runs live (see train.py)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="section.key=value, e.g. train.lr=5e-5")
    args = parser.parse_args()

    try:
        config = load_config(args.config, args.overrides)
    except (ValidationError, ValueError) as error:
        sys.exit(f"Invalid config: {error}")
    run_dir = create_run_dir(config)
    save_run_metadata(run_dir, config, args.overrides)
    print(f"run dir: {run_dir}")
    set_seed(config.train.seed)

    tok = AutoTokenizer.from_pretrained(config.model.name)
    # fp32 master weights; with bf16=True the Trainer runs the forward under bf16 autocast.
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name, dtype=getattr(torch, config.model.dtype)
    )

    train_dataset = to_dataset(tokenize_file(config.data.train_file, tok))
    eval_dataset = None
    if config.data.val_file is not None:
        eval_dataset = to_dataset(tokenize_file(config.data.val_file, tok))
    print(f"train: {len(train_dataset)} | val: {len(eval_dataset) if eval_dataset else 0}")

    run = start_wandb(config, run_dir)  # TRL's wandb callback logs into this run
    trainer = SFTTrainer(
        model=model,
        args=sft_config(config, run_dir, has_eval=eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tok,
    )
    # Baseline: the untrained model's validation loss. Kept aside: train() resets the trainer
    # state, which would drop it from the log history.
    baseline = [{**trainer.evaluate(), "step": 0}] if eval_dataset is not None else []
    trainer.train()

    # TRL raw logs: train (loss, grad_norm, learning_rate, mean_token_accuracy...), eval_loss.
    with (run_dir / "trainer_log.jsonl").open("w") as f:
        for record in baseline + trainer.state.log_history:
            f.write(json.dumps(record) + "\n")

    if config.train.save_checkpoint:
        checkpoint = run_dir / "checkpoint"
        # `model` is the very object the trainer updated in place.
        save_checkpoint(model, tok, checkpoint, end_of_turn=CHATML.end_of_turn)
        print(f"checkpoint: {checkpoint}  (evaluate: eval/bfcl/run_bfcl.py --model-path ...)")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
