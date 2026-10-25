"""Run an SFT experiment described by a YAML config.

Usage:
    uv run python scripts/train.py --config configs/<experiment>.yaml [section.key=value ...]

Example:
    uv run python scripts/train.py --config configs/sft.yaml train.lr=5e-5 train.max_steps=20
"""

import argparse
import io
import random
import sys
from functools import partial
from pathlib import Path

import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from post_training.config import load_config
from post_training.data.masking import CHATML, TokenizedExample, tokenize_file
from post_training.tracking import (
    check_wandb_credentials,
    create_run_dir,
    save_history,
    save_run_metadata,
    start_wandb,
)
from post_training.training.checkpoint import save_checkpoint
from post_training.training.collator import collate
from post_training.training.loop import get_device, train


def main() -> None:
    # Redirected to a file, stdout is block-buffered: progress would only show up by chunks of
    # 8 KB (or at exit). Flush every line so that long runs can be followed live (tail -f).
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="section.key=value, e.g. train.lr=5e-5")
    args = parser.parse_args()

    # Validate everything before touching the GPU or downloading anything.
    try:
        config = load_config(args.config, args.overrides)
        check_wandb_credentials(config)
    except (ValidationError, ValueError, RuntimeError) as error:
        sys.exit(f"Cannot start: {error}")
    run_dir = create_run_dir(config)
    save_run_metadata(run_dir, config, args.overrides)
    print(f"run dir: {run_dir}")

    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)  # also seeds the DataLoader shuffling
    device = get_device()
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(config.model.name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        dtype=getattr(torch, config.model.dtype),
        attn_implementation=config.model.attn_implementation,
    )
    # Used as a plain nn.Module for training: transformers' annotation of `.to()` is wrong
    # under mypy. `hf_model` (same object) keeps the save/generation API for the checkpoint.
    model: torch.nn.Module = hf_model
    model.to(device)

    # Prompts rendered exactly like the BFCL evaluation (see post_training.data.prompt_format).
    examples: list[TokenizedExample] = tokenize_file(config.data.train_file, tok)
    print(f"train conversations: {len(examples)}")
    loader: DataLoader[TokenizedExample] = DataLoader(
        examples,  # type: ignore[arg-type]  # a list is a valid map-style dataset at runtime
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=partial(collate, pad_token_id=tok.pad_token_id),
    )

    val_loader: DataLoader[TokenizedExample] | None = None
    if config.data.val_file is not None:
        val_examples = tokenize_file(config.data.val_file, tok)
        print(f"val conversations: {len(val_examples)}")
        val_loader = DataLoader(
            val_examples,  # type: ignore[arg-type]
            batch_size=config.train.eval_batch_size or config.train.batch_size,
            shuffle=False,  # same batches at every evaluation: comparable numbers
            collate_fn=partial(collate, pad_token_id=tok.pad_token_id),
        )

    run = start_wandb(config, run_dir)
    result = train(
        model,
        loader,
        config.train,
        device,
        val_batches=val_loader,
        on_log=(lambda log: run.log(log, step=int(log["step"]))) if run is not None else None,
    )
    save_history(run_dir, result)
    print(f"train loss: {result.history[0].loss:.3f} -> {result.history[-1].loss:.3f}")
    if result.evals:
        print(f"val loss:   {result.evals[0].val_loss:.3f} -> {result.evals[-1].val_loss:.3f}")

    if config.train.save_checkpoint:
        checkpoint = run_dir / "checkpoint"
        save_checkpoint(hf_model, tok, checkpoint, end_of_turn=CHATML.end_of_turn)
        print(f"checkpoint: {checkpoint}  (evaluate: eval/bfcl/run_bfcl.py --model-path ...)")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
