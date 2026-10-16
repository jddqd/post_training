"""Run an SFT experiment described by a YAML config.

Usage:
    uv run python scripts/train.py --config configs/<experiment>.yaml [section.key=value ...]

Example:
    uv run python scripts/train.py --config configs/sft.yaml train.lr=5e-5 train.max_steps=20
"""

import argparse
import random
import sys
from functools import partial
from pathlib import Path

import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from post_training.config import load_config
from post_training.data.conversation import load_jsonl
from post_training.data.masking import TokenizedExample, tokenize_with_labels
from post_training.tracking import create_run_dir, save_history, save_run_metadata
from post_training.training.collator import collate
from post_training.training.loop import get_device, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="section.key=value, e.g. train.lr=5e-5")
    args = parser.parse_args()

    # Validate everything before touching the GPU or downloading anything.
    try:
        config = load_config(args.config, args.overrides)
    except (ValidationError, ValueError) as error:
        sys.exit(f"Invalid config: {error}")
    run_dir = create_run_dir(config)
    save_run_metadata(run_dir, config, args.overrides)
    print(f"run dir: {run_dir}")

    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)  # also seeds the DataLoader shuffling
    device = get_device()
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(config.model.name)
    # Annotated as nn.Module: transformers' own annotation of `.to()` is wrong under mypy.
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        config.model.name, dtype=getattr(torch, config.model.dtype)
    )
    model.to(device)

    conversations = load_jsonl(config.data.train_file)
    print(f"train conversations: {len(conversations)}")
    examples: list[TokenizedExample] = [
        tokenize_with_labels(c.messages, tok, tools=c.tools) for c in conversations
    ]
    loader: DataLoader[TokenizedExample] = DataLoader(
        examples,  # type: ignore[arg-type]  # a list is a valid map-style dataset at runtime
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=partial(collate, pad_token_id=tok.pad_token_id),
    )

    history = train(model, loader, config.train, device)
    save_history(run_dir, history)
    print(f"loss: {history[0].loss:.3f} -> {history[-1].loss:.3f}")


if __name__ == "__main__":
    main()
