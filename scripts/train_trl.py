"""Run an SFT experiment with TRL's SFTTrainer instead of our hand-written loop.

Same config, data, masking, run directory and checkpoint as scripts/train.py. Only the training
loop changes: TRL brings what transformers' Trainer offers on GPU (fused / 8-bit optimizers,
multi-GPU, length-grouped sampling, padding-free batches, resumable checkpoints).

Our token ids and labels are passed as is (a `labels` column is left untouched by TRL): TRL's own
assistant masking needs `{% generation %}` tags that Qwen3's template lacks, and would not apply
our BFCL template patches.

Usage:
    uv run python scripts/train_trl.py --config configs/<experiment>.yaml [section.key=value ...]

    # Resume an interrupted run (e.g. a stopped pod) from its last checkpoint (needs
    # trl.save_steps): same run directory, config, data order and wandb run.
    uv run python scripts/train_trl.py --resume outputs/<run dir>
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
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from post_training.config import ExperimentConfig, load_config, load_yaml_config
from post_training.data.masking import CHATML, tokenize_file
from post_training.tracking import (
    check_wandb_credentials,
    create_run_dir,
    record_resume,
    save_run_metadata,
    start_wandb,
)
from post_training.training.checkpoint import save_checkpoint
from post_training.training.trl_trainer import sft_config, to_dataset


def start_run(args: argparse.Namespace) -> tuple[ExperimentConfig, Path, SFTConfig]:
    """Config, run directory and training arguments, for a new run or a resumed one.

    Everything that can fail fails here, in seconds, before loading or tokenizing anything.
    """
    if args.resume and args.overrides:
        sys.exit("--resume continues a run with its saved config: no overrides")
    try:
        if args.resume:
            # The resolved config saved at the start of the run (file + overrides + defaults).
            config = load_yaml_config(ExperimentConfig, args.resume / "config.yaml")
        else:
            config = load_config(args.config, args.overrides)
        check_wandb_credentials(config)
    except (ValidationError, ValueError, RuntimeError, FileNotFoundError) as error:
        sys.exit(f"Cannot start: {error}")
    if args.resume and not list((args.resume / "trainer").glob("checkpoint-*")):
        sys.exit(f"No checkpoint to resume from in {args.resume / 'trainer'} (trl.save_steps?)")

    run_dir: Path = args.resume or create_run_dir(config)
    # SFTConfig checks the hardware (e.g. tf32 needs an NVIDIA Ampere GPU, adamw_torch_fused
    # needs CUDA).
    try:
        training_args = sft_config(config, run_dir, has_eval=config.data.val_file is not None)
    except ValueError as error:
        if not args.resume:
            run_dir.rmdir()
        sys.exit(f"Invalid training arguments for this machine: {error}")
    if args.resume:
        training_args.eval_on_start = False  # the baseline evaluation was done at step 0
        record_resume(run_dir)
    else:
        save_run_metadata(run_dir, config, args.overrides)
    print(f"run dir: {run_dir}" + (" (resumed)" if args.resume else ""))
    return config, run_dir, training_args


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)  # follow long runs live (see train.py)

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--resume", type=Path, metavar="RUN_DIR", help="Run to resume")
    parser.add_argument("overrides", nargs="*", help="section.key=value, e.g. train.lr=5e-5")
    args = parser.parse_args()
    config, run_dir, training_args = start_run(args)
    set_seed(config.train.seed)

    tok = AutoTokenizer.from_pretrained(config.model.name)
    # fp32 master weights; with bf16=True the Trainer runs the forward under bf16 autocast.
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        dtype=getattr(torch, config.model.dtype),
        attn_implementation=config.model.attn_implementation,
    )

    with_length = not config.trl.packing  # packing rejects non-sequence columns
    train_dataset = to_dataset(tokenize_file(config.data.train_file, tok), with_length=with_length)
    eval_dataset = None
    if config.data.val_file is not None:
        eval_dataset = to_dataset(tokenize_file(config.data.val_file, tok), with_length=with_length)
    print(f"train: {len(train_dataset)} | val: {len(eval_dataset) if eval_dataset else 0}")

    run = start_wandb(config, run_dir)  # TRL's wandb callback logs into this run
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tok,
    )
    # The baseline evaluation (step 0) runs inside train() (eval_on_start). When resuming,
    # the Trainer reloads the last checkpoint of output_dir: weights, optimizer and scheduler
    # states, RNG states, and skips the batches already seen in the current epoch.
    trainer.train(resume_from_checkpoint=bool(args.resume) or None)

    # TRL raw logs: train (loss, grad_norm, learning_rate, mean_token_accuracy...), eval_loss.
    with (run_dir / "trainer_log.jsonl").open("w") as f:
        for record in trainer.state.log_history:
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
