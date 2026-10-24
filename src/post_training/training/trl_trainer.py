"""Bridge between our experiment config / tokenized data and TRL's SFTTrainer."""

from pathlib import Path

from datasets import Dataset

# Submodule: `trl` exposes it through a lazy import, which mypy cannot follow.
from trl.trainer.sft_config import SFTConfig

from post_training.config import ExperimentConfig
from post_training.data.masking import TokenizedExample


def to_dataset(examples: list[TokenizedExample]) -> Dataset:
    return Dataset.from_dict(
        {
            "input_ids": [e.input_ids for e in examples],
            "labels": [e.labels for e in examples],
            # Precomputed for train_sampling_strategy="group_by_length".
            "length": [len(e.input_ids) for e in examples],
        }
    )


def sft_config(config: ExperimentConfig, run_dir: Path, has_eval: bool) -> SFTConfig:
    """Our experiment config in SFTConfig terms.

    Every SFTConfig default that would change the experiment is set explicitly (e.g.
    max_length=1024 would silently truncate our sequences, seed=42 is not ours).
    """
    train, trl = config.train, config.trl
    evaluate = has_eval and train.eval_every is not None
    return SFTConfig(
        output_dir=str(run_dir / "trainer"),
        run_name=run_dir.name,
        seed=train.seed,
        # --- Optimization: same schedule as our loop (linear warmup, cosine to min_lr_ratio)
        max_steps=train.max_steps,
        per_device_train_batch_size=train.batch_size,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        learning_rate=train.lr,
        weight_decay=train.weight_decay,
        adam_beta1=train.adam_beta1,
        adam_beta2=train.adam_beta2,
        adam_epsilon=train.adam_epsilon,
        warmup_steps=train.warmup_steps,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": train.min_lr_ratio},
        max_grad_norm=train.max_grad_norm,
        optim=trl.optim,
        neftune_noise_alpha=trl.neftune_noise_alpha,
        # --- Precision and memory
        bf16=train.mixed_precision == "bf16",
        tf32=trl.tf32 or None,  # None leaves the PyTorch default (an explicit False is checked)
        gradient_checkpointing=train.gradient_checkpointing,  # TRL defaults to True
        gradient_checkpointing_kwargs={"use_reentrant": False},
        activation_offloading=trl.activation_offloading,
        torch_compile=trl.torch_compile,
        # --- Data: already tokenized, masked and length-filtered by our pipeline
        max_length=None,  # default 1024 would truncate (and cut the <|im_end|> of) long rows
        train_sampling_strategy="group_by_length" if trl.group_by_length else "random",
        padding_free=trl.padding_free,
        packing=trl.packing,
        dataloader_num_workers=trl.dataloader_num_workers,
        dataloader_pin_memory=trl.dataloader_pin_memory,
        # --- Evaluation: before training (baseline), every eval_every steps
        eval_strategy="steps" if evaluate else "no",
        eval_steps=train.eval_every,
        eval_on_start=evaluate,
        per_device_eval_batch_size=train.batch_size,
        # --- Logging
        logging_steps=train.log_every,
        logging_first_step=True,
        report_to=["wandb"] if config.wandb.enabled else "none",
        # --- Checkpoints: only when asked (TRL defaults to every 500 steps)
        save_strategy="steps" if trl.save_steps else "no",
        save_steps=trl.save_steps or 500,
        save_total_limit=trl.save_total_limit,
        save_only_model=trl.save_only_model,
        load_best_model_at_end=trl.load_best_model_at_end,
        metric_for_best_model="eval_loss" if trl.load_best_model_at_end else None,
    )
