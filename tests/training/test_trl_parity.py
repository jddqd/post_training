"""TRL's SFTTrainer must train exactly like our loop: same labels, same loss."""

from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen2ForCausalLM

# Submodule: `trl` exposes it through a lazy import, which mypy cannot follow.
from trl.trainer.sft_trainer import SFTTrainer

from post_training.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, TrlConfig
from post_training.data.masking import IGNORE_INDEX, TokenizedExample
from post_training.training.collator import collate
from post_training.training.loop import forward_loss
from post_training.training.trl_trainer import sft_config, to_dataset

EXAMPLE = TokenizedExample(
    input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
    labels=[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 5, 6, 7, 8],
)


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    return tokenizer


def experiment(tmp_path: Path, **train: object) -> ExperimentConfig:
    data_file = tmp_path / "data.jsonl"
    data_file.touch()
    return ExperimentConfig(
        name="trl",
        output_dir=tmp_path,
        model=ModelConfig(name="tiny"),
        data=DataConfig(train_file=data_file, val_file=data_file),
        train=TrainConfig.model_validate({"max_steps": 1, "warmup_steps": 0, **train}),
    )


def test_sft_config_overrides_every_default_that_would_change_the_experiment(
    tmp_path: Path,
) -> None:
    args = sft_config(
        experiment(tmp_path, max_steps=100, warmup_steps=10, seed=7, batch_size=2, eval_every=10),
        tmp_path,
        True,
    )

    assert args.max_length is None  # TRL default 1024 would truncate long rows
    assert args.eval_on_start is True  # baseline evaluation, like our loop
    assert args.gradient_checkpointing is False  # TRL default True
    assert args.save_strategy == "no"  # TRL default: every 500 steps
    assert args.seed == 7  # TRL default 42
    assert args.optim == "adamw_torch"  # TRL default fused (CUDA only)
    assert args.lr_scheduler_type == "cosine_with_min_lr"
    assert args.lr_scheduler_kwargs == {"min_lr_rate": 0.1}
    assert args.report_to == []
    assert (args.max_steps, args.warmup_steps, args.per_device_train_batch_size) == (100, 10, 2)


def test_trl_specific_options_are_mapped(tmp_path: Path) -> None:
    trl = TrlConfig(
        group_by_length=True,
        optim="adamw_torch_fused",
        save_steps=50,
        neftune_noise_alpha=5.0,
        dataloader_num_workers=2,
        save_only_model=True,
    )
    config = experiment(tmp_path, adam_beta2=0.95).model_copy(update={"trl": trl})

    args = sft_config(config, tmp_path, has_eval=False)

    assert args.train_sampling_strategy == "group_by_length"
    assert args.optim == "adamw_torch_fused"
    assert (args.save_strategy, args.save_steps, args.save_only_model) == ("steps", 50, True)
    assert args.neftune_noise_alpha == 5.0
    assert args.dataloader_num_workers == 2
    assert args.adam_beta2 == 0.95  # shared with our loop (TrainConfig)
    assert (args.eval_strategy, args.eval_on_start) == ("no", False)


def test_trl_trains_on_our_labels_with_our_loss(
    tiny_model: Qwen2ForCausalLM, tok: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    with torch.no_grad():
        our_loss = forward_loss(tiny_model, collate([EXAMPLE], pad_token_id=0)).item()

    trainer = SFTTrainer(
        model=tiny_model,
        args=sft_config(experiment(tmp_path, eval_every=1), tmp_path, has_eval=True),
        train_dataset=to_dataset([EXAMPLE]),
        eval_dataset=to_dataset([EXAMPLE]),
        processing_class=tok,
    )

    # 1. TRL's batches carry our labels, untouched (not a copy of input_ids, not re-masked).
    prepared = trainer.train_dataset  # after TRL's own dataset preparation
    assert isinstance(prepared, Dataset)
    batch = trainer.data_collator([prepared[0]])
    assert batch["labels"].tolist() == [EXAMPLE.labels]

    # 2. Same loss, before any update (TRL may run on MPS: tiny float differences).
    assert abs(trainer.evaluate()["eval_loss"] - our_loss) < 1e-4
    trainer.train()
    first_logged = next(r for r in trainer.state.log_history if "loss" in r)
    assert abs(first_logged["loss"] - our_loss) < 1e-4


def test_packing_keeps_our_labels_and_restarts_positions_per_conversation(
    tiny_model: Qwen2ForCausalLM, tok: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    def example(ids: list[int], prompt_len: int) -> TokenizedExample:
        return TokenizedExample(
            input_ids=ids, labels=[IGNORE_INDEX] * prompt_len + ids[prompt_len:]
        )

    examples = [
        example(list(range(10, 18)), 4),  # 8 tokens
        example(list(range(20, 26)), 2),  # 6 tokens
        example(list(range(30, 35)), 3),  # 5 tokens
    ]
    trl = TrlConfig(packing=True, packing_length=16)
    # model_copy skips the flash_attention_2 check: only the data path is tested here (CPU).
    config = experiment(tmp_path).model_copy(update={"trl": trl})
    args = sft_config(config, tmp_path, has_eval=False)
    assert (args.max_length, args.packing_strategy, args.eval_packing) == (16, "bfd", False)

    trainer = SFTTrainer(
        model=tiny_model,
        args=args,
        train_dataset=to_dataset(examples, with_length=False),
        processing_class=tok,
    )

    prepared = trainer.train_dataset
    assert isinstance(prepared, Dataset)
    # Best-fit decreasing: 8 + 6 tokens in one sequence, 5 in another; nothing truncated.
    assert sorted(sum(row["seq_lengths"]) for row in prepared) == [5, 14]

    batch = trainer.data_collator(list(prepared))
    # Padding-free: one flat sequence, positions restart at 0 for every conversation.
    starts = (batch["position_ids"][0] == 0).nonzero().flatten().tolist()
    assert len(starts) == len(examples)
    segments = torch.tensor_split(batch["input_ids"][0], starts[1:])
    label_segments = torch.tensor_split(batch["labels"][0], starts[1:])
    by_first_token = {e.input_ids[0]: e for e in examples}
    for ids, labels in zip(segments, label_segments, strict=True):
        original = by_first_token[int(ids[0])]
        assert ids.tolist() == original.input_ids
        assert labels.tolist() == original.labels  # our masking, untouched by the packing
