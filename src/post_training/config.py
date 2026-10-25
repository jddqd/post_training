"""Experiment configuration, validated at the program boundary.

An experiment is fully described by a YAML file (see `configs/`) plus optional command-line
overrides in `section.key=value` form. Everything is validated before any model is loaded, so a
typo or an inconsistent value fails at second 0 instead of after an hour of GPU time.
"""

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


class StrictModel(BaseModel):
    # frozen: a config never changes during a run.
    # extra="forbid": an unknown key (e.g. `learning_rate` instead of `lr`) is an error instead of
    # being silently ignored while the run uses the default value.
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(StrictModel):
    name: str
    # Master weights in fp32: pure bf16 weights would round most Adam updates to zero.
    # Use train.mixed_precision=bf16 for bf16 computations.
    dtype: Literal["float32"] = "float32"
    # "sdpa": PyTorch's fused attention (everywhere). "flash_attention_2": CUDA + the flash-attn
    # package; required by padding-free batches and packing.
    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = "sdpa"


class DataConfig(StrictModel):
    # JSONL of normalized conversations (see `post_training.data.conversation`).
    # FilePath: must exist at config load time, not at the first batch.
    train_file: FilePath
    val_file: FilePath | None = None


class TrainConfig(StrictModel):
    max_steps: PositiveInt = 100  # optimizer updates
    batch_size: PositiveInt = 4  # per forward pass
    # Batches per update: effective batch = batch_size x gradient_accumulation_steps.
    gradient_accumulation_steps: PositiveInt = 1
    lr: PositiveFloat = 1e-5
    weight_decay: NonNegativeFloat = 0.0
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    adam_epsilon: PositiveFloat = 1e-8
    warmup_steps: NonNegativeInt = 10
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    max_grad_norm: PositiveFloat = 1.0
    log_every: PositiveInt = 1
    # Validation (if `data.val_file` is set): before training, every `eval_every` steps, at the end.
    eval_every: PositiveInt | None = None
    max_eval_batches: PositiveInt | None = None  # cap for quick evals (e.g. on a laptop)
    # Validation rows per forward (default: batch_size). Differs with packing, where
    # batch_size counts packed sequences but validation rows stay unpacked.
    eval_batch_size: PositiveInt | None = None
    save_checkpoint: bool = True  # final weights in <run dir>/checkpoint, ready for BFCL
    # Memory (step 4). Recompute activations in the backward instead of storing them.
    gradient_checkpointing: bool = False
    # Forward computations in bf16 (half the activation memory); weights and optimizer stay fp32.
    mixed_precision: Literal["no", "bf16"] = "no"
    seed: int = 0

    @model_validator(mode="after")
    def _warmup_shorter_than_training(self) -> Self:
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < max_steps ({self.max_steps})"
            )
        return self


class WandbConfig(StrictModel):
    """Experiment tracking with Weights & Biases (both training scripts)."""

    enabled: bool = False
    project: str = "post-training"
    entity: str | None = None  # team or user; None = the account of the API key
    # "offline" writes the run locally (sync later with `wandb sync`), "disabled" is a no-op.
    mode: Literal["online", "offline", "disabled"] = "online"


class TrlConfig(StrictModel):
    """Options only used by scripts/train_trl.py (TRL's SFTTrainer / transformers' Trainer)."""

    # --- Optimizer
    # "adamw_torch_fused": one kernel for all parameters (CUDA); "adamw_8bit": bitsandbytes (CUDA).
    optim: Literal["adamw_torch", "adamw_torch_fused", "adamw_8bit"] = "adamw_torch"

    # --- Batching
    # Batches of similar lengths: far less padding (32-55% of the compute at batch 4-32 on xLAM).
    group_by_length: bool = False
    # Concatenate the batch into one sequence without padding: needs flash_attention_2.
    padding_free: bool = False
    # Several conversations per sequence (best-fit decreasing, padding-free): flash_attention_2.
    # Every forward then holds batch_size x packing_length tokens, whatever the row lengths.
    packing: bool = False
    # Length of a packed sequence; >= the data max_length (longer rows would be truncated).
    packing_length: PositiveInt | None = None

    # --- Speed / memory
    tf32: bool = False  # TensorFloat-32 for the remaining fp32 matmuls (NVIDIA Ampere and newer)
    torch_compile: bool = (
        False  # faster steps after a compilation warm-up; recompiles on new shapes
    )
    # Move activations to CPU memory instead of recomputing them (alternative to checkpointing).
    activation_offloading: bool = False

    # --- Regularization
    # NEFTune: uniform noise on the input embeddings during training (reported to help chat SFT).
    neftune_noise_alpha: PositiveFloat | None = None

    # --- Data loading
    dataloader_num_workers: NonNegativeInt = 0
    dataloader_pin_memory: bool = True  # faster host -> GPU copies (no effect on MPS)

    # --- Checkpoints: to resume a run interrupted (e.g. a preempted GPU pod)
    save_steps: PositiveInt | None = None
    save_total_limit: PositiveInt = 2
    save_only_model: bool = False  # True: smaller checkpoints, but no optimizer state to resume
    # Final weights = the checkpoint with the lowest validation loss (needs save_steps).
    load_best_model_at_end: bool = False


class ExperimentConfig(StrictModel):
    name: str = Field(pattern=r"^[\w-]+$")  # used in the run directory name
    output_dir: Path = Path("outputs")
    model: ModelConfig
    data: DataConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    trl: TrlConfig = Field(default_factory=TrlConfig)

    @model_validator(mode="after")
    def _compatible_options(self) -> Self:
        needs_flash = [name for name in ("padding_free", "packing") if getattr(self.trl, name)]
        if needs_flash and self.model.attn_implementation != "flash_attention_2":
            raise ValueError(
                f"trl.{needs_flash[0]} needs model.attn_implementation=flash_attention_2"
            )
        if self.trl.packing and self.trl.packing_length is None:
            raise ValueError("trl.packing needs trl.packing_length (tokens per packed sequence)")
        if self.trl.packing and self.trl.group_by_length:
            # Packed sequences all have the same length: there is nothing to group.
            raise ValueError("trl.packing and trl.group_by_length are mutually exclusive")
        if self.trl.load_best_model_at_end:
            every = self.train.eval_every
            if not (self.trl.save_steps and every and self.trl.save_steps % every == 0):
                raise ValueError(
                    "trl.load_best_model_at_end needs trl.save_steps, "
                    "a multiple of train.eval_every"
                )
        return self


class PrepareDataConfig(StrictModel):
    """How to turn a source dataset into normalized train/val JSONL files."""

    source: Literal["xlam", "toolace"] = "xlam"  # picks the converter (data/prepare.py)
    hf_dataset: str = "Salesforce/xlam-function-calling-60k"
    # None = latest; the resolved commit sha is pinned in the report either way.
    revision: str | None = None
    # Lengths are measured with the tokenizer and chat template of the model we will train.
    tokenizer: str
    max_length: PositiveInt = 2048
    val_fraction: float = Field(default=0.03, gt=0.0, lt=1.0)
    shuffle_tools: bool = True
    # Benchmark questions to decontaminate against (eval/bfcl/export_questions.py), if any.
    decontamination_file: FilePath | None = None
    decontamination_ngram: PositiveInt = 8
    decontamination_min_coverage: float = Field(default=0.5, gt=0.0, le=1.0)
    seed: int = 0
    output_dir: Path
    # HF dataset repo ("user/name") the output is pushed to with `prepare_data.py --push`.
    hub_repo: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")
    hub_private: bool = True


class CombineSourceConfig(StrictModel):
    name: str = Field(pattern=r"^[\w-]+$")
    dir: DirectoryPath  # a prepare_data.py output: train.jsonl, val.jsonl, report.json


class CombineDataConfig(StrictModel):
    """Several prepared datasets -> one (scripts/combine_datasets.py)."""

    # Order matters: a conversation found in several sources is kept in the first one.
    sources: list[CombineSourceConfig] = Field(min_length=2)
    seed: int = 0
    output_dir: Path
    hub_repo: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")
    hub_private: bool = True

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"source names must be unique, got {names}")
        return self


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply `section.key=value` overrides to a nested dict (the input is not modified).

    Values are parsed as YAML, so `train.lr=5e-5`, `train.shuffle=false` or `a.b=[1, 2]` get
    sensible types. Pydantic then validates and coerces them like values from the file.
    """
    result = copy.deepcopy(data)
    for override in overrides:
        key, sep, raw_value = override.partition("=")
        if not sep or not key:
            raise ValueError(f"Override must look like 'section.key=value', got {override!r}")
        *parents, leaf = key.split(".")
        node = result
        for part in parents:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override {key!r}: {part!r} is not a section")
            node = child
        node[leaf] = yaml.safe_load(raw_value)
    return result


def load_yaml_config[M: BaseModel](model: type[M], path: Path, overrides: Sequence[str] = ()) -> M:
    """Read a YAML file, apply overrides, validate it as `model`."""
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return model.model_validate(apply_overrides(data, overrides))


def load_config(path: Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    return load_yaml_config(ExperimentConfig, path, overrides)
