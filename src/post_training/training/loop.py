"""The training loop, written out explicitly (no Trainer)."""

import contextlib
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from itertools import islice
from typing import cast

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedModel

from post_training.config import TrainConfig
from post_training.training.collator import Batch
from post_training.training.loss import model_loss_sum
from post_training.training.scheduler import warmup_cosine


@dataclass(frozen=True)
class StepLog:
    step: int
    loss: float
    lr: float
    grad_norm: float  # before clipping
    num_trained_tokens: int
    tokens_per_s: float  # all tokens processed (padding excluded), trained or not


@dataclass(frozen=True)
class EvalLog:
    step: int  # number of optimizer updates done before this evaluation (0 = untrained model)
    val_loss: float
    num_tokens: int  # trained tokens the loss is averaged over


@dataclass
class TrainResult:
    history: list[StepLog] = field(default_factory=list)
    evals: list[EvalLog] = field(default_factory=list)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> Optimizer:
    """AdamW with weight decay on weight matrices only.

    Decaying biases and norm gains (1-D tensors) pulls them towards 0, which only hurts:
    a LayerNorm/RMSNorm gain of 0 kills its whole layer's signal.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in trainable if p.ndim > 1]
    no_decay = [p for p in trainable if p.ndim <= 1]
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def precision_context(
    device: torch.device, mixed_precision: str
) -> contextlib.AbstractContextManager[None]:
    """bf16 autocast: matmuls and activations in bf16, weights untouched (still fp32)."""
    if mixed_precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def forward_loss_sum(
    model: nn.Module, batch: Batch, mixed_precision: str = "no"
) -> tuple[Tensor, Tensor]:
    """(summed loss, number of trained tokens). Logits are only computed where needed."""
    with precision_context(batch.input_ids.device, mixed_precision):
        return model_loss_sum(model, batch.input_ids, batch.attention_mask, batch.labels)


def forward_loss(model: nn.Module, batch: Batch, mixed_precision: str = "no") -> Tensor:
    loss_sum, num_trained = forward_loss_sum(model, batch, mixed_precision)
    # clamp avoids 0/0 = NaN on a batch with no trained token (the sum is 0 then anyway).
    return loss_sum / num_trained.clamp(min=1)


@torch.no_grad()  # no graph: less memory, faster
def evaluate(
    model: nn.Module,
    batches: Iterable[Batch],
    device: torch.device,
    max_batches: int | None = None,
    mixed_precision: str = "no",
) -> tuple[float, int]:
    """Token-weighted mean loss over the validation set, and the number of tokens it covers."""
    was_training = model.training
    model.eval()  # disables dropout: evaluation must be deterministic
    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device, dtype=torch.long)
    for cpu_batch in islice(batches, max_batches):
        loss_sum, num_tokens = forward_loss_sum(model, cpu_batch.to(device), mixed_precision)
        total_loss += loss_sum
        total_tokens += num_tokens
    model.train(was_training)
    n = int(total_tokens.item())
    loss = total_loss.item() / max(n, 1)
    release_cached_memory(device)
    return loss, n


def release_cached_memory(device: torch.device) -> None:
    """Give the accelerator's cached blocks back to the system.

    PyTorch keeps freed GPU memory in a cache for reuse. After an evaluation pass (different
    shapes than training), that cache can push a unified-memory Mac into swap: training then
    slows down by 100x.
    """
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def accumulate_gradients(
    model: nn.Module, micro_batches: Sequence[Batch], device: torch.device, mixed_precision: str
) -> tuple[float, int]:
    """Backward over several micro-batches as if they were one big batch.

    Normalization matters: the loss of the big batch is (sum of all token losses) / (total
    trained tokens). So we count the trained tokens of *all* micro-batches first (labels are on
    the CPU: free), and each micro-batch backpropagates loss_sum_i / total. Averaging the
    micro-batch means instead would weight a token of a 10-token micro-batch 10x more than a
    token of a 100-token one (the bug fixed in transformers' Trainer in 2024).

    Returns the mean loss over all trained tokens and their number.
    """
    total = sum(batch.num_trained_tokens for batch in micro_batches)
    loss_total = 0.0
    for cpu_batch in micro_batches:
        loss_sum, _ = forward_loss_sum(model, cpu_batch.to(device), mixed_precision)
        (loss_sum / max(total, 1)).backward()  # gradients add up across micro-batches
        loss_total += loss_sum.item()
    return loss_total / max(total, 1), total


def _cycle(batches: Iterable[Batch]) -> Iterator[Batch]:
    """Loop over the dataloader forever, starting a new epoch (and a new shuffle) each time.

    `itertools.cycle` would replay the first epoch's batches in the same order.
    """
    while True:
        yield from batches


def train(
    model: nn.Module,
    batches: Iterable[Batch],
    config: TrainConfig,
    device: torch.device,
    *,
    val_batches: Iterable[Batch] | None = None,
    on_log: Callable[[dict[str, float]], None] | None = None,
) -> TrainResult:
    """Train for `config.max_steps` optimizer updates.

    One update consumes `config.gradient_accumulation_steps` batches: the effective batch size
    is batch_size x gradient_accumulation_steps. `on_log` receives every step and evaluation
    log (e.g. to send them to wandb).
    """
    model.to(device)
    if config.gradient_checkpointing:
        # Each layer keeps only its input; the backward recomputes the layer's activations.
        cast(PreTrainedModel, model).gradient_checkpointing_enable()
    model.train()  # enables dropout (if any); `model.eval()` disables it
    optimizer = build_optimizer(model, config.lr, config.weight_decay)
    scheduler = LambdaLR(
        optimizer, warmup_cosine(config.warmup_steps, config.max_steps, config.min_lr_ratio)
    )

    result = TrainResult()

    def run_eval(updates_done: int) -> None:
        if val_batches is None:
            return
        val_loss, n_tokens = evaluate(
            model, val_batches, device, config.max_eval_batches, config.mixed_precision
        )
        log = EvalLog(step=updates_done, val_loss=val_loss, num_tokens=n_tokens)
        result.evals.append(log)
        if on_log is not None:
            on_log({"step": updates_done, "eval/loss": val_loss, "eval/num_tokens": n_tokens})
        print(f"eval after {updates_done:>4} steps | val_loss {val_loss:.4f} | {n_tokens:,} tokens")

    run_eval(0)  # baseline: the untrained model's validation loss
    data = _cycle(batches)
    for step in range(config.max_steps):
        start = time.perf_counter()
        micro_batches = [next(data) for _ in range(config.gradient_accumulation_steps)]

        # 1-2. forward + backward on each micro-batch; gradients accumulate into p.grad
        loss_value, num_trained = accumulate_gradients(
            model, micro_batches, device, config.mixed_precision
        )
        # 3. rescale all gradients together if their global L2 norm exceeds the max
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        lr = float(scheduler.get_last_lr()[0])  # LR used by *this* update
        optimizer.step()  # 4. update: p <- p - lr * adam_direction(p.grad)
        scheduler.step()  # 5. move to next step's LR
        optimizer.zero_grad(set_to_none=True)  # 6. otherwise grads add up across steps

        # .item() waits for the GPU (CUDA/MPS run asynchronously): read the values *before*
        # stopping the clock, otherwise we only time how long it took to queue the work.
        grad_norm_value = grad_norm.item()
        elapsed = time.perf_counter() - start
        n_tokens = sum(int(b.attention_mask.sum().item()) for b in micro_batches)
        log = StepLog(
            step=step,
            loss=loss_value,
            lr=lr,
            grad_norm=grad_norm_value,
            num_trained_tokens=num_trained,
            tokens_per_s=n_tokens / elapsed,
        )
        result.history.append(log)
        if on_log is not None:
            on_log({f"train/{k}" if k != "step" else k: v for k, v in asdict(log).items()})
        if step % config.log_every == 0 or step == config.max_steps - 1:
            print(
                f"step {log.step:>4} | loss {log.loss:.4f} | lr {log.lr:.2e} | "
                f"grad_norm {log.grad_norm:.2f} | trained_tok {log.num_trained_tokens:>4} | "
                f"{log.tokens_per_s:,.0f} tok/s"
            )

        updates_done = step + 1
        is_last = updates_done == config.max_steps
        if is_last or (config.eval_every and updates_done % config.eval_every == 0):
            run_eval(updates_done)
    return result
