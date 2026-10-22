"""The training loop, written out explicitly (no Trainer)."""

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import islice

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from post_training.config import TrainConfig
from post_training.training.collator import Batch
from post_training.training.loss import causal_lm_loss, causal_lm_loss_sum
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


def forward_loss(model: nn.Module, batch: Batch) -> Tensor:
    # We don't pass `labels=` to the model: it would compute its own loss on top of ours.
    logits: Tensor = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
    return causal_lm_loss(logits, batch.labels)


@torch.no_grad()  # no graph: less memory, faster
def evaluate(
    model: nn.Module, batches: Iterable[Batch], device: torch.device, max_batches: int | None = None
) -> tuple[float, int]:
    """Token-weighted mean loss over the validation set, and the number of tokens it covers."""
    was_training = model.training
    model.eval()  # disables dropout: evaluation must be deterministic
    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device, dtype=torch.long)
    for cpu_batch in islice(batches, max_batches):
        batch = cpu_batch.to(device)
        logits: Tensor = model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).logits
        loss_sum, num_tokens = causal_lm_loss_sum(logits, batch.labels)
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
    val_batches: Iterable[Batch] | None = None,
) -> TrainResult:
    model.to(device)
    model.train()  # enables dropout (if any); `model.eval()` disables it
    optimizer = build_optimizer(model, config.lr, config.weight_decay)
    scheduler = LambdaLR(
        optimizer, warmup_cosine(config.warmup_steps, config.max_steps, config.min_lr_ratio)
    )

    result = TrainResult()

    def run_eval(updates_done: int) -> None:
        if val_batches is None:
            return
        val_loss, n_tokens = evaluate(model, val_batches, device, config.max_eval_batches)
        result.evals.append(EvalLog(step=updates_done, val_loss=val_loss, num_tokens=n_tokens))
        print(f"eval after {updates_done:>4} steps | val_loss {val_loss:.4f} | {n_tokens:,} tokens")

    run_eval(0)  # baseline: the untrained model's validation loss
    data = _cycle(batches)
    for step in range(config.max_steps):
        start = time.perf_counter()
        batch = next(data).to(device)

        loss = forward_loss(model, batch)  # 1. forward: logits -> scalar loss
        loss.backward()  # 2. backward: d(loss)/d(param) accumulated into p.grad
        # 3. rescale all gradients together if their global L2 norm exceeds the max
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        lr = float(scheduler.get_last_lr()[0])  # LR used by *this* update
        optimizer.step()  # 4. update: p <- p - lr * adam_direction(p.grad)
        scheduler.step()  # 5. move to next step's LR
        optimizer.zero_grad(set_to_none=True)  # 6. otherwise grads add up across steps

        # .item() waits for the GPU (CUDA/MPS run asynchronously): read the values *before*
        # stopping the clock, otherwise we only time how long it took to queue the work.
        loss_value, grad_norm_value = loss.item(), grad_norm.item()
        elapsed = time.perf_counter() - start
        log = StepLog(
            step=step,
            loss=loss_value,
            lr=lr,
            grad_norm=grad_norm_value,
            num_trained_tokens=batch.num_trained_tokens,
            tokens_per_s=batch.attention_mask.sum().item() / elapsed,
        )
        result.history.append(log)
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
