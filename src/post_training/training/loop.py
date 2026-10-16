"""The training loop, written out explicitly (no Trainer)."""

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from post_training.config import TrainConfig
from post_training.training.collator import Batch
from post_training.training.loss import causal_lm_loss
from post_training.training.scheduler import warmup_cosine


@dataclass(frozen=True)
class StepLog:
    step: int
    loss: float
    lr: float
    grad_norm: float  # before clipping
    num_trained_tokens: int
    tokens_per_s: float  # all tokens processed (padding excluded), trained or not


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
) -> list[StepLog]:
    model.to(device)
    model.train()  # enables dropout (if any); `model.eval()` disables it
    optimizer = build_optimizer(model, config.lr, config.weight_decay)
    scheduler = LambdaLR(
        optimizer, warmup_cosine(config.warmup_steps, config.max_steps, config.min_lr_ratio)
    )

    history: list[StepLog] = []
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

        # .item() forces a GPU sync: fine here, but avoid it every step in a fast loop.
        elapsed = time.perf_counter() - start
        log = StepLog(
            step=step,
            loss=loss.item(),
            lr=lr,
            grad_norm=grad_norm.item(),
            num_trained_tokens=batch.num_trained_tokens,
            tokens_per_s=batch.attention_mask.sum().item() / elapsed,
        )
        history.append(log)
        if step % config.log_every == 0 or step == config.max_steps - 1:
            print(
                f"step {log.step:>4} | loss {log.loss:.4f} | lr {log.lr:.2e} | "
                f"grad_norm {log.grad_norm:.2f} | trained_tok {log.num_trained_tokens:>4} | "
                f"{log.tokens_per_s:,.0f} tok/s"
            )
    return history
