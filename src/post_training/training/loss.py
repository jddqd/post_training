"""Causal language modeling loss for SFT."""

import torch.nn.functional as F
from torch import Tensor

from post_training.data.masking import IGNORE_INDEX


def causal_lm_loss_sum(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Summed next-token cross-entropy over trained positions, and the number of positions.

    Keeping sum and count separate lets callers average over several batches correctly:
    sum(loss_sums) / sum(counts) weights every token equally, whereas averaging per-batch means
    over-weights the tokens of batches with few trained tokens.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    loss_sum = F.cross_entropy(
        # Upcast: softmax over a ~150k vocabulary in bf16 loses precision.
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss_sum, (shift_labels != IGNORE_INDEX).sum()


def causal_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean next-token cross-entropy over trained positions.

    The logits at position t are the model's prediction for token t+1, so we compare
    `logits[:, :-1]` with `labels[:, 1:]`:

        input  :  <|im_start|>  assistant  \\n     4       <|im_end|>
        logits :  -> assistant  -> \\n     -> 4    -> eos  -> \\n
        labels :  (dropped)     -100       -100    4       <|im_end|>

    Args:
        logits: (batch, seq, vocab) unnormalized scores.
        labels: (batch, seq) target ids, IGNORE_INDEX where no loss is wanted.

    Returns:
        Scalar loss, averaged over all trained tokens of the batch (0 if there are none).
    """
    loss_sum, num_trained = causal_lm_loss_sum(logits, labels)
    # clamp avoids 0/0 = NaN on a batch with no trained token (the sum is 0 then anyway).
    return loss_sum / num_trained.clamp(min=1)
