"""Causal language modeling loss for SFT."""

from typing import cast

import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel

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


def model_loss_sum(
    model: nn.Module, input_ids: Tensor, attention_mask: Tensor, labels: Tensor
) -> tuple[Tensor, Tensor]:
    """Same result as `causal_lm_loss_sum(model(...).logits, labels)`, far less memory.

    The full logits tensor is (batch, seq, vocab): 151 936 scores per position, i.e. 1 GB in
    fp32 for a 1 764-token sequence, plus as much for its gradient. But only the positions
    predicting a trained token (~11% in xLAM) enter the loss. So we run the decoder to get the
    hidden states, keep the rows of those positions, and apply the output layer (`lm_head`) to
    them only.
    """
    hf_model = cast(PreTrainedModel, model)
    hidden: Tensor = hf_model.get_decoder()(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # The KV cache only serves generation; in training it would just hold K/V of every layer.
        use_cache=False,
    ).last_hidden_state

    # Position t predicts token t+1 (same shift as causal_lm_loss_sum).
    shift_labels = labels[:, 1:]
    trained = shift_labels != IGNORE_INDEX
    selected_hidden = hidden[:, :-1][trained]  # (n_trained, hidden_size)

    lm_head = cast(nn.Module, hf_model.get_output_embeddings())
    logits = lm_head(selected_hidden).float()  # (n_trained, vocab) instead of (batch, seq, vocab)
    loss_sum = F.cross_entropy(logits, shift_labels[trained], reduction="sum")
    return loss_sum, trained.sum()
