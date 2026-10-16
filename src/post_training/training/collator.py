"""Turn variable-length tokenized examples into padded tensors."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from post_training.data.masking import IGNORE_INDEX, TokenizedExample


@dataclass(frozen=True)
class Batch:
    input_ids: Tensor  # (batch, seq) token ids, padded with `pad_token_id`
    attention_mask: Tensor  # (batch, seq) 1 = real token, 0 = padding
    labels: Tensor  # (batch, seq) target ids, IGNORE_INDEX on masked and padded positions

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )

    @property
    def num_trained_tokens(self) -> int:
        return int((self.labels != IGNORE_INDEX).sum().item())


def collate(examples: Sequence[TokenizedExample], pad_token_id: int) -> Batch:
    """Right-pad every example to the longest one in the batch.

    The pad id itself does not matter: padded positions are hidden from attention by
    `attention_mask` and excluded from the loss by `IGNORE_INDEX` in `labels`.
    """
    max_len = max(len(example.input_ids) for example in examples)
    input_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    labels = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long)

    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(example.labels)

    return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
