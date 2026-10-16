import torch

from post_training.data.masking import IGNORE_INDEX, TokenizedExample
from post_training.training.collator import collate

PAD = 0


def test_right_pads_to_the_longest_example() -> None:
    batch = collate(
        [
            TokenizedExample(input_ids=[5, 6, 7], labels=[IGNORE_INDEX, 6, 7]),
            TokenizedExample(input_ids=[8], labels=[8]),
        ],
        pad_token_id=PAD,
    )

    assert batch.input_ids.tolist() == [[5, 6, 7], [8, PAD, PAD]]
    assert batch.attention_mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch.labels.tolist() == [[IGNORE_INDEX, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]
    assert batch.input_ids.dtype == torch.long


def test_counts_only_trained_tokens() -> None:
    batch = collate(
        [
            TokenizedExample(input_ids=[5, 6, 7], labels=[IGNORE_INDEX, 6, 7]),
            TokenizedExample(input_ids=[8], labels=[8]),
        ],
        pad_token_id=PAD,
    )

    assert batch.num_trained_tokens == 3
