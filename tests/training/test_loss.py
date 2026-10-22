import math

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM

from post_training.data.masking import IGNORE_INDEX
from post_training.training.loss import causal_lm_loss, causal_lm_loss_sum, model_loss_sum
from tests.training.conftest import VOCAB_SIZE


def test_matches_a_manual_next_token_computation() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, VOCAB_SIZE)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 7, 9]])

    # Position 1 predicts label 7, position 2 predicts label 9. Position 3 predicts nothing.
    log_probs = F.log_softmax(logits[0], dim=-1)
    expected = -(log_probs[1, 7] + log_probs[2, 9]) / 2

    assert torch.allclose(causal_lm_loss(logits, labels), expected)


def test_masked_positions_have_no_effect_on_loss_or_gradient() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, VOCAB_SIZE, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 7, 9]])

    causal_lm_loss(logits, labels).backward()
    assert logits.grad is not None

    # Logits at position t are judged against labels[t + 1]: positions 0 (-> label -100)
    # and 3 (-> past the end) receive no gradient; positions 1 and 2 do.
    assert torch.all(logits.grad[0, 0] == 0)
    assert torch.all(logits.grad[0, 3] == 0)
    assert torch.any(logits.grad[0, 1] != 0)
    assert torch.any(logits.grad[0, 2] != 0)


def test_is_zero_not_nan_when_nothing_is_trained() -> None:
    logits = torch.randn(2, 3, VOCAB_SIZE)
    labels = torch.full((2, 3), IGNORE_INDEX)

    loss = causal_lm_loss(logits, labels)

    assert not math.isnan(loss.item())
    assert loss.item() == 0.0


def test_matches_the_loss_computed_inside_transformers(tiny_model: Qwen2ForCausalLM) -> None:
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
    labels = input_ids.clone()
    labels[:, :6] = IGNORE_INDEX  # mask a "prompt"

    output = tiny_model(input_ids=input_ids, labels=labels)

    assert torch.allclose(causal_lm_loss(output.logits, labels), output.loss)


def test_selected_positions_loss_equals_the_full_logits_loss(tiny_model: Qwen2ForCausalLM) -> None:
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 7:] = 0  # second sequence is padded
    labels = input_ids.clone()
    labels[:, :5] = IGNORE_INDEX
    labels[1, 7:] = IGNORE_INDEX

    full_sum, full_count = causal_lm_loss_sum(
        tiny_model(input_ids=input_ids, attention_mask=attention_mask).logits, labels
    )
    selected_sum, selected_count = model_loss_sum(tiny_model, input_ids, attention_mask, labels)

    assert selected_count == full_count
    assert torch.allclose(selected_sum, full_sum, atol=1e-5)
