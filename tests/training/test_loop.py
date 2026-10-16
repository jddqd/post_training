import torch
from transformers import Qwen2ForCausalLM

from post_training.config import TrainConfig
from post_training.data.masking import IGNORE_INDEX, TokenizedExample
from post_training.training.collator import collate
from post_training.training.loop import build_optimizer, train


def test_overfits_a_single_example(tiny_model: Qwen2ForCausalLM) -> None:
    # The canonical sanity check for a training loop: if it cannot memorize one example,
    # something is broken (labels misaligned, gradients not flowing, LR schedule, ...).
    example = TokenizedExample(
        input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        labels=[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5, 6, 7, 8],
    )
    batches = [collate([example], pad_token_id=0)]
    config = TrainConfig(max_steps=60, lr=1e-2, warmup_steps=5, log_every=1000)

    history = train(tiny_model, batches, config, device=torch.device("cpu"))

    assert history[0].loss > 3.0  # ~ln(64) = 4.16 for a random model
    assert history[-1].loss < 0.1


def test_weight_decay_skips_biases_and_norms(tiny_model: Qwen2ForCausalLM) -> None:
    optimizer = build_optimizer(tiny_model, lr=1e-3, weight_decay=0.1)
    decay_group, no_decay_group = optimizer.param_groups

    assert decay_group["weight_decay"] == 0.1
    assert all(p.ndim > 1 for p in decay_group["params"])
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.ndim <= 1 for p in no_decay_group["params"])
    assert len(no_decay_group["params"]) > 0  # Qwen2 has RMSNorm gains and q/k/v biases
