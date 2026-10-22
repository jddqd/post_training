import torch
from transformers import Qwen2ForCausalLM

from post_training.config import TrainConfig
from post_training.data.masking import IGNORE_INDEX, TokenizedExample
from post_training.training.collator import Batch, collate
from post_training.training.loop import build_optimizer, evaluate, train


def test_overfits_a_single_example(tiny_model: Qwen2ForCausalLM) -> None:
    # The canonical sanity check for a training loop: if it cannot memorize one example,
    # something is broken (labels misaligned, gradients not flowing, LR schedule, ...).
    example = TokenizedExample(
        input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        labels=[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5, 6, 7, 8],
    )
    batches = [collate([example], pad_token_id=0)]
    config = TrainConfig(max_steps=60, lr=1e-2, warmup_steps=5, log_every=1000)

    history = train(tiny_model, batches, config, device=torch.device("cpu")).history

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


def make_batch(n_trained: int, length: int = 8) -> Batch:
    """One sequence of `length` tokens, the last `n_trained` of which are trained."""
    ids = list(range(1, length + 1))
    labels = [IGNORE_INDEX] * (length - n_trained) + ids[length - n_trained :]
    return collate([TokenizedExample(input_ids=ids, labels=labels)], pad_token_id=0)


def test_evaluate_weights_every_token_equally(tiny_model: Qwen2ForCausalLM) -> None:
    cpu = torch.device("cpu")
    small, large = make_batch(n_trained=2), make_batch(n_trained=6)

    loss_small, n_small = evaluate(tiny_model, [small], cpu)
    loss_large, n_large = evaluate(tiny_model, [large], cpu)
    loss_both, n_both = evaluate(tiny_model, [small, large], cpu)

    # Shift: a sequence with k trained labels has k-1 or k predicted positions.
    assert n_both == n_small + n_large
    expected = (loss_small * n_small + loss_large * n_large) / n_both  # not the mean of means
    assert abs(loss_both - expected) < 1e-5


def test_evaluate_restores_training_mode(tiny_model: Qwen2ForCausalLM) -> None:
    tiny_model.train()

    evaluate(tiny_model, [make_batch(n_trained=4)], torch.device("cpu"))

    assert tiny_model.training


def test_validation_runs_before_every_n_steps_and_at_the_end(
    tiny_model: Qwen2ForCausalLM,
) -> None:
    batch = make_batch(n_trained=5)
    config = TrainConfig(max_steps=25, lr=1e-2, warmup_steps=2, eval_every=10, log_every=1000)

    result = train(tiny_model, [batch], config, torch.device("cpu"), val_batches=[batch])

    assert [e.step for e in result.evals] == [0, 10, 20, 25]
    assert result.evals[-1].val_loss < result.evals[0].val_loss


def test_no_validation_without_val_batches(tiny_model: Qwen2ForCausalLM) -> None:
    config = TrainConfig(max_steps=3, warmup_steps=1, eval_every=1, log_every=1000)

    result = train(tiny_model, [make_batch(n_trained=4)], config, torch.device("cpu"))

    assert result.evals == []
