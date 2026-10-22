import torch
from torch import Tensor
from transformers import Qwen2ForCausalLM

from post_training.config import TrainConfig
from post_training.data.masking import IGNORE_INDEX, TokenizedExample
from post_training.training.collator import Batch, collate
from post_training.training.loop import (
    accumulate_gradients,
    build_optimizer,
    evaluate,
    forward_loss,
    forward_loss_sum,
    train,
)


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


def gradients_after_one_backward(model: Qwen2ForCausalLM, **options: str) -> tuple[float, Tensor]:
    model.zero_grad()
    loss = forward_loss(model, make_batch(n_trained=5), **options)
    loss.backward()
    weight = model.get_input_embeddings().weight
    assert isinstance(weight, Tensor)
    assert weight.grad is not None
    return loss.item(), weight.grad.clone()


def test_gradient_checkpointing_gives_the_same_loss_and_gradients(
    tiny_model: Qwen2ForCausalLM,
) -> None:
    tiny_model.train()
    loss, grad = gradients_after_one_backward(tiny_model)

    tiny_model.gradient_checkpointing_enable()
    loss_ckpt, grad_ckpt = gradients_after_one_backward(tiny_model)

    # Same math, the activations are just recomputed during the backward.
    assert abs(loss - loss_ckpt) < 1e-6
    assert torch.allclose(grad, grad_ckpt, atol=1e-6)


def test_bf16_autocast_keeps_fp32_weights_and_a_close_loss(tiny_model: Qwen2ForCausalLM) -> None:
    loss_fp32, _ = gradients_after_one_backward(tiny_model)
    loss_bf16, _ = gradients_after_one_backward(tiny_model, mixed_precision="bf16")

    assert abs(loss_fp32 - loss_bf16) < 0.05  # bf16 has ~3 significant digits
    config = TrainConfig(max_steps=2, warmup_steps=1, mixed_precision="bf16", log_every=1000)
    train(tiny_model, [make_batch(n_trained=5)], config, torch.device("cpu"))
    assert all(p.dtype == torch.float32 for p in tiny_model.parameters())  # master weights


def test_training_runs_with_every_memory_option(tiny_model: Qwen2ForCausalLM) -> None:
    batch = make_batch(n_trained=5)
    config = TrainConfig(
        max_steps=30,
        lr=1e-2,
        warmup_steps=2,
        gradient_checkpointing=True,
        mixed_precision="bf16",
        log_every=1000,
    )

    history = train(tiny_model, [batch], config, torch.device("cpu")).history

    assert history[-1].loss < history[0].loss


def embedding_grad(model: Qwen2ForCausalLM) -> Tensor:
    weight = model.get_input_embeddings().weight
    assert isinstance(weight, Tensor) and weight.grad is not None
    return weight.grad.clone()


def test_accumulation_equals_one_big_batch(tiny_model: Qwen2ForCausalLM) -> None:
    cpu = torch.device("cpu")
    short = TokenizedExample(input_ids=[1, 2, 3, 4], labels=[IGNORE_INDEX, IGNORE_INDEX, 3, 4])
    long = TokenizedExample(
        input_ids=[5, 6, 7, 8, 9, 10, 11, 12], labels=[IGNORE_INDEX, 6, 7, 8, 9, 10, 11, 12]
    )

    tiny_model.zero_grad()
    loss_big, n_big = accumulate_gradients(tiny_model, [collate([short, long], 0)], cpu, "no")
    grad_big = embedding_grad(tiny_model)

    tiny_model.zero_grad()
    loss_acc, n_acc = accumulate_gradients(
        tiny_model, [collate([short], 0), collate([long], 0)], cpu, "no"
    )
    grad_acc = embedding_grad(tiny_model)

    assert n_acc == n_big
    assert abs(loss_acc - loss_big) < 1e-5
    assert torch.allclose(grad_acc, grad_big, atol=1e-6)


def test_accumulated_loss_is_weighted_by_tokens(tiny_model: Qwen2ForCausalLM) -> None:
    # sum of token losses / total tokens, not the mean of the micro-batch means (which would
    # weight each of the 2 tokens of `short` like 3.5 tokens of `long`).
    cpu = torch.device("cpu")
    short = TokenizedExample(input_ids=[1, 2, 3, 4], labels=[IGNORE_INDEX, IGNORE_INDEX, 3, 4])
    long = TokenizedExample(
        input_ids=[5, 6, 7, 8, 9, 10, 11, 12], labels=[IGNORE_INDEX, 6, 7, 8, 9, 10, 11, 12]
    )
    sum_short, n_short = forward_loss_sum(tiny_model, collate([short], 0))
    sum_long, n_long = forward_loss_sum(tiny_model, collate([long], 0))

    loss_acc, n_acc = accumulate_gradients(
        tiny_model, [collate([short], 0), collate([long], 0)], cpu, "no"
    )

    assert (int(n_short), int(n_long), n_acc) == (2, 7, 9)
    expected = (sum_short + sum_long).item() / 9
    assert abs(loss_acc - expected) < 1e-5


def test_one_update_consumes_gradient_accumulation_steps_batches(
    tiny_model: Qwen2ForCausalLM,
) -> None:
    batch = make_batch(n_trained=5)  # 4 predicted positions after the shift
    config = TrainConfig(max_steps=3, warmup_steps=1, gradient_accumulation_steps=4, log_every=1000)
    logs: list[dict[str, float]] = []

    history = train(tiny_model, [batch], config, torch.device("cpu"), on_log=logs.append).history

    assert len(history) == 3
    assert all(log.num_trained_tokens == 4 * batch.num_trained_tokens for log in history)
    assert [log["step"] for log in logs] == [0, 1, 2]
    assert {"train/loss", "train/lr", "train/grad_norm"} <= set(logs[0])
