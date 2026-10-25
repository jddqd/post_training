"""Save a trained model so that it can be served and evaluated (e.g. by BFCL with vLLM)."""

from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizerBase


def save_checkpoint(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, path: Path, end_of_turn: str
) -> None:
    """Weights + tokenizer + a generation config that stops at the end of an assistant turn.

    SFT teaches the model to close its turn with `end_of_turn` (`<|im_end|>`), but generation
    stops on `generation_config.eos_token_id`. Qwen3-0.6B-Base only lists `<|endoftext|>` there:
    without adding the end-of-turn token, the fine-tuned model would keep generating after its
    answer, at inference and under vLLM (which reads this file).
    """
    end_id = tokenizer.convert_tokens_to_ids(end_of_turn)
    if not isinstance(end_id, int) or end_id == tokenizer.unk_token_id:
        raise ValueError(f"{end_of_turn!r} is not a token of this tokenizer")

    generation_config = model.generation_config
    assert generation_config is not None
    eos = generation_config.eos_token_id
    eos_ids: list[int] = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    if end_id not in eos_ids:
        generation_config.eos_token_id = [end_id, *eos_ids]

    # Training with gradient checkpointing turns the KV cache off (transformers' Trainer sets
    # use_cache=False); saved as is, every generation with transformers would recompute the
    # whole prefix at each token. vLLM ignores the flag, `generate` does not.
    model.config.use_cache = True

    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
