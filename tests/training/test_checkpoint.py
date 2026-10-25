from pathlib import Path

import pytest
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen2ForCausalLM,
)

from post_training.training.checkpoint import save_checkpoint

ENDOFTEXT, IM_END = 151643, 151645


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    return tokenizer


def test_generation_stops_at_end_of_turn_after_reload(
    tiny_model: Qwen2ForCausalLM, tok: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    tiny_model.generation_config.eos_token_id = ENDOFTEXT  # like Qwen3-0.6B-Base

    save_checkpoint(tiny_model, tok, tmp_path, end_of_turn="<|im_end|>")

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert reloaded.generation_config.eos_token_id == [IM_END, ENDOFTEXT]
    assert AutoTokenizer.from_pretrained(tmp_path).get_vocab() == tok.get_vocab()


def test_end_of_turn_is_not_added_twice(tiny_model: Qwen2ForCausalLM, tok, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    tiny_model.generation_config.eos_token_id = [IM_END, ENDOFTEXT]  # like Qwen3-0.6B

    save_checkpoint(tiny_model, tok, tmp_path, end_of_turn="<|im_end|>")

    assert tiny_model.generation_config.eos_token_id == [IM_END, ENDOFTEXT]


def test_unknown_end_of_turn_token_is_refused(
    tiny_model: Qwen2ForCausalLM, tok: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="not a token"):
        save_checkpoint(tiny_model, tok, tmp_path, end_of_turn="<|not_a_token|>")


def test_kv_cache_is_re_enabled_after_training_turned_it_off(
    tiny_model: Qwen2ForCausalLM, tok: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    tiny_model.config.use_cache = False  # what the Trainer does with gradient checkpointing

    save_checkpoint(tiny_model, tok, tmp_path, end_of_turn="<|im_end|>")

    assert AutoModelForCausalLM.from_pretrained(tmp_path).config.use_cache is True
