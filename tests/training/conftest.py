import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

VOCAB_SIZE = 64


@pytest.fixture
def tiny_model() -> Qwen2ForCausalLM:
    """A randomly initialized Qwen2 with the real architecture but tiny sizes (~100k params)."""
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return Qwen2ForCausalLM(config)
