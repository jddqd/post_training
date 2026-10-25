"""On a CUDA GPU: packed conversations must not attend to each other (run by runpod_setup.sh).

Packing puts several conversations in one sequence; only the varlen flash-attention kernels,
fed with the sequence boundaries (position_ids / cu_seq_lens from TRL's collator), keep each
conversation's attention to itself. If they silently fell back to dense attention, training
would see the previous conversations as context: no error, just wrong training. Compared here:
the logits of two texts computed separately, then packed through TRL's own collator.
"""

from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Submodule: `trl` exposes it through a lazy import, which mypy cannot follow.
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

MODEL = "Qwen/Qwen3-0.6B-Base"


@torch.no_grad()
def test_packed_conversations_do_not_attend_to_each_other() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda"
    )
    model.eval()
    a = tok("The capital of France is Paris, and the capital of Japan is").input_ids
    b = tok("def add(x, y):\n    return x + y\n\nprint(add(2,").input_ids

    def log_probs(**inputs: Any) -> torch.Tensor:
        # The collator also returns plain ints (max_length_q / max_length_k).
        on_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        logits = model(**on_gpu).logits[0]
        return torch.log_softmax(logits.float(), dim=-1)

    separate = torch.cat(
        [log_probs(input_ids=torch.tensor([a])), log_probs(input_ids=torch.tensor([b]))]
    )

    # Exactly what training receives with bfd packing: TRL's padding-free collator on a packed
    # row (seq_lengths marks the boundaries).
    collator = DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id, padding_free=True)
    batch = collator([{"input_ids": a + b, "seq_lengths": [len(a), len(b)]}])
    batch.pop("labels")
    packed = log_probs(**batch)

    def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """KL(q || p) per position: weighs each token by its probability. A max |difference|
        over the whole vocabulary would be dominated by tokens of probability ~1e-10, whose bf16
        logits (resolution ~0.1) differ with the matmul shapes alone."""
        return (q.exp() * (q - p)).sum(-1)

    first, second = slice(0, len(a)), slice(len(a), None)
    # The second conversation is the one that would see the first through the attention:
    # packed, it must match its separate computation (measured on an A40: KL 0.0).
    assert kl(packed[second], separate[second]).max().item() < 1e-3
    assert torch.equal(packed.argmax(-1), separate.argmax(-1))
    # The first one only differs by bf16 noise from the longer matmuls (KL ~0.01 measured,
    # the same with dense attention: it sees nothing before it either way).
    assert kl(packed[first], separate[first]).max().item() < 0.05

    # Sensitivity: without the boundaries (dense causal attention), the second conversation
    # sees the first and its predictions change (measured: KL 8.4). Otherwise the check above
    # proves nothing.
    dense = log_probs(input_ids=torch.tensor([a + b]))
    assert kl(dense[second], separate[second]).max().item() > 1.0
