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

    # bf16: the same computation in another order differs by ~1e-2 in log-probabilities.
    assert (packed - separate).abs().max().item() < 0.1
    assert torch.equal(packed.argmax(-1), separate.argmax(-1))

    # Sensitivity: without the boundaries, b's tokens see a's (dense causal attention), and
    # its log-probabilities change far beyond bf16 noise. Otherwise the check above proves
    # nothing.
    dense = log_probs(input_ids=torch.tensor([a + b]))
    assert (dense[len(a) :] - separate[len(a) :]).abs().max().item() > 1.0
