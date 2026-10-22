"""The prompts we train on must be exactly the prompts BFCL evaluates with.

Compares our rendering (chat template + `bfcl_chat_template`) with BFCL's own Qwen handler on
real BFCL test entries. Run with:  uv run --project eval/bfcl pytest eval/bfcl
"""

from typing import Any

import pytest
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.utils import load_dataset_entry
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from post_training.data.prompt_format import bfcl_chat_template

MODEL = "Qwen/Qwen3-0.6B-Base"
CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL)
    return tokenizer


@pytest.fixture(scope="module")
def handler() -> QwenFCHandler:
    return QwenFCHandler(MODEL, temperature=0.001, registry_name=f"{MODEL}-FC", is_fc_model=True)


def entries(category: str, n: int = 25) -> list[dict[str, Any]]:
    # Exactly what `bfcl generate` loads: the language hint is already appended, once.
    return load_dataset_entry(category)[:n]


@pytest.mark.parametrize("category", CATEGORIES)
def test_our_prompt_matches_bfcl(
    category: str, tok: PreTrainedTokenizerBase, handler: QwenFCHandler
) -> None:
    template = bfcl_chat_template(tok)
    for entry in entries(category):
        messages = entry["question"][0]  # single-turn: the first (and only) turn
        functions = entry["function"]

        expected = handler._format_prompt(messages, functions)
        ours = tok.apply_chat_template(
            messages,
            tools=functions,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
        )

        assert ours == expected, entry["id"]
        # Our training data appends the hint once too (BFCL_PYTHON_HINT in data/xlam.py).
        assert all(f["description"].count("Python 3 syntax") == 1 for f in functions)
