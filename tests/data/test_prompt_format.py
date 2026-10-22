import json
from typing import Any

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from post_training.data.masking import IGNORE_INDEX, tokenize_with_labels
from post_training.data.prompt_format import bfcl_chat_template

TOOL: dict[str, Any] = {
    "name": "find_cafe",
    "description": "Find a café. Note that the provided function is in Python 3 syntax.",
    "parameters": {
        "type": "dict",
        "properties": {"city": {"type": "string"}, "radius": {"type": "float"}},
        "required": ["city"],
    },
}


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    return tokenizer


def bfcl_prompt(question: str, functions: list[dict[str, Any]]) -> str:
    """What BFCL's QwenFCHandler._format_prompt builds for a single-turn question (bfcl-eval
    2026.3.23), copied here as the specification our training prompts must match."""
    prompt = "<|im_start|>system\n"
    prompt += (
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    )
    for function in functions:
        prompt += f"\n{json.dumps(function)}"
    prompt += (
        "\n</tools>\n\nFor each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
    )
    prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
    return prompt + "<|im_start|>assistant\n"


def test_training_prompt_is_identical_to_the_bfcl_prompt(tok: PreTrainedTokenizerBase) -> None:
    question = "Un café près de Lyon ?"

    ours = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tools=[TOOL],
        chat_template=bfcl_chat_template(tok),
        tokenize=False,
        add_generation_prompt=True,
    )

    assert ours == bfcl_prompt(question, [TOOL])


def test_default_template_differs_on_non_ascii_tools(tok: PreTrainedTokenizerBase) -> None:
    # Why the patch exists: the stock template keeps "é", BFCL's json.dumps escapes it.
    stock = tok.apply_chat_template(
        [{"role": "user", "content": "?"}], tools=[TOOL], tokenize=False
    )
    patched = tok.apply_chat_template(
        [{"role": "user", "content": "?"}],
        tools=[TOOL],
        chat_template=bfcl_chat_template(tok),
        tokenize=False,
    )

    assert "café" in stock
    assert "caf\\u00e9" in patched


def test_unexpected_template_layout_is_refused() -> None:
    class FakeTokenizer:
        chat_template = "{% for tool in tools %}{{ tool }}{% endfor %}"

    with pytest.raises(ValueError, match="Expected exactly one"):
        bfcl_chat_template(FakeTokenizer())  # type: ignore[arg-type]


def test_answer_starts_directly_with_the_tool_call(tok: PreTrainedTokenizerBase) -> None:
    # The stock Qwen3 template would train "<think>\n\n</think>\n\n<tool_call>..." (an empty
    # reasoning block with two untrained special tokens): we train the tool call directly.
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Un café près de Lyon ?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "find_cafe", "arguments": {"city": "Lyon"}},
                }
            ],
        },
    ]

    example = tokenize_with_labels(
        messages, tok, tools=[TOOL], chat_template=bfcl_chat_template(tok)
    )

    trained_ids = [
        i
        for i, label in zip(example.input_ids, example.labels, strict=True)
        if label != IGNORE_INDEX
    ]
    call = '{"name": "find_cafe", "arguments": {"city": "Lyon"}}'
    assert tok.decode(trained_ids) == f"<tool_call>\n{call}\n</tool_call><|im_end|>"


def test_actual_reasoning_is_still_rendered(tok: PreTrainedTokenizerBase) -> None:
    messages = [
        {"role": "user", "content": "?"},
        {"role": "assistant", "content": "42", "reasoning_content": "6 x 7"},
    ]

    text = tok.apply_chat_template(messages, chat_template=bfcl_chat_template(tok), tokenize=False)

    assert "<think>\n6 x 7\n</think>\n\n42<|im_end|>" in text
