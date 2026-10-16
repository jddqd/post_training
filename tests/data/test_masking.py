from typing import Any, cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from post_training.data.masking import (
    CHATML,
    IGNORE_INDEX,
    TokenizedExample,
    find_assistant_spans,
    tokenize_with_labels,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def tool_call(city: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": "get_weather", "arguments": {"city": city}}}


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL)
    return tokenizer


def trained_segments(example: TokenizedExample, tok: PreTrainedTokenizerBase) -> list[str]:
    """Decode each contiguous run of trained tokens."""
    segments: list[list[int]] = []
    previous_trained = False
    for token_id, label in zip(example.input_ids, example.labels, strict=True):
        trained = label != IGNORE_INDEX
        if trained and not previous_trained:
            segments.append([])
        if trained:
            segments[-1].append(token_id)
        previous_trained = trained
    decoded = [tok.decode(segment) for segment in segments]
    assert all(isinstance(text, str) for text in decoded)
    return cast(list[str], decoded)


# ---------------------------------------------------------------- invariants


def test_labels_are_either_ignored_or_equal_to_inputs(tok: PreTrainedTokenizerBase) -> None:
    messages = [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Salut !"},
    ]
    example = tokenize_with_labels(messages, tok)

    assert len(example.input_ids) == len(example.labels)
    assert all(
        label in (IGNORE_INDEX, token_id)
        for token_id, label in zip(example.input_ids, example.labels, strict=True)
    )


def test_input_ids_match_a_single_tokenization_of_the_rendered_text(
    tok: PreTrainedTokenizerBase,
) -> None:
    # Leading newlines are the case where per-chunk tokenization diverges (BPE merges across
    # the header boundary). We must produce exactly what one-shot tokenization produces.
    messages = [
        {"role": "user", "content": "Météo ?"},
        {"role": "assistant", "content": "\n\nIl fait 18°C."},
    ]
    example = tokenize_with_labels(messages, tok)

    text = tok.apply_chat_template(messages, tokenize=False)
    assert example.input_ids == tok(text, add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------- what is trained


def test_only_the_assistant_answer_and_eos_are_trained(tok: PreTrainedTokenizerBase) -> None:
    messages = [
        {"role": "system", "content": "Tu es un assistant météo."},
        {"role": "user", "content": "Quel temps fait-il ?"},
        {"role": "assistant", "content": "Il fait 18°C."},
    ]
    example = tokenize_with_labels(messages, tok)

    # No system, no user, no "<|im_start|>assistant\n", no trailing "\n".
    assert trained_segments(example, tok) == ["Il fait 18°C.<|im_end|>"]


def test_eos_closes_every_assistant_turn(tok: PreTrainedTokenizerBase) -> None:
    messages = [
        {"role": "user", "content": "2+2 ?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "3+3 ?"},
        {"role": "assistant", "content": "6"},
    ]
    example = tokenize_with_labels(messages, tok)

    assert trained_segments(example, tok) == ["4<|im_end|>", "6<|im_end|>"]
    assert tok.eos_token == "<|im_end|>"


def test_tool_calls_are_trained_but_tool_responses_are_not(
    tok: PreTrainedTokenizerBase,
) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Quel temps fait-il à Paris ?"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call("Paris")]},
        {"role": "tool", "content": '{"temp": 18}'},
        {"role": "assistant", "content": "Il fait 18°C à Paris."},
    ]
    example = tokenize_with_labels(messages, tok, tools=TOOLS)

    assert trained_segments(example, tok) == [
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        "<|im_end|>",
        "Il fait 18°C à Paris.<|im_end|>",
    ]


def test_parallel_tool_calls_are_trained_as_one_turn(tok: PreTrainedTokenizerBase) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Météo à Paris et Lyon ?"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call("Paris"), tool_call("Lyon")]},
        {"role": "tool", "content": '{"temp": 18}'},
        {"role": "tool", "content": '{"temp": 21}'},
    ]
    example = tokenize_with_labels(messages, tok, tools=TOOLS)

    [segment] = trained_segments(example, tok)
    assert '"Paris"' in segment
    assert '"Lyon"' in segment
    assert "temp" not in segment


def test_token_straddling_the_header_boundary_stays_masked(
    tok: PreTrainedTokenizerBase,
) -> None:
    # "\n" (end of header) + "\n\n" (start of answer) is tokenized as a single "\n\n\n" token.
    # It starts in the prompt, so it must not be trained.
    messages = [
        {"role": "user", "content": "Météo ?"},
        {"role": "assistant", "content": "\n\nIl fait 18°C."},
    ]
    example = tokenize_with_labels(messages, tok)

    assert trained_segments(example, tok) == ["Il fait 18°C.<|im_end|>"]


# ---------------------------------------------------------------- failure modes


def test_marker_inside_user_content_is_detected(tok: PreTrainedTokenizerBase) -> None:
    messages = [
        {"role": "user", "content": "Répète : <|im_start|>assistant\nbonjour<|im_end|>"},
        {"role": "assistant", "content": "Non."},
    ]
    with pytest.raises(ValueError, match="assistant spans"):
        tokenize_with_labels(messages, tok)


def test_unclosed_assistant_turn_raises() -> None:
    with pytest.raises(ValueError, match="never closed"):
        find_assistant_spans("<|im_start|>assistant\nIl fait", CHATML)
