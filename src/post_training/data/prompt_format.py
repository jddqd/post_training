"""Render training conversations in the format the BFCL evaluation uses.

Two patches to the Qwen3 chat template:

1. Prompt side, to match BFCL exactly. BFCL's Qwen handler builds the prompt itself (it does
   not call `apply_chat_template`). With flat tools (no `{"type": "function"}` wrapper) the
   Qwen3 template already produces the same text, except that BFCL serializes each tool with
   `json.dumps(tool)`, which escapes non-ASCII characters (`"Café"` -> `"Caf\\u00e9"`), whereas
   the template's `tojson` filter keeps them. 3% of the xLAM training conversations contain
   non-ASCII characters in their tools.

2. Answer side, our choice for a base model. The Qwen3 template writes an empty
   `<think>\\n\\n</think>\\n\\n` block before every last assistant answer (Qwen3 is a hybrid
   reasoning model). Our data has no reasoning, and those two special tokens have untrained
   embeddings in Qwen3-Base: we drop the empty block, so the model answers directly with
   `<tool_call>`. BFCL accepts both (it strips a reasoning block if there is one). A non-empty
   `reasoning_content` is still rendered.
"""

from transformers import PreTrainedTokenizerBase

_PATCHES = {
    # 1. tools serialized like BFCL's json.dumps
    "tool | tojson": "tool | tojson(ensure_ascii=True)",
    # 2. a <think> block only when there is actual reasoning
    "loop.last or (not loop.last and reasoning_content)": "reasoning_content",
}


def bfcl_chat_template(tokenizer: PreTrainedTokenizerBase) -> str:
    """The tokenizer's chat template with the two patches above."""
    template = tokenizer.chat_template
    if not isinstance(template, str):
        raise ValueError("The tokenizer has no chat template.")
    for old, new in _PATCHES.items():
        if template.count(old) != 1:
            # A different template layout: patching blindly could silently change the prompt.
            raise ValueError(f"Expected exactly one {old!r} in the chat template.")
        template = template.replace(old, new)
    return template
