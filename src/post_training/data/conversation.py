"""The normalized conversation format the training pipeline reads.

Every source dataset is converted once to a JSONL file where each line is:

    {"messages": [{"role": "user", "content": "..."}, ...], "tools": [{...}, ...]}

Messages follow the Hugging Face chat format. Tools are BFCL function docs
(`{"name", "description", "parameters"}`), the exact objects BFCL puts in the prompt; the chat
template serializes them as is. Training only ever reads this format: adding a dataset never
touches training code.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator

from post_training.config import StrictModel

ROLES = frozenset({"system", "user", "assistant", "tool"})


class Conversation(StrictModel):
    messages: list[dict[str, Any]] = Field(min_length=1)
    # Per conversation: in function-calling datasets each sample has its own set of tools.
    tools: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("messages")
    @classmethod
    def _check_messages(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, message in enumerate(messages):
            if message.get("role") not in ROLES:
                raise ValueError(f"message {i}: role must be one of {sorted(ROLES)}")
        if not any(m["role"] == "assistant" for m in messages):
            # Nothing would be trained: the example would only cost compute.
            raise ValueError("conversation has no assistant message")
        return messages


def load_jsonl(path: Path) -> list[Conversation]:
    """Load and validate a conversations file, reporting the line number of any bad record."""
    conversations: list[Conversation] = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                conversations.append(Conversation.model_validate_json(line))
            except ValidationError as error:
                raise ValueError(f"{path}:{line_number}: invalid conversation\n{error}") from error
    return conversations


def save_jsonl(path: Path, conversations: Iterable[Conversation]) -> None:
    """Write conversations in the format `load_jsonl` reads (UTF-8, one JSON object per line)."""
    with path.open("w") as f:
        for conversation in conversations:
            f.write(conversation.model_dump_json() + "\n")
