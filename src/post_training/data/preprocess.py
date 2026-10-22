"""Source-agnostic preprocessing steps, applied to normalized conversations."""

import hashlib
import json
import random
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Sequence

from post_training.data.conversation import Conversation


def normalize_text(text: str) -> str:
    """Canonical Unicode (NFC: 'é' as one code point, not 'e' + accent) and no outer spaces."""
    return unicodedata.normalize("NFC", text).strip()


def clean(conversation: Conversation) -> Conversation:
    messages = [
        {**m, "content": normalize_text(m["content"])} if isinstance(m.get("content"), str) else m
        for m in conversation.messages
    ]
    return conversation.model_copy(update={"messages": messages})


def dedup_key(conversation: Conversation) -> str:
    """Hash of the messages only (question + answer), not the tools.

    Two rows asking the same thing with the same calls are duplicates even if their tool lists
    differ. Two rows with the same question but different arguments are *not*: they teach the
    model to copy the right values.
    """
    canonical = json.dumps(conversation.messages, sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest()


def shuffle_tools(conversation: Conversation, rng: random.Random) -> Conversation:
    """Random tool order. In xLAM the called function is listed first in 49% of the rows (33%
    expected by chance): without shuffling, "call the first tool" becomes a useful shortcut."""
    tools = list(conversation.tools)
    rng.shuffle(tools)
    return conversation.model_copy(update={"tools": tools})


def category(conversation: Conversation) -> str:
    """BFCL-style category of a single-turn function-calling conversation."""
    calls = [c for m in conversation.messages for c in m.get("tool_calls") or []]
    if not calls:
        return "no_call"
    if len(calls) == 1:
        return "simple" if len(conversation.tools) == 1 else "multiple"
    distinct = {c["function"]["name"] for c in calls}
    return "parallel" if len(distinct) == 1 else "parallel_multiple"


def stratified_split[T](
    items: Sequence[T], key: Callable[[T], str], val_fraction: float, seed: int
) -> tuple[list[T], list[T]]:
    """Split so that every group (e.g. category) has the same proportion in train and val."""
    rng = random.Random(seed)
    groups: dict[str, list[T]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)

    train: list[T] = []
    val: list[T] = []
    for name in sorted(groups):  # sorted: same order, hence same split, on every run
        group = groups[name]
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
