"""The data preparation pipeline: raw source rows -> clean, split, normalized conversations.

    source rows
      -> 1. convert + validate   (rejects: unsupported types, invalid calls, ...)
      -> 2. clean                (Unicode NFC, outer whitespace)
      -> 3. exact dedup          (on question + answer)
      -> 3b. decontamination     (drop questions overlapping the benchmark, if configured)
      -> 4. shuffle tool order   (removes the position bias)
      -> 5. length filter        (drop, never truncate)
      -> 6. stratified split     (same category mix in train and val)

Every step records how many rows survive (the funnel) and every rejected row is kept with its
reason, so that what we delete can be inspected.
"""

import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from transformers import PreTrainedTokenizerBase

from post_training.config import PrepareDataConfig
from post_training.data.conversation import Conversation
from post_training.data.decontaminate import Decontaminator
from post_training.data.masking import tokenize_with_labels
from post_training.data.preprocess import (
    category,
    clean,
    dedup_key,
    shuffle_tools,
    stratified_split,
)
from post_training.data.prompt_format import bfcl_chat_template
from post_training.data.xlam import RejectedError, convert_row


@dataclass(frozen=True)
class Rejected:
    reason: str
    detail: str
    row: Mapping[str, Any]


@dataclass
class PreparedData:
    train: list[Conversation]
    val: list[Conversation]
    train_lengths: list[int]  # token count of each train conversation, same order
    rejected: list[Rejected] = field(default_factory=list)
    funnel: list[tuple[str, int]] = field(default_factory=list)

    def rejected_counts(self) -> Counter[str]:
        return Counter(r.reason for r in self.rejected)


def prepare_xlam(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    config: PrepareDataConfig,
    decontaminator: Decontaminator | None = None,
) -> PreparedData:
    rejected: list[Rejected] = []
    funnel: list[tuple[str, int]] = []
    rng = random.Random(config.seed)
    chat_template = bfcl_chat_template(tokenizer)

    # 1-2. Convert, validate and clean.
    converted: list[tuple[Conversation, Mapping[str, Any]]] = []
    n_source = 0
    for row in rows:
        n_source += 1
        try:
            converted.append((clean(convert_row(dict(row))), row))
        except RejectedError as error:
            rejected.append(Rejected(error.reason, error.detail, row))
    funnel += [("source", n_source), ("valid", len(converted))]

    # 3. Exact dedup, after cleaning so that trivial variants collapse. First occurrence wins.
    seen: set[str] = set()
    unique: list[tuple[Conversation, Mapping[str, Any]]] = []
    for conversation, row in converted:
        key = dedup_key(conversation)
        if key in seen:
            rejected.append(Rejected("duplicate", key, row))
        else:
            seen.add(key)
            unique.append((conversation, row))
    funnel.append(("deduplicated", len(unique)))

    # 3b. Decontamination: never train on (near) copies of the evaluation questions, otherwise
    # the benchmark measures memorization. Checked on every user message.
    if decontaminator is not None:
        clean_rows: list[tuple[Conversation, Mapping[str, Any]]] = []
        for conversation, row in unique:
            hits = [
                hit
                for m in conversation.messages
                if m["role"] == "user" and (hit := decontaminator.check(m["content"]))
            ]
            if hits:
                rejected.append(Rejected("contaminated", str(hits[0]), row))
            else:
                clean_rows.append((conversation, row))
        unique = clean_rows
        funnel.append(("decontaminated", len(unique)))

    # 4-5. Shuffle tools, then measure the length the model will actually see (same tokenizer
    # and BFCL-style chat template as training). Tokenizing also checks that masking finds
    # every assistant turn.
    kept: list[tuple[Conversation, int]] = []
    for original, row in unique:
        conversation = shuffle_tools(original, rng) if config.shuffle_tools else original
        try:
            n_tokens = len(
                tokenize_with_labels(
                    conversation.messages,
                    tokenizer,
                    conversation.tools,
                    chat_template=chat_template,
                ).input_ids
            )
        except ValueError as error:
            rejected.append(Rejected("masking_error", str(error), row))
            continue
        if n_tokens > config.max_length:
            rejected.append(Rejected("too_long", f"{n_tokens} tokens", row))
            continue
        kept.append((conversation, n_tokens))
    funnel.append(("length_ok", len(kept)))

    # 6. Split last: after dedup, so that no duplicate leaks between train and val.
    train, val = stratified_split(
        kept, lambda pair: category(pair[0]), config.val_fraction, config.seed
    )
    funnel += [("train", len(train)), ("val", len(val))]

    return PreparedData(
        train=[conversation for conversation, _ in train],
        val=[conversation for conversation, _ in val],
        train_lengths=[n_tokens for _, n_tokens in train],
        rejected=rejected,
        funnel=funnel,
    )
