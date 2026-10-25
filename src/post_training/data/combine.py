"""Combine several prepared datasets (same normalized format) into one.

Each source keeps its own train / validation split: the combined validation set is the union of
the sources' validation sets, so a validation loss stays comparable with each source's own.
Across sources:
- exact duplicates (same messages) are kept once, in the first source listed;
- a validation conversation that is in another source's train set is dropped (no leak).
"""

import random
from collections import Counter
from dataclasses import dataclass, field

from post_training.data.conversation import Conversation
from post_training.data.preprocess import dedup_key


@dataclass(frozen=True)
class Source:
    name: str
    train: list[Conversation]
    val: list[Conversation]


@dataclass
class CombinedData:
    train: list[Conversation]
    val: list[Conversation]
    # Rows kept per source and split, e.g. {"xlam": {"train": 55627, "val": 1721}}.
    kept: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped: Counter[str] = field(default_factory=Counter)  # "<source>/<split>: <reason>"


def combine(sources: list[Source], seed: int) -> CombinedData:
    train_keys = {dedup_key(c) for source in sources for c in source.train}
    seen: set[str] = set()
    result = CombinedData(train=[], val=[])
    for source in sources:
        result.kept[source.name] = {}
        for split, rows, output in (
            ("train", source.train, result.train),
            ("val", source.val, result.val),
        ):
            kept = 0
            for conversation in rows:
                key = dedup_key(conversation)
                if split == "val" and key in train_keys:  # checked first: the worse problem
                    result.dropped[f"{source.name}/{split}: in a train set"] += 1
                elif key in seen:
                    result.dropped[f"{source.name}/{split}: duplicate"] += 1
                else:
                    seen.add(key)
                    output.append(conversation)
                    kept += 1
            result.kept[source.name][split] = kept
    # Interleave the sources: a file read in order (or a sequential sampler) sees the mix.
    rng = random.Random(seed)
    rng.shuffle(result.train)
    rng.shuffle(result.val)
    return result
