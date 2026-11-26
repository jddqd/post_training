"""A finished sequence's cache must not keep the batch's arrays alive (no model needed)."""

import gc
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache, BatchKVCache

from rl_swe_agent.agent.generation import detach

BATCH = 17


def kv_batch() -> Any:
    """An attention layer's cache for the batch: K/V of 4096 positions per sequence."""
    cache = BatchKVCache([0] * BATCH)
    keys = mx.zeros((BATCH, 2, 4096, 256), mx.bfloat16)
    cache.update_and_fetch(keys, keys * 1)  # type: ignore[no-untyped-call]
    return cache


def delta_batch() -> Any:
    """A DeltaNet layer's cache for the batch: a fixed state per sequence."""
    cache = ArraysCache(1)
    cache[0] = mx.zeros((BATCH, 16, 128, 128), mx.float32)
    return cache


def nbytes(cache: Any) -> int:
    """The K/V or state arrays of a cache (not its small bookkeeping vectors)."""
    leaves = [leaf for _, leaf in tree_flatten(cache.state)]  # type: ignore[str-unpack]
    return sum(a.nbytes for a in leaves if isinstance(a, mx.array) and a.ndim > 1)


def kept(make_batch: Callable[[], Any], fix: Callable[[list[Any]], Any]) -> float:
    """Bytes still held for one extracted sequence once the batch is gone, over its own size."""
    batch = make_batch()
    mx.eval(batch.state)
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    whole = nbytes(batch)
    one = batch.extract(3)  # what mlx-lm hands back when sequence 3 finishes
    fix([one])
    del batch
    gc.collect()
    mx.clear_cache()
    return (mx.get_active_memory() - before + whole) / nbytes(one)


@pytest.mark.parametrize("make_batch", [kv_batch, delta_batch], ids=["attention", "deltanet"])
def test_an_extracted_cache_keeps_the_whole_batch_alive_until_detached(
    make_batch: Callable[[], Any],
) -> None:
    assert kept(make_batch, lambda cache: None) > BATCH / 2  # mlx-lm's extract, as is
    assert kept(make_batch, detach) == pytest.approx(1, abs=0.1)
