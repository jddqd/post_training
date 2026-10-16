import itertools

import pytest

from post_training.training.scheduler import warmup_cosine


def test_warmup_ramps_linearly_to_peak() -> None:
    schedule = warmup_cosine(warmup_steps=4, total_steps=20)

    assert [schedule(step) for step in range(4)] == [0.25, 0.5, 0.75, 1.0]


def test_cosine_decays_to_min_ratio_and_stays_there() -> None:
    schedule = warmup_cosine(warmup_steps=4, total_steps=20, min_lr_ratio=0.1)

    assert schedule(4) == pytest.approx(1.0)
    assert schedule(12) == pytest.approx(0.55)  # halfway through the decay
    assert schedule(20) == pytest.approx(0.1)
    assert schedule(100) == pytest.approx(0.1)


def test_never_increases_after_warmup() -> None:
    schedule = warmup_cosine(warmup_steps=4, total_steps=20)
    values = [schedule(step) for step in range(4, 30)]

    assert all(a >= b for a, b in itertools.pairwise(values))
