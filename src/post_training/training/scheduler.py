"""Learning-rate schedule: linear warmup then cosine decay."""

import math
from collections.abc import Callable


def warmup_cosine(
    warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1
) -> Callable[[int], float]:
    """Return `step -> multiplier` of the peak LR, for `torch.optim.lr_scheduler.LambdaLR`.

    - Warmup: ramps linearly up to 1. Adam's moment estimates are noisy for the first steps;
      large updates then can wreck the pretrained weights.
    - Cosine: decays smoothly from 1 to `min_lr_ratio` at `total_steps`, then stays there.
    """

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return multiplier
