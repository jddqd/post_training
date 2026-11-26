"""Memory over time while rollouts run, sampled by a background thread (for the timeline plot).

    with ResourceMonitor() as monitor:
        groups = run_step(...)
    monitor.samples   # [{"time", "mlx_gb", "mlx_cache_gb", "process_gb", "system_used_gb"}, ...]

mlx_gb: MLX's buffers in use (weights, caches, activations: the GPU's working memory, in unified
memory); mlx_cache_gb: freed buffers MLX keeps for reuse, still held on the GPU (the two together
must stay under Metal's limit); process_gb: this Python process's resident memory;
system_used_gb: the whole Mac (Docker's VM included). Peaks of the last kind are what can freeze
the machine.
"""

import threading
import time
from typing import Any

import mlx.core as mx
import psutil


class ResourceMonitor:
    def __init__(self, period: float = 0.25) -> None:
        self.period = period
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="monitor", daemon=True)
        self._process = psutil.Process()

    def __enter__(self) -> "ResourceMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while True:
            self.samples.append(
                {
                    "time": time.time(),
                    "mlx_gb": mx.get_active_memory() / 1e9,
                    "mlx_cache_gb": mx.get_cache_memory() / 1e9,
                    "process_gb": self._process.memory_info().rss / 1e9,
                    "system_used_gb": psutil.virtual_memory().used / 1e9,
                }
            )
            if self._stop.wait(self.period):
                return
