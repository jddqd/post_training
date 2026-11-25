"""Token generation for the policy: one turn of one episode, from its new tokens and its cache.

    generation = generator.generate(prompt, cache, max_tokens)
    generation.tokens, generation.logprobs, generation.cache   # the cache for the next turn

The cache holds the episode so far (prompt and every generated token, the last one included),
so each turn feeds only its new tokens. Two generators:

- SequentialGenerator: in the caller's thread, one sequence at a time (mlx-lm's generate_step);
- BatchedGenerator: the turns of concurrent episodes (one thread each, DefaultAgent blocking on
  generate) are batched by one generation thread with mlx-lm's BatchGenerator (continuous
  batching: a sequence joins when its turn starts and leaves when it ends, its cache kept).
  Generation is limited by reading the weights: 8 sequences cost little more than one.

Every MLX computation of a BatchedGenerator stays in its thread.
"""

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, generate_step
from mlx_lm.models.cache import make_prompt_cache


@dataclass
class Generation:
    tokens: list[int]  # the stop token included, if one ended the turn
    logprobs: list[float]  # of each token under the model's distribution (before the sampler)
    cache: Any


class Generator(Protocol):
    """A context manager: `with make_generator(...) as generator:` (agent/mlx_model.py)."""

    def generate(self, prompt: list[int], cache: Any, max_tokens: int) -> Generation: ...

    def __enter__(self) -> "Generator": ...

    def __exit__(self, *exc: object) -> None: ...


class SequentialGenerator:
    def __init__(self, model: Any, sampler: Any, stop_tokens: set[int]) -> None:
        self.model, self.sampler, self.stop_tokens = model, sampler, stop_tokens

    def __enter__(self) -> "SequentialGenerator":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def generate(self, prompt: list[int], cache: Any, max_tokens: int) -> Generation:
        cache = cache if cache is not None else make_prompt_cache(self.model)
        tokens: list[int] = []
        logprobs: list[float] = []
        steps = generate_step(
            mx.array(prompt), self.model, max_tokens=max_tokens, sampler=self.sampler,
            prompt_cache=cache,
        )  # fmt: skip
        for token, distribution in steps:
            tokens.append(int(token))
            logprobs.append(distribution[token].item())  # type: ignore[arg-type]
            if tokens[-1] in self.stop_tokens:
                break
        return Generation(tokens, logprobs, cache)


@dataclass
class _Request:
    prompt: list[int]
    cache: Any
    max_tokens: int
    future: "Future[Generation]"


class BatchedGenerator:
    """Use as a context manager: the generation thread stops on exit."""

    def __init__(self, model: Any, sampler: Any, stop_tokens: set[int], batch_size: int) -> None:
        self.model, self.sampler, self.stop_tokens = model, sampler, stop_tokens
        self.batch_size = batch_size
        self.requests: queue.Queue[_Request | None] = queue.Queue()
        self.thread = threading.Thread(target=self._serve, name="generation", daemon=True)
        self.thread.start()

    def generate(self, prompt: list[int], cache: Any, max_tokens: int) -> Generation:
        future: Future[Generation] = Future()
        self.requests.put(_Request(prompt, cache, max_tokens, future))
        return future.result()

    def close(self) -> None:
        self.requests.put(None)
        self.thread.join()

    def __enter__(self) -> "BatchedGenerator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _serve(self) -> None:
        batch = BatchGenerator(
            self.model,
            stop_tokens=[[t] for t in self.stop_tokens],
            sampler=self.sampler,
            completion_batch_size=self.batch_size,
            prefill_batch_size=self.batch_size,
        )
        running: dict[int, tuple[_Request, list[int], list[float]]] = {}
        try:
            while True:
                # Wait for work when idle, else take what arrived and keep generating.
                block = not running
                while True:
                    try:
                        request = self.requests.get(block=block)
                    except queue.Empty:
                        break
                    if request is None:
                        return
                    (uid,) = batch.insert(
                        [request.prompt], max_tokens=[request.max_tokens], caches=[request.cache]
                    )
                    running[uid] = (request, [], [])
                    block = False
                _, responses = batch.next()  # type: ignore[no-untyped-call]
                for response in responses:
                    request, tokens, logprobs = running[response.uid]
                    tokens.append(int(response.token))
                    logprobs.append(response.logprobs[response.token].item())
                    if response.finish_reason is not None:
                        del running[response.uid]
                        request.future.set_result(
                            Generation(tokens, logprobs, response.prompt_cache)
                        )
        except BaseException as error:
            # Fail the episodes instead of leaving their threads waiting.
            failed = [request for request, _, _ in running.values()]
            while not self.requests.empty():
                if (request := self.requests.get()) is not None:
                    failed.append(request)
            for request in failed:
                request.future.set_exception(error)
            raise
        finally:
            batch.close()  # type: ignore[no-untyped-call]
