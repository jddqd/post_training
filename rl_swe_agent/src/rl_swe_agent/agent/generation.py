"""Token generation for the policy: one turn of one episode, from its new tokens and its cache.

    generation = generator.generate(prompt, cache, max_tokens)
    generation.tokens, generation.logprobs, generation.cache   # the cache for the next turn
    generation.version                                          # of the weights that sampled it

The cache holds the episode so far (prompt and every generated token, the last one included),
so each turn feeds only its new tokens. Two generators:

- SequentialGenerator: in the caller's thread, one sequence at a time (mlx-lm's generate_step);
- BatchedGenerator: the turns of concurrent episodes (one thread each, DefaultAgent blocking on
  generate) are batched by one generation thread with mlx-lm's BatchGenerator (continuous
  batching: a sequence joins when its turn starts and leaves when it ends, its cache kept).
  Generation is limited by reading the weights: 8 sequences cost little more than one.

Every MLX computation of a BatchedGenerator stays in its thread.
"""

import json
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, generate_step
from mlx_lm.models.cache import ArraysCache, make_prompt_cache


@dataclass
class Generation:
    tokens: list[int]  # the stop token included, if one ended the turn
    logprobs: list[float]  # of each token under the model's distribution (before the sampler)
    cache: Any
    version: int  # of the weights: the behaviour policy, for importance sampling
    # time.time() of: the request, its entry into the batch, its first token (the prompt read),
    # its last token. The batched steps between are shared with the other sequences.
    submitted: float = 0.0
    inserted: float = 0.0
    first_token: float = 0.0
    finished: float = 0.0


class Generator(Protocol):
    """A context manager: `with make_generator(...) as generator:` (agent/mlx_model.py)."""

    def generate(self, prompt: list[int], cache: Any, max_tokens: int) -> Generation: ...

    def __enter__(self) -> "Generator": ...

    def __exit__(self, *exc: object) -> None: ...


class SequentialGenerator:
    def __init__(self, model: Any, sampler: Any, stop_tokens: set[int], version: int = 0) -> None:
        self.model, self.sampler, self.stop_tokens = model, sampler, stop_tokens
        self.version = version

    def __enter__(self) -> "SequentialGenerator":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def generate(self, prompt: list[int], cache: Any, max_tokens: int) -> Generation:
        start = time.time()
        cache = cache if cache is not None else make_prompt_cache(self.model)
        tokens: list[int] = []
        logprobs: list[float] = []
        first_token = start
        steps = generate_step(
            mx.array(prompt), self.model, max_tokens=max_tokens, sampler=self.sampler,
            prompt_cache=cache,
        )  # fmt: skip
        for token, distribution in steps:
            if not tokens:
                first_token = time.time()
            tokens.append(int(token))
            logprobs.append(distribution[token].item())  # type: ignore[arg-type]
            if tokens[-1] in self.stop_tokens:
                break
        return Generation(
            tokens, logprobs, cache, self.version, start, start, first_token, time.time()
        )


def detach(cache: list[Any]) -> list[Any]:
    """A finished sequence's cache, with arrays of its own: until then, each waiting episode kept
    the whole batch's cache alive (it waits for its command, then for its next turn).
    - DeltaNet states (ArraysCache) come out as slices of the batch's: copied here (17.8 MB kept
      per layer for one 1.05 MB sequence at batch 17, x 18 layers).
    - Attention K/V (KVCache) come out as lazy copies of a slice: not evaluated, they hold the
      batch's whole K/V (1.8 GB at 22 sequences of 6,600 tokens, per finished sequence: the
      memory then climbed by that much at each exit). Evaluated here.
    """
    for layer in cache:
        if isinstance(layer, ArraysCache):
            layer.cache = [None if a is None else mx.contiguous(a) for a in layer.cache]
    mx.eval([layer.state for layer in cache])  # copied now, so the batch's arrays can be freed
    return cache


@dataclass
class _Request:
    prompt: list[int]
    cache: Any
    max_tokens: int
    future: "Future[Generation]"
    submitted: float
    context: int = 0  # tokens in the cache plus the prompt: the attention span of the request
    inserted: float = 0.0
    first_token: float = 0.0


class BatchedGenerator:
    """Use as a context manager: the generation thread stops on exit."""

    def __init__(
        self,
        model: Any,
        sampler: Any,
        stop_tokens: set[int],
        batch_size: int,
        version: int = 0,
        prefill_batch_size: int = 8,
        trace: Path | None = None,
    ) -> None:
        """`batch_size` sequences decode together; at most `prefill_batch_size` prompts are read
        at once (their activations add up: 24 first prompts together ran out of GPU memory).
        `trace`: a JSON line per batch step with MLX's memory and what the step processed,
        written as it goes (it survives a crash, to find what makes memory peak)."""
        self.model, self.sampler, self.stop_tokens = model, sampler, stop_tokens
        self.batch_size, self.version = batch_size, version
        self.prefill_batch_size = min(prefill_batch_size, batch_size)
        self.error: BaseException | None = None
        self.trace = trace
        # (time.time(), sequences that got a token) per batched decoding step: the batch size
        # and throughput over time. Written by the generation thread only.
        self.steps: list[tuple[float, int]] = []
        self.requests: queue.Queue[_Request | None] = queue.Queue()
        self.thread = threading.Thread(target=self._serve, name="generation", daemon=True)
        self.thread.start()

    def generate(self, prompt: list[int], cache: Any, max_tokens: int) -> Generation:
        future: Future[Generation] = Future()
        offset = next((c.offset for c in cache or [] if hasattr(c, "offset")), 0)
        request = _Request(prompt, cache, max_tokens, future, time.time(), offset + len(prompt))
        del cache  # the batch copies it: the request's reference is dropped once it is inserted
        self.requests.put(request)
        while True:  # a request queued after the generation thread died would wait forever
            try:
                return future.result(timeout=1)
            except TimeoutError:
                if not self.thread.is_alive():
                    raise RuntimeError("the generation thread stopped") from self.error

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
            prefill_batch_size=self.prefill_batch_size,
        )
        running: dict[int, tuple[_Request, list[int], list[float]]] = {}
        trace = self.trace.open("a") if self.trace else None
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
                    request.inserted = time.time()
                    request.cache = None  # merged into the batch: free the episode's copy
                    running[uid] = (request, [], [])
                    block = False
                start = time.time()
                if trace is not None:
                    mx.reset_peak_memory()  # the trace records each step's own peak
                _, responses = batch.next()  # type: ignore[no-untyped-call]
                now = time.time()
                if trace is not None:
                    self._write_trace(trace, start, now, running)
                if responses:
                    self.steps.append((now, len(responses)))
                for response in responses:
                    request, tokens, logprobs = running[response.uid]
                    if not tokens:
                        request.first_token = now
                    tokens.append(int(response.token))
                    logprobs.append(response.logprobs[response.token].item())
                    if response.finish_reason is not None:
                        del running[response.uid]
                        request.future.set_result(
                            Generation(
                                tokens,
                                logprobs,
                                detach(response.prompt_cache),
                                self.version,
                                request.submitted,
                                request.inserted,
                                request.first_token,
                                now,
                            )
                        )
        except BaseException as error:
            self.error = error
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
            if trace is not None:
                trace.close()

    @staticmethod
    def _write_trace(
        trace: IO[str],
        start: float,
        end: float,
        running: dict[int, tuple[_Request, list[int], list[float]]],
    ) -> None:
        """Memory after the step and its peak, the sequences decoding, and the prompts being
        read (not yet a first token): [prompt tokens, context tokens] each."""
        prefilling = [[len(r.prompt), r.context] for r, tokens, _ in running.values() if not tokens]
        contexts = [r.context + len(tokens) for r, tokens, _ in running.values()]
        line = {
            "time": start,
            "seconds": end - start,
            "active_gb": mx.get_active_memory() / 1e9,
            "cache_gb": mx.get_cache_memory() / 1e9,
            "peak_gb": mx.get_peak_memory() / 1e9,
            "decoding": len(running) - len(prefilling),
            "prefilling": prefilling,
            "max_context": max(contexts, default=0),
            "sum_context": sum(contexts),
        }
        trace.write(json.dumps(line) + "\n")
        trace.flush()
