"""Minimal OpenAI-compatible completions server, to run BFCL on a laptop (transformers + MPS).

BFCL's local handlers format the prompt themselves and send it to `POST /v1/completions`, after
checking `GET /v1/models`. Their usual servers (vLLM, SGLang) need a CUDA GPU; this one runs
anywhere: connections are accepted concurrently (BFCL sends requests in parallel) but generations
run one at a time. Use it to test the evaluation plumbing on small subsets, not for full runs.

Usage:
    uv run --project eval/bfcl python eval/bfcl/serve_local.py --model Qwen/Qwen3-0.6B
"""

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# BFCL's default temperature is 0.001: treat anything this small as greedy decoding.
GREEDY_BELOW = 0.01


class Model:
    def __init__(self, name: str, max_new_tokens: int | None) -> None:
        self.name = name
        self.max_new_tokens = max_new_tokens
        self.lock = threading.Lock()  # one generation at a time on one device
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def complete(self, prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
            self.device
        )
        n_prompt = inputs["input_ids"].shape[1]
        sampling = temperature >= GREEDY_BELOW
        # BFCL asks for up to 4096 tokens. Capping makes runs faster but truncates long reasoning
        # (Qwen3 thinks by default), which lowers the score: only cap explicitly.
        limit = max_tokens if self.max_new_tokens is None else min(max_tokens, self.max_new_tokens)
        with self.lock:
            output = self.model.generate(
                **inputs,
                max_new_tokens=limit,
                do_sample=sampling,
                temperature=temperature if sampling else None,
                top_p=None,
                top_k=None,
            )
        new_tokens = output[0, n_prompt:]
        # Generation stops on generation_config.eos_token_id (e.g. <|im_end|> for our
        # checkpoints), not necessarily on tokenizer.eos_token (<|endoftext|> for a base model).
        stop_ids = self.model.generation_config.eos_token_id
        stop_ids = [stop_ids] if isinstance(stop_ids, int) else list(stop_ids or [])
        finished = len(new_tokens) > 0 and new_tokens[-1].item() in stop_ids
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.name,
            "choices": [
                {
                    "index": 0,
                    # Like vLLM's default: special tokens (EOS) removed, <think> kept.
                    "text": self.tokenizer.decode(new_tokens, skip_special_tokens=True),
                    "finish_reason": "stop" if finished else "length",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": n_prompt,
                "completion_tokens": len(new_tokens),
                "total_tokens": n_prompt + len(new_tokens),
            },
        }


def make_handler(model: Model) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": model.name, "object": "model"}]})
            else:
                self._send(404, {"error": f"unknown path {self.path}"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/completions":
                self._send(404, {"error": f"unknown path {self.path}"})
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            start = time.perf_counter()
            response = model.complete(
                request["prompt"], request.get("max_tokens", 512), request.get("temperature", 0.0)
            )
            usage = response["usage"]
            print(
                f"{usage['prompt_tokens']:>5} prompt + {usage['completion_tokens']:>4} generated "
                f"tokens in {time.perf_counter() - start:.1f}s",
                flush=True,
            )
            self._send(200, response)

        def log_message(self, format: str, *args: Any) -> None:
            pass  # one line per completion above is enough

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF id or local checkpoint directory")
    parser.add_argument("--port", type=int, default=1053)  # BFCL's default LOCAL_SERVER_PORT
    parser.add_argument(
        "--max-new-tokens", type=int, help="Cap generation length (faster, but biases scores)"
    )
    args = parser.parse_args()

    model = Model(args.model, args.max_new_tokens)
    print(f"serving {args.model} on {model.device} at http://localhost:{args.port}/v1", flush=True)
    # A thread per connection: a single-threaded server's short accept queue makes BFCL's
    # parallel requests time out. Generation itself is serialized by `Model.lock`.
    ThreadingHTTPServer(("localhost", args.port), make_handler(model)).serve_forever()


if __name__ == "__main__":
    main()
