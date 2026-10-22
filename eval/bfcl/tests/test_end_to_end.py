"""End-to-end BFCL run on 5 prompts (one per category), against the local server.

    checkpoint -> serve_local.py (HTTP server) <- run_bfcl.py (bfcl generate + evaluate)
               -> summary.json

Uses a tiny randomly initialized Qwen3 saved like a training checkpoint: it checks that the
evaluation plumbing works and reports honestly (no failed generation), not the scores.
"""

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from post_training.training.checkpoint import save_checkpoint

EVAL_DIR = Path(__file__).resolve().parents[1]
CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]


def tiny_checkpoint(path: Path) -> Path:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        eos_token_id=151643,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    save_checkpoint(Qwen3ForCausalLM(config), tokenizer, path, end_of_turn="<|im_end|>")
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return int(s.getsockname()[1])


def wait_until_ready(url: str, server: subprocess.Popen[bytes], timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with code {server.returncode}")
        try:
            urllib.request.urlopen(f"{url}/models")
            return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"server not ready after {timeout}s")


def test_bfcl_end_to_end_on_five_prompts(tmp_path: Path) -> None:
    checkpoint = tiny_checkpoint(tmp_path / "checkpoint")
    port = free_port()
    url = f"http://localhost:{port}/v1"
    server = subprocess.Popen(
        [
            sys.executable,
            str(EVAL_DIR / "serve_local.py"),
            "--model",
            str(checkpoint),
            "--port",
            str(port),
            "--max-new-tokens",
            "8",  # a random model never stops on its own
        ],
    )
    try:
        wait_until_ready(url, server)
        subprocess.run(
            [
                sys.executable,
                str(EVAL_DIR / "run_bfcl.py"),
                "--name",
                "e2e",
                "--limit",
                "1",
                "--model-path",
                str(checkpoint),
                "--server-url",
                url,
                "--output-dir",
                str(tmp_path / "bfcl"),
            ],
            check=True,
        )
    finally:
        server.terminate()
        server.wait()

    [run_dir] = (tmp_path / "bfcl").glob("e2e-*")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["inference_errors"] == 0
    assert summary["served_model"] == str(checkpoint)

    # One generated answer per category, i.e. 5 prompts in total.
    results = [
        json.loads(line)
        for path in (run_dir / "result").rglob("*.json")
        for line in path.read_text().splitlines()
    ]
    assert sorted(r["id"] for r in results) == sorted(f"{c}_0" for c in CATEGORIES)
    assert "Python Simple AST" in summary["scores"]["data_non_live"]
