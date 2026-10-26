import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # eval/bfcl is not a package

from run_bfcl import bfcl_env, count_inference_errors, generate_command


def test_failed_generations_are_counted(tmp_path: Path) -> None:
    # A timeout is stored as a result string: BFCL would score it as a correct "no call".
    results = tmp_path / "Qwen_Qwen3-0.6B-FC" / "non_live"
    results.mkdir(parents=True)
    (results / "BFCL_v4_irrelevance_result.json").write_text(
        json.dumps({"id": "irrelevance_0", "result": "Error during inference: Request timed out."})
        + "\n"
        + json.dumps({"id": "irrelevance_1", "result": "I cannot help with that."})
        + "\n"
    )

    assert count_inference_errors(tmp_path) == 1


def args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values = {
        "model": "Qwen/Qwen3-0.6B-FC",
        "model_path": str(tmp_path),  # a local directory: no download
        "categories": "simple_python,irrelevance",
        "limit": None,
        "num_threads": 32,
        "server_url": None,
        "backend": "vllm",
    }
    return argparse.Namespace(**{**values, **overrides})


def test_generation_sends_concurrent_requests_to_the_server(tmp_path: Path) -> None:
    command = generate_command(args(tmp_path))

    assert command[command.index("--num-threads") + 1] == "32"
    assert command[command.index("--local-model-path") + 1] == str(tmp_path.resolve())
    assert command[-2:] == ["--backend", "vllm"]
    remote = generate_command(args(tmp_path, server_url="http://localhost:1053/v1", limit=5))
    assert "--skip-server-setup" in remote and "--run-ids" in remote


def test_vllm_server_does_not_jit_compile_flashinfer(tmp_path: Path) -> None:
    env = bfcl_env({"PATH": "/usr/bin"}, tmp_path)

    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"  # its JIT build needs nvcc and ninja
    assert env["BFCL_PROJECT_ROOT"] == str(tmp_path)
    assert (
        bfcl_env({"VLLM_USE_FLASHINFER_SAMPLER": "1"}, tmp_path)["VLLM_USE_FLASHINFER_SAMPLER"]
        == "1"
    )
