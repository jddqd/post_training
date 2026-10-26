"""Run BFCL (generate + evaluate) on a model and summarize the scores.

BFCL is used as a pinned dependency through its CLI, never forked. Each run gets its own
directory (`BFCL_PROJECT_ROOT`) holding BFCL's raw results, scores and our summary.json.

Examples:
    # Mac: small subset, against the local server (start serve_local.py first, same weights)
    uv run --project eval/bfcl python eval/bfcl/run_bfcl.py --name base --limit 20 \\
        --model-path Qwen/Qwen3-0.6B-Base --server-url http://localhost:1053/v1

    # GPU (RunPod): baseline of the untrained base model (HF id), BFCL starts vLLM itself
    uv run --project eval/bfcl python eval/bfcl/run_bfcl.py --name base \\
        --model-path Qwen/Qwen3-0.6B-Base

    # GPU (RunPod): a fine-tuned checkpoint written by scripts/train.py
    uv run --project eval/bfcl python eval/bfcl/run_bfcl.py --name sft-run-1 \\
        --model-path outputs/sft-qwen3-base-xlam-.../checkpoint
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# The categories xLAM trains for (single-turn AST), plus irrelevance: xLAM has no "no call"
# example, so this is where we expect SFT to hurt.
DEFAULT_CATEGORIES = "simple_python,multiple,parallel,parallel_multiple,irrelevance"
REPO_ROOT = Path(__file__).resolve().parents[2]


def write_subset(project_root: Path, categories: list[str], limit: int) -> None:
    """First `limit` entries of each category, in BFCL's test_case_ids_to_generate.json format."""
    from bfcl_eval.utils import load_dataset_entry  # noqa: PLC0415  (needs BFCL_PROJECT_ROOT set)

    ids = {c: [entry["id"] for entry in load_dataset_entry(c)][:limit] for c in categories}
    (project_root / "test_case_ids_to_generate.json").write_text(json.dumps(ids, indent=2))


def resolve_model_path(model_path: str) -> Path:
    """A local directory as is, otherwise a Hugging Face id downloaded to the local cache.

    BFCL's `--local-model-path` needs a directory; the registry name (`--model`) only picks the
    handler, i.e. the prompt format and output parsing, not the weights.
    """
    if Path(model_path).is_dir():
        return Path(model_path).resolve()
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return Path(snapshot_download(model_path))


def bfcl(args: list[str], env: dict[str, str]) -> None:
    command = [sys.executable, "-m", "bfcl_eval", *args]
    print("$ bfcl", " ".join(args))
    subprocess.run(command, env=env, check=True)


def count_inference_errors(result_dir: Path) -> int:
    """Entries whose generation failed (timeout, server error...).

    BFCL scores them like any answer: a failed generation contains no function call, which
    counts as *correct* for irrelevance. A run with errors has meaningless scores.
    """
    errors = 0
    for path in result_dir.rglob("*.json"):
        for line in path.read_text().splitlines():
            errors += "Error during inference" in str(json.loads(line).get("result"))
    return errors


def served_model(server_url: str) -> str:
    """The model id the server reports: BFCL itself never checks which weights it is served."""
    with urllib.request.urlopen(f"{server_url.rstrip('/')}/models") as response:
        return str(json.load(response)["data"][0]["id"])


def read_scores(score_dir: Path) -> dict[str, dict[str, str]]:
    """BFCL's leaderboard CSVs, one row per model, keyed by file name."""
    scores = {}
    for path in sorted(score_dir.glob("data_*.csv")):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            scores[path.stem] = rows[0]
    return scores


def bfcl_env(base: dict[str, str], project_root: Path) -> dict[str, str]:
    """Environment for the BFCL commands (and the vLLM server BFCL starts)."""
    env = {**base, "BFCL_PROJECT_ROOT": str(project_root)}
    # vLLM's FlashInfer sampler is compiled at startup (JIT), which needs nvcc and ninja: absent
    # on a plain PyTorch image, so the server crashes before answering. The PyTorch sampler is
    # as fast for BFCL's greedy decoding. An explicit value in the environment wins.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return env


def generate_command(args: argparse.Namespace) -> list[str]:
    command = ["generate", "--model", args.model, "--test-category", args.categories]
    if args.limit:
        command.append("--run-ids")
    command += ["--local-model-path", str(resolve_model_path(args.model_path))]
    # Concurrent requests: vLLM batches them (continuous batching). BFCL's default sends one
    # at a time and leaves the GPU mostly idle.
    command += ["--num-threads", str(args.num_threads)]
    command += ["--skip-server-setup"] if args.server_url else ["--backend", args.backend]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Run name (output directory prefix)")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B-FC",
        help="BFCL registry name: chooses the handler, i.e. prompt format and output parsing",
    )
    parser.add_argument(
        "--model-path",
        required=True,  # without it BFCL loads the registry model's weights (the Instruct one)
        help="Weights to evaluate with that handler: checkpoint dir or HF id",
    )
    parser.add_argument("--categories", default=DEFAULT_CATEGORIES)
    parser.add_argument("--limit", type=int, help="Only the first N entries of each category")
    parser.add_argument("--server-url", help="Use an already running OpenAI-compatible server")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "bfcl")
    parser.add_argument(
        "--num-threads", type=int, default=32, help="Concurrent requests to the server"
    )
    parser.add_argument(
        "--allow-errors", action="store_true", help="Don't fail when some generations errored"
    )
    args = parser.parse_args()

    project_root = (args.output_dir / f"{args.name}-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    project_root.mkdir(parents=True)
    env = bfcl_env(dict(os.environ), project_root)
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)  # for write_subset's import
    categories = args.categories.split(",")

    evaluate = ["evaluate", "--model", args.model, "--test-category", args.categories]
    if args.limit:
        write_subset(project_root, categories, args.limit)
        evaluate.append("--partial-eval")  # score only the entries we generated
    if args.server_url:
        env["REMOTE_OPENAI_BASE_URL"] = args.server_url

    bfcl(generate_command(args), env)
    bfcl(evaluate, env)

    errors = count_inference_errors(project_root / "result")
    summary = {
        "model": args.model,
        "model_path": args.model_path,
        # With --server-url, the weights are whatever that server loaded: record it.
        "served_model": served_model(args.server_url) if args.server_url else args.model_path,
        "categories": categories,
        "limit": args.limit,
        "inference_errors": errors,
        "scores": read_scores(project_root / "score"),
    }
    (project_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nresults in {project_root}")
    for name, row in summary["scores"].items():
        shown = {k: v for k, v in row.items() if v not in ("", "N/A")}
        print(f"{name}: {shown}")
    if errors and not args.allow_errors:
        sys.exit(f"{errors} generations failed: the scores above are not meaningful.")


if __name__ == "__main__":
    main()
