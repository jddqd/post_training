"""First look: can Qwen3.5 solve OpenCodeInstruct tasks, and do the tests hold?

Single-turn generation (no agent yet), greedy, one answer per task and per model, checked by the
verifier in Docker. The dataset reference is re-run too: its score in the dataset says whether
NVIDIA's run passed, ours says whether the tests are usable.

    uv run python scripts/try_models.py   # Docker (OrbStack) must be reachable
"""

import argparse
import json
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from rl_swe_agent.sandbox.verifier import verify
from rl_swe_agent.tasks.opencodeinstruct import (
    download_shard,
    extract_code,
    instruction,
    load_tasks,
    sample_by_score,
)

MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--per-score", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shard = download_shard(0, args.data_dir / "raw" / "opencodeinstruct")
    tasks = sample_by_score(load_tasks(shard), [1.0, 0.0], args.per_score, args.seed)
    results: dict[str, dict[str, Any]] = {
        t.id: {"score": t.score, "function": t.function, "tests": len(t.tests)} for t in tasks
    }
    for task in tasks:
        verdict = verify(task.reference, task.tests)
        results[task.id]["reference"] = verdict.statuses or [verdict.load_error]

    for name in MODELS:
        model, tokenizer = load(name)[:2]
        for task in tasks:
            prompt = tokenizer.apply_chat_template(  # type: ignore[no-untyped-call]
                [{"role": "user", "content": instruction(task)}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            answer = generate(
                model, tokenizer, prompt, max_tokens=args.max_tokens, sampler=make_sampler(0.0)
            )
            verdict = verify(extract_code(answer), task.tests)
            results[task.id][name] = {
                "answer": answer,
                "statuses": verdict.statuses,
                "load_error": verdict.load_error,
            }
            print(f"{name} {task.id[:8]} score={task.score}: {verdict.passed}/{len(task.tests)}")

    out = args.data_dir / "try" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'task':10} {'dataset':>7} {'ref (ours)':>10}", *(f"{m[5:]:>14}" for m in MODELS))
    for task in tasks:
        row = results[task.id]
        n = len(task.tests)
        cells = [row["reference"], *(row[m]["statuses"] for m in MODELS)]
        passed = [f"{statuses.count('pass')}/{n}" for statuses in cells]
        print(
            f"{task.id[:8]:10} {task.score:>7} {passed[0]:>10}", *(f"{p:>14}" for p in passed[1:])
        )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
