"""Everything needed to reproduce a run, saved next to its outputs."""

import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from post_training.config import ExperimentConfig
from post_training.training.loop import TrainResult


def create_run_dir(config: ExperimentConfig) -> Path:
    run_dir = config.output_dir / f"{config.name}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def git_state() -> dict[str, Any]:
    """Commit hash and whether the working tree has uncommitted changes (None outside git)."""

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def save_run_metadata(run_dir: Path, config: ExperimentConfig, overrides: Sequence[str]) -> None:
    # The *resolved* config (file + overrides + defaults): re-running with it alone reproduces
    # the run, even if the defaults in the code change later.
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    )
    metadata = {
        "command": sys.argv,
        "overrides": list(overrides),
        "git": git_state(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def save_history(run_dir: Path, result: TrainResult) -> None:
    """history.jsonl: one line per training step; evals.jsonl: one line per validation."""
    for name, records in (("history", result.history), ("evals", result.evals)):
        with (run_dir / f"{name}.jsonl").open("w") as f:
            for record in records:
                f.write(json.dumps(asdict(record)) + "\n")
