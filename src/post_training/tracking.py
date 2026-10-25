"""Everything needed to reproduce a run, saved next to its outputs."""

import json
import netrc
import os
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


def check_wandb_credentials(config: ExperimentConfig) -> None:
    """Fail at startup if online wandb logging has no API key.

    Without one, `wandb.init` asks for it on the terminal: a run started in the background
    (nohup, a detached tmux) would wait forever, silently, after minutes of preparation.
    `wandb login` stores the key in ~/.netrc; WANDB_API_KEY works too.
    """
    if not config.wandb.enabled or config.wandb.mode != "online":
        return
    if os.environ.get("WANDB_API_KEY"):
        return
    try:
        path = os.environ.get("NETRC")  # None: ~/.netrc
        if netrc.netrc(path).authenticators("api.wandb.ai"):
            return
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    raise RuntimeError(
        "wandb is enabled but has no API key: run `wandb login` (or set WANDB_API_KEY), "
        "or pass wandb.mode=offline / wandb.enabled=false"
    )


def record_resume(run_dir: Path) -> None:
    """Append this resumption (command, code version, time) to the run's metadata.json."""
    path = run_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.setdefault("resumes", []).append(
        {
            "command": sys.argv,
            "git": git_state(),
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    path.write_text(json.dumps(metadata, indent=2))


def start_wandb(config: ExperimentConfig, run_dir: Path) -> Any:
    """Start a wandb run named after the run directory, holding the resolved config.

    Returns the run (None when disabled). TRL's Trainer, with `report_to="wandb"`, logs into this
    active run instead of creating its own: both scripts share the same naming and metadata.
    """
    if not config.wandb.enabled:
        return None
    import wandb  # noqa: PLC0415  (only needed when tracking is on)

    return wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=run_dir.name,
        # The run directory's name as the id: resuming a run (train_trl.py --resume) continues
        # the same wandb run instead of starting a new one.
        id=run_dir.name,
        resume="allow",
        mode=config.wandb.mode,
        dir=run_dir,
        config={**config.model_dump(mode="json"), "git": git_state()},
    )


def save_history(run_dir: Path, result: TrainResult) -> None:
    """history.jsonl: one line per training step; evals.jsonl: one line per validation."""
    for name, records in (("history", result.history), ("evals", result.evals)):
        with (run_dir / f"{name}.jsonl").open("w") as f:
            for record in records:
                f.write(json.dumps(asdict(record)) + "\n")
