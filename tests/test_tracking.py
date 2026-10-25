import json
from pathlib import Path

import pytest
import yaml

from post_training.config import DataConfig, ExperimentConfig, ModelConfig, WandbConfig
from post_training.tracking import (
    check_wandb_credentials,
    create_run_dir,
    save_history,
    save_run_metadata,
    start_wandb,
)
from post_training.training.loop import EvalLog, StepLog, TrainResult


def make_config(tmp_path: Path) -> ExperimentConfig:
    train_file = tmp_path / "train.jsonl"
    train_file.touch()
    return ExperimentConfig(
        name="test",
        output_dir=tmp_path,
        model=ModelConfig(name="m"),
        data=DataConfig(train_file=train_file),
    )


def test_saved_config_reloads_to_the_same_config(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_dir = create_run_dir(config)

    save_run_metadata(run_dir, config, overrides=["train.lr=1e-4"])

    reloaded = ExperimentConfig.model_validate(
        yaml.safe_load((run_dir / "config.yaml").read_text())
    )
    assert reloaded == config
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["overrides"] == ["train.lr=1e-4"]
    assert set(metadata["git"]) == {"commit", "dirty"}


def test_history_and_evals_are_one_json_object_per_line(tmp_path: Path) -> None:
    history = [
        StepLog(
            step=i,
            loss=1.0 / (i + 1),
            lr=1e-5,
            grad_norm=1.0,
            num_trained_tokens=10,
            tokens_per_s=100.0,
        )
        for i in range(3)
    ]

    evals = [
        EvalLog(step=0, val_loss=2.0, num_tokens=100),
        EvalLog(step=3, val_loss=1.0, num_tokens=100),
    ]

    save_history(tmp_path, TrainResult(history=history, evals=evals))

    lines = (tmp_path / "history.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [0, 1, 2]
    lines = (tmp_path / "evals.jsonl").read_text().splitlines()
    assert [json.loads(line)["val_loss"] for line in lines] == [2.0, 1.0]


def test_wandb_is_off_by_default(tmp_path: Path) -> None:
    assert start_wandb(make_config(tmp_path), tmp_path) is None


def test_wandb_run_holds_the_resolved_config(tmp_path: Path) -> None:
    config = make_config(tmp_path).model_copy(
        update={"wandb": WandbConfig(enabled=True, mode="offline", project="tests")}
    )
    run_dir = create_run_dir(config)

    run = start_wandb(config, run_dir)  # offline: written locally, no network, no account
    try:
        assert run.name == run_dir.name
        assert run.config["train"]["lr"] == config.train.lr
        assert "git" in run.config
        run.log({"train/loss": 1.0}, step=0)
    finally:
        run.finish()
    assert (run_dir / "wandb").is_dir()


def test_online_wandb_without_api_key_fails_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "no-netrc"))
    online = make_config(tmp_path).model_copy(update={"wandb": WandbConfig(enabled=True)})

    with pytest.raises(RuntimeError, match="wandb login"):
        check_wandb_credentials(online)

    (tmp_path / "netrc").write_text("machine api.wandb.ai login user password secret\n")
    monkeypatch.setenv("NETRC", str(tmp_path / "netrc"))
    check_wandb_credentials(online)  # `wandb login` writes this entry
    offline = online.model_copy(update={"wandb": WandbConfig(enabled=True, mode="offline")})
    monkeypatch.setenv("NETRC", str(tmp_path / "no-netrc"))
    check_wandb_credentials(offline)  # no key needed offline
