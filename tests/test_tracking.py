import json
from pathlib import Path

import yaml

from post_training.config import DataConfig, ExperimentConfig, ModelConfig
from post_training.tracking import create_run_dir, save_history, save_run_metadata
from post_training.training.loop import StepLog


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


def test_history_is_one_json_object_per_step(tmp_path: Path) -> None:
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

    save_history(tmp_path, history)

    lines = (tmp_path / "history.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [0, 1, 2]
