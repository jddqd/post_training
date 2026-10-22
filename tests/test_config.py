from pathlib import Path

import pytest
from pydantic import ValidationError

from post_training.config import (
    PrepareDataConfig,
    TrainConfig,
    apply_overrides,
    load_config,
    load_yaml_config,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

MINIMAL_YAML = """
name: test
model:
  name: some/model
data:
  train_file: {train_file}
train:
  max_steps: 10
  warmup_steps: 2
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    train_file = tmp_path / "train.jsonl"
    train_file.touch()
    path = tmp_path / "config.yaml"
    path.write_text(MINIMAL_YAML.format(train_file=train_file))
    return path


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_config_in_the_repo_is_valid(path: Path, tmp_path: Path) -> None:
    # Data files are not versioned (CI doesn't have them): point the paths at an empty file so
    # that everything else in the config is still validated.
    placeholder = tmp_path / "data.jsonl"
    placeholder.touch()
    load_config(path, [f"data.train_file={placeholder}", f"data.val_file={placeholder}"])


@pytest.mark.parametrize(
    "path", sorted((CONFIGS_DIR / "data").glob("*.yaml")), ids=lambda p: p.name
)
def test_every_data_config_in_the_repo_is_valid(path: Path, tmp_path: Path) -> None:
    placeholder = tmp_path / "questions.jsonl"  # data files are not versioned
    placeholder.touch()
    load_yaml_config(PrepareDataConfig, path, [f"decontamination_file={placeholder}"])


def test_loads_values_and_fills_defaults(config_file: Path) -> None:
    config = load_config(config_file)

    assert config.model.name == "some/model"
    assert config.train.max_steps == 10
    assert config.train.lr == TrainConfig().lr  # default
    assert config.data.train_file.name == "train.jsonl"


def test_overrides_take_precedence_and_are_typed(config_file: Path) -> None:
    config = load_config(config_file, ["train.lr=5e-5", "train.max_steps=20", "name=other"])

    assert config.train.lr == 5e-5
    assert isinstance(config.train.lr, float)  # "5e-5" is a string in YAML 1.1; pydantic coerces
    assert config.train.max_steps == 20
    assert config.name == "other"


def test_apply_overrides_does_not_mutate_its_input() -> None:
    data = {"train": {"lr": 1.0}}

    result = apply_overrides(data, ["train.lr=2.0"])

    assert result == {"train": {"lr": 2.0}}
    assert data == {"train": {"lr": 1.0}}


# ---------------------------------------------------------------- errors caught at second 0


def test_typo_in_a_key_is_rejected(config_file: Path) -> None:
    with pytest.raises(ValidationError, match="learning_rate"):
        load_config(config_file, ["train.learning_rate=1e-4"])


def test_typo_in_a_section_is_rejected(config_file: Path) -> None:
    with pytest.raises(ValidationError, match="trian"):
        load_config(config_file, ["trian.lr=1e-4"])


@pytest.mark.parametrize(
    "override",
    ["train.lr=-1e-5", "train.lr=abc", "train.max_steps=0", "train.min_lr_ratio=1.5"],
)
def test_invalid_values_are_rejected(config_file: Path, override: str) -> None:
    with pytest.raises(ValidationError):
        load_config(config_file, [override])


def test_warmup_must_be_shorter_than_training(config_file: Path) -> None:
    with pytest.raises(ValidationError, match="warmup_steps"):
        load_config(config_file, ["train.warmup_steps=10"])


def test_malformed_override_is_rejected(config_file: Path) -> None:
    with pytest.raises(ValueError, match=r"section\.key=value"):
        load_config(config_file, ["train.lr"])


def test_cannot_override_inside_a_scalar(config_file: Path) -> None:
    with pytest.raises(ValueError, match="not a section"):
        load_config(config_file, ["name.sub=1"])


def test_config_is_immutable(config_file: Path) -> None:
    config = load_config(config_file)

    with pytest.raises(ValidationError, match="frozen"):
        config.train.lr = 1.0  # type: ignore[misc]


def test_missing_train_file_is_rejected_at_load_time(config_file: Path) -> None:
    with pytest.raises(ValidationError, match="train_file"):
        load_config(config_file, ["data.train_file=does/not/exist.jsonl"])


def test_run_name_must_be_usable_as_a_directory_name(config_file: Path) -> None:
    with pytest.raises(ValidationError, match="name"):
        load_config(config_file, ["name=../escape"])
