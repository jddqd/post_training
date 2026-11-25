import pytest
from hydra.errors import ConfigCompositionException
from pydantic import ValidationError

from rl_swe_agent.config import PROJECT, compose


def test_the_defaults() -> None:
    config = compose()
    assert config.model.name == "Qwen/Qwen3.5-0.8B"
    assert config.model.quantize is None
    assert config.sampling.temperature == 1.0
    assert config.rollout.max_in_flight == 16
    assert config.tasks.path == PROJECT / "data/tasks/data/val.jsonl"


def test_groups_and_values_are_overridden() -> None:
    config = compose(["model=qwen3_5_2b", "sampling=qwen", "rollout.tasks_per_step=2"])
    assert config.model.name == "Qwen/Qwen3.5-2B"
    assert config.sampling.top_k == 20
    assert config.rollout.tasks_per_step == 2


def test_a_misspelt_key_is_an_error() -> None:
    with pytest.raises(ConfigCompositionException):  # Hydra: the key does not exist
        compose(["rollout.grup_size=4"])
    with pytest.raises(ValidationError, match="Extra inputs"):  # added on purpose: Pydantic
        compose(["+rollout.grup_size=4"])


@pytest.mark.parametrize(
    "override",
    ["rollout.max_tokens=8000", "model.quantize=3", "sampling.top_p=0", "tasks.split=dev"],
)
def test_invalid_values_fail_before_anything_is_loaded(override: str) -> None:
    with pytest.raises(ValidationError):
        compose([override])


def test_a_config_cannot_change_during_a_run() -> None:
    config = compose()
    with pytest.raises(ValidationError):
        config.rollout.group_size = 4
