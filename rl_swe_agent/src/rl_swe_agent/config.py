"""Hyperparameters: composed by Hydra from configs/, validated here before anything is loaded.

    config = compose(["model=qwen3_5_2b", "rollout.group_size=4"])  # tests, notebooks
    config = validate(cfg)  # in a @hydra.main script

Unknown keys and inconsistent values fail at once, not after an hour of rollouts. The prompts and
sandbox options stay in mini-swe-agent's format (agent/agent.yaml); `mini_config` injects the
hyperparameters into it, so each value has a single source.
"""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from hydra import compose as hydra_compose
from hydra import initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

PROJECT = Path(__file__).resolve().parents[2]  # rl_swe_agent/
CONFIGS = PROJECT / "configs"
AGENT_YAML = Path(__file__).parent / "agent" / "agent.yaml"


class StrictModel(BaseModel):
    # frozen: a config never changes during a run; extra="forbid": a misspelt key is an error
    # instead of being ignored while the default value is used.
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(StrictModel):
    name: str  # Hugging Face repository, loaded by mlx-lm
    quantize: Literal[4, 8] | None = None  # bits; None: bf16 weights


class SamplingConfig(StrictModel):
    temperature: NonNegativeFloat = 1.0
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: NonNegativeInt = 0  # 0: off


class RolloutConfig(StrictModel):
    group_size: PositiveInt = 8
    tasks_per_batch: PositiveInt = 1
    step_limit: PositiveInt = 15
    max_consecutive_format_errors: PositiveInt = 3
    max_tokens: PositiveInt = 1024
    max_context_tokens: PositiveInt = 8000
    command_timeout: PositiveInt = 10

    @property
    def batch_size(self) -> int:
        return self.group_size * self.tasks_per_batch

    @model_validator(mode="after")
    def _a_response_fits_in_the_context(self) -> Self:
        if self.max_tokens >= self.max_context_tokens:
            raise ValueError("max_tokens must be smaller than max_context_tokens")
        return self


class TasksConfig(StrictModel):
    data_dir: Path  # the task set's Hub repository, cached there (relative to rl_swe_agent/)
    split: Literal["train", "val", "test"]
    n: PositiveInt | None = None  # None: the whole split
    seed: int = 0

    @property
    def root(self) -> Path:
        return self.data_dir if self.data_dir.is_absolute() else PROJECT / self.data_dir

    @property
    def path(self) -> Path:
        return self.root / "data" / f"{self.split}.jsonl"


class Config(StrictModel):
    model: ModelConfig
    sampling: SamplingConfig
    rollout: RolloutConfig
    tasks: TasksConfig
    seed: int = 0


def validate(cfg: DictConfig) -> Config:
    return Config.model_validate(OmegaConf.to_container(cfg, resolve=True))


def compose(overrides: list[str] | None = None) -> Config:
    """The config outside a Hydra script, with command-line style overrides."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return validate(hydra_compose("config", overrides=overrides or []))


def load_agent_yaml() -> dict[str, Any]:
    """agent.yaml as is (agent, environment, model): mini_config adds the hyperparameters."""
    agent: dict[str, Any] = yaml.safe_load(AGENT_YAML.read_text())
    return agent


def mini_config(config: Config) -> dict[str, Any]:
    """agent.yaml with the hyperparameters: what DefaultAgent, the sandbox and the model read.

    `template_vars` are extra variables for the templates, passed to agent.run().
    """
    mini = load_agent_yaml()
    rollout = config.rollout
    mini["agent"]["step_limit"] = rollout.step_limit
    mini["agent"]["max_consecutive_format_errors"] = rollout.max_consecutive_format_errors
    environment = mini["environment"]
    # Killed inside the container: mini-swe-agent's own timeout only stops the docker exec
    # client, and the command would keep running. Its timeout stays as a safety net.
    timeout = ["timeout", "-k", "2", str(rollout.command_timeout)]
    environment["interpreter"] = timeout + environment["interpreter"]
    environment["timeout"] = rollout.command_timeout + 10
    mini["template_vars"] = {"command_timeout": rollout.command_timeout}
    return mini
