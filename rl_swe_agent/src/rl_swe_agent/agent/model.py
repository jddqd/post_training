"""mini-swe-agent's `Model`, for a policy that returns raw text (a local model, or a script).

DefaultAgent calls `query(messages)` and expects an assistant message whose `extra.actions`
holds the bash commands to run. A subclass only provides `generate(messages) -> text`; the
XML tool call is parsed here (agent/toolcall.py).

One change from mini-swe-agent: on a format error, its models drop the faulty response from the
conversation (only the error message is added). Here the response stays, followed by the error:
the model sees what it did wrong, and the episode remains one sequence of generated and
environment tokens, as RL needs.
"""

import time
from typing import Any

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages

from rl_swe_agent.agent.toolcall import parse_actions


class TextModel:
    def __init__(self, config: dict[str, Any]) -> None:
        """`config`: the `model` section of agent.yaml (observation and format error templates)."""
        self.config = config
        self.n_calls = 0

    def generate(self, messages: list[dict[str, Any]]) -> str:
        raise NotImplementedError

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        text = self.generate(messages)
        call_id = f"call_{self.n_calls}"
        self.n_calls += 1
        extra: dict[str, Any] = {"timestamp": time.time()}
        response = {"role": "assistant", "content": text, "extra": extra}
        try:
            actions = parse_actions(
                text, format_error_template=self.config["format_error_template"], call_id=call_id
            )
        except FormatError as error:
            raise FormatError(response, *error.messages) from None
        extra["actions"] = actions
        return response

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """One `tool` message per action: the command's output, in the observation template."""
        observations: list[dict[str, Any]] = format_toolcall_observation_messages(
            actions=message["extra"]["actions"],
            outputs=outputs,
            observation_template=self.config["observation_template"],
            template_vars=template_vars,
        )
        return observations

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {}

    def serialize(self) -> dict[str, Any]:
        return {"info": {"model_type": type(self).__name__, "model_calls": self.n_calls}}


class ScriptedModel(TextModel):
    """Answers a fixed list of responses, in order: tests the loop without a model."""

    def __init__(self, config: dict[str, Any], responses: list[str]) -> None:
        super().__init__(config)
        self.responses = responses

    def generate(self, messages: list[dict[str, Any]]) -> str:
        return self.responses[self.n_calls]
