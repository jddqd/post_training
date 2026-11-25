"""The MLX policy, with Qwen3.5-0.8B (skipped when it is not in the Hugging Face cache)."""

from typing import Any

import mlx.core as mx
import pytest
from huggingface_hub import try_to_load_from_cache
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from mlx_lm import load

from rl_swe_agent.agent.mlx_model import IM_END, MLXModel, continuation
from rl_swe_agent.config import compose, mini_config
from rl_swe_agent.rollout.episode import Episode, run_episode
from rl_swe_agent.sandbox.verifier import docker_ready
from rl_swe_agent.tasks.opencodeinstruct import CodeTask

MODEL = "Qwen/Qwen3.5-0.8B"
pytestmark = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(MODEL, "config.json"), str), reason=f"needs {MODEL}"
)
CONFIG = compose(["rollout.step_limit=4"])
TASK = CodeTask(
    id="mlx-test",
    problem="Return the sum of a list of integers.",
    reference="def total(xs):\n    return sum(xs)\n",
    tests=[f"assert total({xs}) == {sum(xs)}" for xs in ([1, 2], [], [5], [-1, 1], [3] * 4)] * 2,
    function="total",
    signature="def total(xs):",
    score=1.0,
    domain="generic",
)
SYSTEM = {"role": "system", "content": "You are a programmer."}
USER = {"role": "user", "content": "List the files."}


@pytest.fixture(scope="module")
def policy() -> tuple[Any, Any]:
    model, tokenizer = load(MODEL)[:2]
    return model, tokenizer


def render(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    text: str = tokenizer.apply_chat_template(
        messages, tools=[BASH_TOOL], add_generation_prompt=True, enable_thinking=False,
        tokenize=False,
    )  # fmt: skip
    return text


def assistant(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}


def tool(content: str) -> dict[str, str]:
    return {"role": "tool", "content": content, "tool_call_id": "call_0"}


def test_continuation_is_the_templates_text(policy: tuple[Any, Any]) -> None:
    _, tokenizer = policy
    a, b = "Listing.\n<tool_call>ls</tool_call>", "<tool_call>pwd</tool_call>"
    first, second, third = tool("a.py"), tool("<output>\n/workspace\n</output>"), tool("done")
    messages = [SYSTEM, USER, assistant(a), first, assistant(b), second, third]
    incremental = (
        render(tokenizer, messages[:2])
        + a + IM_END + continuation([first])
        + b + IM_END + continuation([second, third])
    )  # fmt: skip
    assert incremental == render(tokenizer, messages)


def test_a_format_error_is_a_user_turn(policy: tuple[Any, Any]) -> None:
    _, tokenizer = policy
    error = {"role": "user", "content": "Format error: No tool calls found."}
    messages = [SYSTEM, USER, assistant("Thinking."), error]
    # Before a later user message the template drops the response's empty <think> block: only
    # what follows the response is compared (the history is never re-rendered).
    assert render(tokenizer, messages).endswith("Thinking." + IM_END + continuation([error]))


@pytest.fixture(scope="module")
def episode(policy: tuple[Any, Any]) -> Episode:
    if not docker_ready():
        pytest.skip("needs Docker and python:3.12-slim")
    mini = mini_config(CONFIG)
    mx.random.seed(0)
    model = MLXModel(*policy, mini["model"], CONFIG.sampling, CONFIG.rollout)
    return run_episode(TASK, model, mini)


def spans(mask: list[int]) -> list[tuple[int, int]]:
    """[start, end) of each run of generated tokens."""
    runs, start = [], None
    for i, m in enumerate([*mask, 0]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    return runs


def test_the_tokens_are_the_conversation(episode: Episode, policy: tuple[Any, Any]) -> None:
    result = episode
    _, tokenizer = policy
    assert len(result.ids) == len(result.mask) == len(result.logprobs)
    responses = [m["content"] for m in result.messages if m["role"] == "assistant"]
    generated = spans(result.mask)
    assert len(generated) == len(responses) == result.model_calls
    for (start, end), response in zip(generated, responses, strict=True):
        tokens = result.ids[start:end]
        if tokens[-1] in tokenizer.eos_token_ids:
            tokens = tokens[:-1]
        assert tokenizer.decode(tokens) == response
    first_prompt = render(tokenizer, result.messages[:2])
    assert tokenizer.decode(result.ids[: generated[0][0]]) == first_prompt


def test_logprobs_match_one_pass_over_the_whole_episode(
    episode: Episode, policy: tuple[Any, Any]
) -> None:
    """The cached, turn-by-turn generation and a single forward pass see the same tokens."""
    result = episode
    model, _ = policy
    logits = model(mx.array(result.ids[:-1])[None])[0].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    positions = [i for i, m in enumerate(result.mask) if m]
    recomputed = [logprobs[i - 1, result.ids[i]].item() for i in positions]
    sampled = [result.logprobs[i] for i in positions]
    differences = [abs(a - b) for a, b in zip(recomputed, sampled, strict=True)]
    assert sum(differences) / len(differences) < 0.05
    assert max(differences) < 0.5
