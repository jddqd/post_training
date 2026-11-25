"""The policy: Qwen3.5 in MLX, token in / token out, behind mini-swe-agent's Model interface.

DefaultAgent hands over the conversation as text messages. Re-rendering it with the chat template
every turn would retokenize what the model generated: the tokens can differ from the sampled ones
(BPE is not unique) and the template rewrites the history (it trims messages, and drops the empty
`<think>` block of responses before a later user message). The training would then score tokens
the model never sampled. So the episode is kept as one list of token ids:

    ids      [system + task + prompt] [response 1] [observation 1 + prompt] [response 2] ...
    mask      0 ...                     1 ...        0 ...                    1 ...
    logprob   0 ...                     sampled      0 ...                    sampled

- the first prompt is the only full template rendering;
- each response is appended as sampled, with the log-probability of each token;
- what follows a response (observations, format errors, the next generation prompt) is written
  as the template would, and only that new text is tokenized;
- the KV cache is kept across turns: each turn feeds only the new tokens to the model.

The log-probabilities are those of the model's distribution (MLX's `generate_step`: before the
sampler), which is the sampling distribution at temperature 1 without top-p / top-k, as in RL.
"""

from typing import Any

import mlx.core as mx
from minisweagent.exceptions import LimitsExceeded
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from mlx.nn.layers.quantized import quantize
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from rl_swe_agent.agent.model import TextModel
from rl_swe_agent.config import ModelConfig, RolloutConfig, SamplingConfig

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
GENERATION_PROMPT = f"{IM_START}assistant\n<think>\n\n</think>\n\n"  # non-thinking mode


def continuation(messages: list[dict[str, Any]]) -> str:
    """Qwen3.5's template for the messages after a response (closed by <|im_end|>), then the
    generation prompt. Tool results are grouped in one user turn, as the template does."""
    text = "\n"
    for i, message in enumerate(messages):
        content = message["content"].strip()
        if message["role"] == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                text += f"{IM_START}user"
            text += f"\n<tool_response>\n{content}\n</tool_response>"
            if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                text += f"{IM_END}\n"
        elif message["role"] == "user":
            text += f"{IM_START}user\n{content}{IM_END}\n"
        else:
            raise ValueError(f"unexpected {message['role']} message after a response")
    return text + GENERATION_PROMPT


def load_policy(config: ModelConfig) -> tuple[Any, Any]:
    """The weights and tokenizer, shared by every episode (optionally quantized, in memory)."""
    model, tokenizer = load(config.name)[:2]
    if config.quantize is not None:
        quantize(model, group_size=64, bits=config.quantize)
        mx.eval(model.parameters())
    return model, tokenizer


class MLXModel(TextModel):
    """One episode's policy: create one per episode (the weights are shared, not copied)."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        templates: dict[str, Any],
        sampling: SamplingConfig,
        rollout: RolloutConfig,
    ) -> None:
        """`templates`: the `model` section of agent.yaml (observation, format errors)."""
        super().__init__(templates)
        self.model, self.tokenizer = model, tokenizer
        self.sampler = make_sampler(
            temp=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k
        )
        self.max_tokens = rollout.max_tokens
        self.max_context_tokens = rollout.max_context_tokens
        self.im_end = tokenizer.convert_tokens_to_ids(IM_END)
        self.cache = make_prompt_cache(model)
        self.n_messages = 0  # messages already in `ids`, the last response included
        self.last_response = ""
        self.pending: list[int] = []  # <|im_end|> owed by a response cut at max_tokens

    def generate(self, messages: list[dict[str, Any]]) -> str:
        prompt = self.prompt(messages)
        budget = min(self.max_tokens, self.max_context_tokens - len(self.ids) - len(prompt))
        if budget <= 0:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "ContextLimitExceeded",
                    "extra": {"exit_status": "ContextLimitExceeded", "submission": ""},
                }
            )
        self.append(prompt, generated=False)
        tokens: list[int] = []
        logprobs: list[float] = []
        steps = generate_step(
            mx.array(prompt), self.model, max_tokens=budget, sampler=self.sampler,
            prompt_cache=self.cache,
        )  # fmt: skip
        for token, distribution in steps:
            tokens.append(int(token))
            logprobs.append(distribution[token].item())  # type: ignore[arg-type]
            if tokens[-1] in self.tokenizer.eos_token_ids:
                break
        self.append(tokens, generated=True, logprobs=logprobs)
        self.pending = [] if tokens[-1] == self.im_end else [self.im_end]
        self.n_messages = len(messages) + 1  # DefaultAgent adds the response next
        end = -1 if tokens[-1] in self.tokenizer.eos_token_ids else len(tokens)
        self.last_response = str(self.tokenizer.decode(tokens[:end]))
        return self.last_response

    def prompt(self, messages: list[dict[str, Any]]) -> list[int]:
        """The new token ids: the whole first prompt, then what follows the last response."""
        if not self.ids:
            ids: list[int] = self.tokenizer.apply_chat_template(
                messages, tools=[BASH_TOOL], add_generation_prompt=True, enable_thinking=False
            )
            return ids
        response = messages[self.n_messages - 1]
        if response["role"] != "assistant" or response["content"] != self.last_response:
            raise RuntimeError("the conversation does not continue this model's last response")
        text = continuation(messages[self.n_messages :])
        return self.pending + list(self.tokenizer.encode(text, add_special_tokens=False))

    def append(
        self, tokens: list[int], generated: bool, logprobs: list[float] | None = None
    ) -> None:
        self.ids += tokens
        self.mask += [int(generated)] * len(tokens)
        self.logprobs += logprobs if logprobs is not None else [0.0] * len(tokens)
