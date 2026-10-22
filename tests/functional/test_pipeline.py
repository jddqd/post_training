"""End-to-end run of the training pipeline on 5 conversations, with both training scripts.

    xLAM-like rows -> prepare_xlam -> train/val JSONL -> scripts/train.py     -> run dir, checkpoint
                                                      -> scripts/train_trl.py -> run dir, checkpoint
                   -> generation from the checkpoint with a BFCL-format prompt

Uses the real tokenizer and chat template with a tiny randomly initialized Qwen3 (real vocab,
2 small layers): it checks the plumbing between the stages, not what the model learns.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from post_training.config import PrepareDataConfig
from post_training.data.conversation import load_jsonl, save_jsonl
from post_training.data.prepare import prepare_xlam
from post_training.data.prompt_format import bfcl_chat_template

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = "Qwen/Qwen3-0.6B-Base"
IM_END, ENDOFTEXT = 151645, 151643

WEATHER = {
    "name": "get_weather",
    "description": "Current temperature of a city",
    "parameters": {"city": {"type": "str", "description": "City name"}},
}
INTEGRATE = {"name": "integrate", "parameters": {"f": {"type": "Callable[[float], float]"}}}


def xlam_row(city: str, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "query": f"What's the weather in {city}?",
        "tools": json.dumps(tools or [WEATHER]),
        "answers": json.dumps([{"name": "get_weather", "arguments": {"city": city}}]),
    }


ROWS = [
    *(xlam_row(city) for city in ["Paris", "Lyon", "Tokyo", "Lima", "Oslo"]),
    xlam_row("Paris"),  # duplicate
    xlam_row("Rome", tools=[WEATHER, INTEGRATE]),  # unsupported parameter type
]


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(TOKENIZER)
    return tokenizer


def tiny_qwen3(path: Path, tok: PreTrainedTokenizerBase) -> Path:
    """A Qwen3 with the real vocabulary and tied embeddings, but 2 layers of width 32."""
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        eos_token_id=ENDOFTEXT,  # like Qwen3-0.6B-Base: stops on <|endoftext|> only
    )
    Qwen3ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)
    return path


@pytest.mark.parametrize("script", ["train.py", "train_trl.py"])
def test_prepare_train_checkpoint_generate(
    script: str, tok: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    # 1. Data preparation
    data = prepare_xlam(
        ROWS,
        tok,
        PrepareDataConfig(tokenizer=TOKENIZER, output_dir=tmp_path / "data", val_fraction=0.2),
    )
    assert dict(data.funnel) == {
        "source": 7,
        "valid": 6,
        "deduplicated": 5,
        "length_ok": 5,
        "train": 4,
        "val": 1,
    }
    train_file, val_file = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    save_jsonl(train_file, data.train)
    save_jsonl(val_file, data.val)
    assert load_jsonl(train_file) == data.train

    # 2. Training, through the real entry point
    config = {
        "name": "functional",
        "output_dir": str(tmp_path / "runs"),
        "model": {"name": str(tiny_qwen3(tmp_path / "tiny", tok))},
        "data": {"train_file": str(train_file), "val_file": str(val_file)},
        "train": {
            "max_steps": 3,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 1,
            "eval_every": 2,
        },
    }
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(yaml.safe_dump(config))
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--config", str(config_file)],
        check=True,
        cwd=REPO_ROOT,
    )

    # 3. Run directory
    [run_dir] = (tmp_path / "runs").glob("functional-*")
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["train"]["max_steps"] == 3
    assert "git" in json.loads((run_dir / "metadata.json").read_text())
    if script == "train.py":
        history = (run_dir / "history.jsonl").read_text().splitlines()
        assert [json.loads(line)["step"] for line in history] == [0, 1, 2]
        evals = [json.loads(line) for line in (run_dir / "evals.jsonl").read_text().splitlines()]
        assert [e["step"] for e in evals] == [0, 2, 3]  # baseline, every 2 steps, final
        assert all(e["num_tokens"] > 0 for e in evals)
    else:
        logs = [
            json.loads(line) for line in (run_dir / "trainer_log.jsonl").read_text().splitlines()
        ]
        assert [r["step"] for r in logs if "eval_loss" in r] == [0, 2, 3]  # same as our loop
        assert [r["step"] for r in logs if "loss" in r] == [1, 2, 3]  # Trainer steps start at 1

    # 4. Checkpoint: reloadable, and generation stops at the end of an assistant turn
    checkpoint = run_dir / "checkpoint"
    model = Qwen3ForCausalLM.from_pretrained(checkpoint)
    assert model.generation_config.eos_token_id == [IM_END, ENDOFTEXT]

    # 5. Generation from a BFCL-format prompt (the evaluation's input)
    reloaded_tok = AutoTokenizer.from_pretrained(checkpoint)
    conversation = data.val[0]
    prompt = reloaded_tok.apply_chat_template(
        conversation.messages[:1],
        tools=[dict(t) for t in conversation.tools],
        chat_template=bfcl_chat_template(reloaded_tok),
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(prompt, str)
    assert prompt.endswith("<|im_start|>assistant\n")
    inputs = reloaded_tok(prompt, add_special_tokens=False, return_tensors="pt")
    # transformers annotates `generate`'s `self` with a protocol mypy can't match: runtime is fine.
    output = model.generate(**inputs, max_new_tokens=5, do_sample=False)  # type: ignore[misc]
    assert isinstance(output, torch.Tensor)
    assert output.shape[1] > inputs["input_ids"].shape[1]
