"""nvidia/OpenCodeInstruct as code tasks: a function to write, checked by its unit tests.

Each row has a problem statement (`input`), an LLM solution (`output`), about ten `assert` lines
(`unit_tests`) and the fraction of those asserts the solution passed in NVIDIA's run
(`average_test_score`). The tests are LLM-written too: a score of 0 can mean a hard problem, a
wrong reference or broken tests, hence our own re-run of the reference (sandbox/verifier.py).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "nvidia/OpenCodeInstruct"
SHARD = "data/train-{index:05d}-of-00050.parquet"
TASKS_REPO = "jddqd/opencodeinstruct-rl-tasks"  # our task set (scripts/build_dataset.py)

_CALLED = re.compile(r"assert\s+(?:not\s+)?([A-Za-z_]\w*)\(")
_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_OPEN_BLOCK = re.compile(r"```(?:python)?\n(.*)\Z", re.DOTALL)  # never closed: answer cut off


@dataclass(frozen=True)
class CodeTask:
    id: str
    problem: str
    reference: str  # the dataset's solution (code only)
    tests: list[str]  # one `assert` statement each
    function: str  # the function every test calls
    signature: str  # its `def` line in the reference, given to the model
    score: float  # dataset's average_test_score of the reference
    domain: str


def download_shard(index: int, data_dir: Path) -> Path:
    path = hf_hub_download(REPO, SHARD.format(index=index), repo_type="dataset", local_dir=data_dir)
    return Path(path)


def download_split(split: str, data_dir: Path) -> Path:
    """train, val or test of our published task set: data_dir/data/<split>.jsonl."""
    path = hf_hub_download(
        TASKS_REPO, f"data/{split}.jsonl", repo_type="dataset", local_dir=data_dir
    )
    return Path(path)


def read_tasks(path: Path) -> list[CodeTask]:
    with path.open() as f:
        return [CodeTask(**json.loads(line)) for line in f]


def extract_code(text: str) -> str:
    """The last code block of an answer (an unclosed one if cut off), else the whole answer."""
    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        return str(blocks[-1]).strip()
    opened = _OPEN_BLOCK.search(text)
    return (opened.group(1) if opened else text).strip()


def to_task(row: dict[str, object]) -> CodeTask | None:
    """A task whose tests all call one top-level function of the reference, else None."""
    tests = [t.strip() for t in json.loads(str(row["unit_tests"])) if t.strip()]
    called = {m for t in tests for m in _CALLED.findall(t)}
    if not tests or len(called) != 1:
        return None  # classes, helpers, several functions or no plain call: not in this first set
    (function,) = called
    reference = extract_code(str(row["output"]))
    signature = re.search(rf"^def {function}\(.*?\).*?:$", reference, re.MULTILINE | re.DOTALL)
    if signature is None:
        return None
    return CodeTask(
        id=str(row["id"]),
        problem=str(row["input"]),
        reference=reference,
        tests=tests,
        function=function,
        signature=signature.group(0),
        score=float(str(row["average_test_score"])),
        domain=str(row["domain"]),
    )


def load_tasks(shard: Path) -> list[CodeTask]:
    rows = pq.read_table(shard).to_pylist()
    return [task for row in rows if (task := to_task(row)) is not None]
