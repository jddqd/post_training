import json
from dataclasses import asdict
from pathlib import Path

from rl_swe_agent.tasks.opencodeinstruct import (
    CodeTask,
    extract_code,
    instruction,
    read_tasks,
    sample_by_score,
    to_task,
)

REFERENCE = "def add(a, b):\n    return a + b\n"


def row(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "x",
        "input": "Add two numbers.",
        "output": f"Here:\n```python\n{REFERENCE}```\nDone.",
        "unit_tests": json.dumps(["\nassert add(1, 2) == 3\n", "\nassert not add(0, 0)\n"]),
        "average_test_score": "1.0",
        "domain": "generic",
    }
    return base | changes


def test_extract_code_takes_the_last_block_or_the_whole_text() -> None:
    assert extract_code("a\n```python\nx = 1\n```\nb\n```\ny = 2\n```") == "y = 2"
    assert extract_code("x = 1\n") == "x = 1"
    assert extract_code("Sure:\n```python\ndef f():\n    return 1") == "def f():\n    return 1"


def test_to_task_keeps_one_called_function_with_its_signature() -> None:
    task = to_task(row())
    assert task is not None
    assert task.function == "add"
    assert task.signature == "def add(a, b):"
    assert task.reference == REFERENCE.strip()
    assert task.tests == ["assert add(1, 2) == 3", "assert not add(0, 0)"]
    assert task.score == 1.0


def test_to_task_rejects_several_functions_or_a_missing_definition() -> None:
    two = json.dumps(["assert add(1, 2) == 3", "assert sub(2, 1) == 1"])
    assert to_task(row(unit_tests=two)) is None
    assert to_task(row(output="```python\ndef plus(a, b):\n    return a + b\n```")) is None
    assert to_task(row(unit_tests=json.dumps(["assert Adder().add(1, 2) == 3"]))) is None


def test_sample_by_score_is_reproducible_and_balanced() -> None:
    tasks = [
        CodeTask(str(i), "", "", [], "f", "def f():", score, "generic")
        for i, score in enumerate([1.0, 0.0] * 10)
    ]
    first = sample_by_score(tasks, [1.0, 0.0], 3, seed=0)
    assert first == sample_by_score(tasks, [1.0, 0.0], 3, seed=0)
    assert [t.score for t in first] == [1.0] * 3 + [0.0] * 3


def test_instruction_gives_the_expected_signature() -> None:
    task = to_task(row())
    assert task is not None
    assert "def add(a, b):" in instruction(task)


def test_read_tasks_reads_back_the_published_format(tmp_path: Path) -> None:
    task = to_task(row())
    assert task is not None
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(asdict(task)) + "\n")
    assert read_tasks(path) == [task]
