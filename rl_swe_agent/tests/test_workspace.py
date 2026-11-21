import subprocess
import sys
from pathlib import Path

from rl_swe_agent.opencodeinstruct import CodeTask
from rl_swe_agent.workspace import PUBLIC_TESTS, SOLUTION, build_workspace

TESTS = [
    "assert total([1, 2]) == 3",
    "items = [4]; total(items); assert items == [4]",  # a compound test, as in the dataset
    "assert total([]) == 0",
    *[f"assert total([{i}]) == {i}" for i in range(7)],
]
TASK = CodeTask(
    id="t",
    problem="Sum a list.",
    reference="from typing import List\n\n\ndef total(xs: List[int]) -> int:\n    return sum(xs)\n",
    tests=TESTS,
    function="total",
    signature="def total(xs: List[int]) -> int:",  # `List` is never imported: must still load
    score=1.0,
    domain="generic",
)


def run_public_tests(root: Path, solution: str | None = None) -> subprocess.CompletedProcess[str]:
    workspace = build_workspace(TASK)
    workspace.write(root)
    if solution is not None:
        (root / SOLUTION).write_text(solution)
    return subprocess.run(
        [sys.executable, PUBLIC_TESTS], cwd=root, capture_output=True, text=True, check=False
    )


def test_three_public_tests_and_the_rest_hidden() -> None:
    workspace = build_workspace(TASK)
    assert workspace.public_tests == TESTS[:3]
    assert workspace.hidden_tests == TESTS[3:]
    assert set(workspace.files) == {SOLUTION, PUBLIC_TESTS}
    for hidden in workspace.hidden_tests:
        assert hidden not in workspace.files[PUBLIC_TESTS]


def test_the_starting_solution_is_the_signature_with_an_empty_body() -> None:
    stub = build_workspace(TASK).files[SOLUTION]
    assert "def total(xs: List[int]) -> int:\n    pass\n" in stub


def test_the_empty_solution_loads_and_fails_with_what_it_returned(tmp_path: Path) -> None:
    result = run_public_tests(tmp_path)
    assert result.returncode == 1
    assert "test 1 FAILED: assert total([1, 2]) == 3\n  got: None" in result.stdout
    assert "test 2 passed" in result.stdout  # nothing is returned, the list is unchanged
    assert result.stdout.strip().endswith("2 of 3 failed")


def test_a_correct_solution_passes(tmp_path: Path) -> None:
    result = run_public_tests(tmp_path, solution=TASK.reference)
    assert result.returncode == 0
    assert result.stdout.strip().endswith("OK: all public tests passed")


def test_an_exception_is_reported_with_its_type(tmp_path: Path) -> None:
    result = run_public_tests(tmp_path, solution="def total(xs):\n    return 1 / 0\n")
    assert result.returncode == 1
    assert "test 1 ERROR" in result.stdout
    assert "ZeroDivisionError: division by zero" in result.stdout
