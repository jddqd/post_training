"""Shared by the tests (importable: tests/ is on pytest's and mypy's path)."""

from rl_swe_agent.tasks.opencodeinstruct import CodeTask

DOUBLE = CodeTask(
    id="double-test",
    problem="Double a number.",
    reference="def double(x):\n    return 2 * x\n",
    tests=[f"assert double({x}) == {2 * x}" for x in (1, 0, -1, 3, 10, -7, 100, 5, 2, 8)],
    function="double",
    signature="def double(x):",
    score=1.0,
    domain="generic",
)
