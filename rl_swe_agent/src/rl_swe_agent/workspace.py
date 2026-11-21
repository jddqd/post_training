"""A code task as the small project the agent works in.

    solution.py      the expected signature with an empty body: the agent writes the function
    test_public.py   the first N_PUBLIC asserts (usually the statement's examples), runnable with
                     `python test_public.py`, with a readable report the agent can act on

The other asserts stay hidden: they never enter the agent's container. The reward runs all the
tests (public and hidden) on the final solution.py in a separate container (verifier.py).
"""

from dataclasses import dataclass
from pathlib import Path

from rl_swe_agent.opencodeinstruct import CodeTask

N_PUBLIC = 3
SOLUTION = "solution.py"
PUBLIC_TESTS = "test_public.py"

# Annotations are never evaluated: a signature like `def f(x: List[int])` stays importable
# without the reference's imports (which could hint at the solution).
SOLUTION_TEMPLATE = """from __future__ import annotations


{signature}
    pass
"""

# What the agent sees when it runs the public tests: each test on its own (one failure does not
# hide the others), the value actually returned for a failed comparison, the error otherwise.
PUBLIC_TESTS_TEMPLATE = '''"""Public tests for solution.py: python test_public.py"""

import ast
import sys
import traceback

import solution

TESTS = {tests}


def returned(test, namespace):
    """The left side of the final comparison, re-evaluated: what the function gave."""
    tree = ast.parse(test)
    *setup, last = tree.body
    if not (isinstance(last, ast.Assert) and isinstance(last.test, ast.Compare)):
        return None
    exec(compile(ast.Module(body=setup, type_ignores=[]), "<test>", "exec"), namespace)
    return repr(eval(compile(ast.Expression(last.test.left), "<test>", "eval"), namespace))[:300]


def main():
    failed = 0
    for number, test in enumerate(TESTS, 1):
        try:
            exec(test, dict(vars(solution)))
            print(f"test {{number}} passed")
        except AssertionError:
            failed += 1
            print(f"test {{number}} FAILED: {{test}}")
            try:
                got = returned(test, dict(vars(solution)))
            except Exception:
                got = None
            if got is not None:
                print(f"  got: {{got}}")
        except Exception:
            failed += 1
            print(f"test {{number}} ERROR: {{test}}")
            print("  " + traceback.format_exc(limit=-1).strip().replace("\\n", "\\n  "))
    print("OK: all public tests passed" if not failed else f"{{failed}} of {{len(TESTS)}} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
'''


@dataclass(frozen=True)
class Workspace:
    task: CodeTask
    files: dict[str, str]  # path relative to the workspace root -> content
    public_tests: list[str]
    hidden_tests: list[str]

    def write(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for name, content in self.files.items():
            (root / name).write_text(content)


def build_workspace(task: CodeTask, n_public: int = N_PUBLIC) -> Workspace:
    public, hidden = task.tests[:n_public], task.tests[n_public:]
    tests = "[\n" + "".join(f"    {test!r},\n" for test in public) + "]"
    return Workspace(
        task=task,
        files={
            SOLUTION: SOLUTION_TEMPLATE.format(signature=task.signature),
            PUBLIC_TESTS: PUBLIC_TESTS_TEMPLATE.format(tests=tests),
        },
        public_tests=public,
        hidden_tests=hidden,
    )
