import pytest

from rl_swe_agent.sandbox.verifier import IMAGE, Verdict, docker_ready, verify


def test_reward_is_one_only_when_every_test_passes() -> None:
    assert Verdict(["pass", "pass"]).reward == 1.0
    assert Verdict(["pass", "fail"]).reward == 0.0
    assert Verdict([]).reward == 0.0  # no test ran: never a success


@pytest.mark.skipif(not docker_ready(), reason=f"needs Docker and the {IMAGE} image")
def test_verify_in_docker() -> None:
    code = "def f(x):\n    return x + 1\n"
    verdict = verify(code, ["assert f(1) == 2", "assert f(1) == 3", "assert g(1)"])
    assert verdict.statuses == ["pass", "fail", "error: NameError"]
    assert verify("def f(:", ["assert f(1)"]).load_error.startswith("SyntaxError")
    endless = "def f():\n    while True:\n        pass\n"
    assert verify(endless, ["assert f()"], per_test_timeout=1).statuses == ["timeout"]
