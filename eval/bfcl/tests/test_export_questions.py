import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # eval/bfcl is not a package

from export_questions import benchmark_questions


def test_exports_user_messages_of_every_category() -> None:
    records = benchmark_questions()

    by_id = {r["id"]: r for r in records}
    assert by_id["simple_python_0"]["text"].startswith("Find the area of a triangle")
    assert {"simple_python", "irrelevance", "live_multiple", "multi_turn_base"} <= {
        r["category"] for r in records
    }
    assert all(r["text"].strip() for r in records)
