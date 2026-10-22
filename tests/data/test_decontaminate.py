import json
from pathlib import Path

from post_training.data.decontaminate import (
    BenchmarkText,
    Decontaminator,
    load_benchmark_texts,
    words,
)

BENCHMARK = [
    BenchmarkText("simple_python_9", "Calculate the area of a circle with a radius of 5 units."),
    BenchmarkText(
        "live_multiple_655",
        "Hi, I'm looking for Pop songs from the album 'Warrior' "
        "that were released in 2012. Can you help me find them?",
    ),
    BenchmarkText("irrelevance_1", "Hello?"),
]


def test_normalization_ignores_case_punctuation_and_unicode_forms() -> None:
    assert words("Café, déjà-vu!") == words("CAFÉ DÉJÀ vu")


def test_exact_copy_is_contaminated() -> None:
    hit = Decontaminator(BENCHMARK).check("calculate the AREA of a circle with a radius of 5 units")

    assert hit is not None
    assert (hit.benchmark_id, hit.coverage) == ("simple_python_9", 1.0)


def test_copy_embedded_in_a_longer_question_is_contaminated() -> None:
    hit = Decontaminator(BENCHMARK).check(
        "What is the area of a circle with a radius of 5 units? Also, find the median of 1, 4, 9."
    )

    assert hit is not None
    assert hit.benchmark_id == "simple_python_9"


def test_benchmark_question_reused_with_other_values_is_contaminated() -> None:
    hit = Decontaminator(BENCHMARK).check(
        "Calculate the area of a circle with a radius of 12.7 units."
    )

    assert hit is not None
    assert hit.benchmark_id == "simple_python_9"
    assert 0.5 <= hit.coverage < 1.0


def test_shared_generic_phrasing_is_not_contaminated() -> None:
    # Shares the 8-gram-ish tail "can you help me find them" but covers little of either text.
    question = "I'm looking for all KFC locations in California. Can you help me find them?"

    assert Decontaminator(BENCHMARK).check(question) is None


def test_short_texts_only_match_exactly() -> None:
    decontaminator = Decontaminator(BENCHMARK)

    assert decontaminator.check("hello") is not None  # exact (normalized)
    assert decontaminator.check("Hello there, how are you?") is None


def test_threshold_is_respected() -> None:
    question = "Calculate the area of a circle with a radius of 12.7 units."

    assert Decontaminator(BENCHMARK, min_coverage=1.0).check(question) is None


def test_loads_the_exported_questions(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"id": "simple_python_0", "category": "simple_python", "text": "Q?"}) + "\n"
    )

    assert load_benchmark_texts(path) == [BenchmarkText("simple_python_0", "Q?")]
