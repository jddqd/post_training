import yaml

from post_training.data.combine import Source, combine
from post_training.data.conversation import Conversation
from post_training.data.hub import combined_card


def conv(question: str) -> Conversation:
    return Conversation(
        messages=[{"role": "user", "content": question}, {"role": "assistant", "content": "A"}]
    )


def test_sources_are_unioned_split_by_split() -> None:
    a = Source("a", train=[conv("a1"), conv("a2")], val=[conv("a3")])
    b = Source("b", train=[conv("b1")], val=[conv("b2")])

    data = combine([a, b], seed=0)

    assert sorted(c.messages[0]["content"] for c in data.train) == ["a1", "a2", "b1"]
    assert sorted(c.messages[0]["content"] for c in data.val) == ["a3", "b2"]
    assert data.kept == {"a": {"train": 2, "val": 1}, "b": {"train": 1, "val": 1}}
    assert not data.dropped


def test_duplicates_kept_once_and_no_val_row_from_another_train_set() -> None:
    a = Source("a", train=[conv("same"), conv("leak")], val=[conv("a-val")])
    b = Source("b", train=[conv("same")], val=[conv("leak"), conv("b-val")])

    data = combine([a, b], seed=0)

    assert [c.messages[0]["content"] for c in data.train].count("same") == 1
    assert sorted(c.messages[0]["content"] for c in data.val) == ["a-val", "b-val"]
    assert data.dropped == {"b/train: duplicate": 1, "b/val: in a train set": 1}
    assert data.kept["b"] == {"train": 0, "val": 1}


def test_combination_is_deterministic_and_interleaves_sources() -> None:
    a = Source("a", train=[conv(f"a{i}") for i in range(50)], val=[])
    b = Source("b", train=[conv(f"b{i}") for i in range(50)], val=[])

    first, second = combine([a, b], seed=0), combine([a, b], seed=0)

    assert first.train == second.train
    assert {c.messages[0]["content"][0] for c in first.train[:20]} == {"a", "b"}


def test_combined_card_credits_every_source_with_its_license() -> None:
    def source(name: str, kind: str, dataset: str) -> dict[str, object]:
        return {
            "name": name,
            "source": kind,
            "dataset": dataset,
            "revision": "abc",
            "hub_repo": f"me/{name}",
            "kept": {"train": 10, "val": 1},
            "train_tokens": {"total": 1000},
        }

    report = {
        "config": {"seed": 0},
        "sources": [
            source("xlam", "xlam", "Salesforce/xlam-function-calling-60k"),
            source("toolace", "toolace", "Team-ACE/ToolACE"),
        ],
        "dropped": {},
        "categories": {"train": {"simple": 20}, "val": {"simple": 2}},
    }

    card = combined_card(report)

    header = yaml.safe_load(card.split("---")[1])
    assert header["license"] == "other"  # two licenses: each row keeps its source's
    assert header["license_name"] == "apache-2.0-and-cc-by-4.0"
    assert header["source_datasets"] == ["Salesforce/xlam-function-calling-60k", "Team-ACE/ToolACE"]
    assert "CC-BY-4.0 by Salesforce" in card and "Apache-2.0 by Team-ACE" in card
    assert "| toolace | 10 | 1 | 1,000 |" in card
