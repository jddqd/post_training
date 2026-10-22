"""Drop training conversations that overlap with the evaluation benchmark (BFCL).

A training question is contaminated if, after normalization (NFKC, lowercase, words only):
  - it is identical to a benchmark question, or
  - the n-grams it shares with one benchmark question cover at least `min_coverage` of the
    shorter of the two texts.

Why coverage instead of "shares one n-gram": measured on xLAM vs BFCL, a single shared 8-gram is
mostly generic phrasing ("can you help me find them") or number runs ("1 2 3 4 5 6 7 8"), while
the standard 13-gram rule misses short questions (BFCL's median is 21 words). Coverage >= 50% of
8-grams catches exact copies, copies embedded in a longer question, and benchmark questions
reused with other values ("area of a circle with a radius of 12.7 units" vs "... of 5 units").
It cannot catch paraphrases: that would need embeddings.

No pairwise comparison: benchmark n-grams go into an index (n-gram -> question ids), and each
training question only looks up its own n-grams.
"""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower())


def ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass(frozen=True)
class BenchmarkText:
    id: str
    text: str


@dataclass(frozen=True)
class Contamination:
    benchmark_id: str
    coverage: float  # 1.0 for an exact match

    def __str__(self) -> str:
        return f"{self.benchmark_id} ({self.coverage:.0%})"


def load_benchmark_texts(path: Path) -> list[BenchmarkText]:
    """Records written by eval/bfcl/export_questions.py."""
    with path.open() as f:
        return [BenchmarkText(r["id"], r["text"]) for r in map(json.loads, f) if r["text"]]


class Decontaminator:
    def __init__(self, texts: Sequence[BenchmarkText], n: int = 8, min_coverage: float = 0.5):
        self.n = n
        self.min_coverage = min_coverage
        self._ids = [t.id for t in texts]
        self._exact = {" ".join(words(t.text)): t.id for t in texts}
        self._ngrams = [ngrams(words(t.text), n) for t in texts]
        self._index: dict[tuple[str, ...], set[int]] = defaultdict(set)
        for i, grams in enumerate(self._ngrams):
            for gram in grams:
                self._index[gram].add(i)

    def check(self, text: str) -> Contamination | None:
        tokens = words(text)
        if (exact := self._exact.get(" ".join(tokens))) is not None:
            return Contamination(exact, 1.0)
        grams = ngrams(tokens, self.n)
        # Texts shorter than n words have no n-gram: only the exact match applies to them.
        shared = Counter(i for gram in grams for i in self._index.get(gram, ()))
        best: Contamination | None = None
        for i, count in shared.items():
            coverage = count / min(len(grams), len(self._ngrams[i]))
            if coverage >= self.min_coverage and (best is None or coverage > best.coverage):
                best = Contamination(self._ids[i], coverage)
        return best
