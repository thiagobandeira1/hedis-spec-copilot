"""In-memory BM25 leg over the same chunks the dense index holds (ADR-003).

Built at startup from the ``chunks.jsonl`` sidecar; the lexical leg is what catches exact
measure codes, thresholds, and other tokens a small dense embedder blurs.
"""

import re
from collections.abc import Iterable, Sequence

from rank_bm25 import BM25Okapi

from hedis_copilot.corpus.models import Chunk

_TOKEN_SPLIT_RE = re.compile(r"\W+")


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_SPLIT_RE.split(text.lower()) if token]


class Bm25Index:
    """BM25Okapi over chunk ``embed_text`` with the same filters as the dense leg."""

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._chunks: list[Chunk] = sorted(chunks, key=lambda c: c.chunk_id)
        self._bm25: BM25Okapi | None = (
            BM25Okapi([_tokenize(chunk.embed_text) for chunk in self._chunks])
            if self._chunks
            else None
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def query(
        self,
        text: str,
        k: int,
        measure_ids: list[str] | None = None,
        plan_year: int | None = None,
    ) -> list[tuple[str, float]]:
        """Lexical top-k as ``(chunk_id, bm25 score)``; filters applied before ranking.

        Zero-score chunks are dropped — a chunk sharing no query token earns no rank, so
        it cannot leak fusion credit. Ties break on chunk_id for determinism.
        """
        if self._bm25 is None or k <= 0:
            return []
        allowed: set[str] | None = set(measure_ids) if measure_ids is not None else None
        scores: Sequence[float] = self._bm25.get_scores(_tokenize(text))
        candidates: list[tuple[str, float]] = []
        for chunk, score in zip(self._chunks, scores, strict=True):
            if allowed is not None and chunk.measure_id not in allowed:
                continue
            if plan_year is not None and chunk.plan_year != plan_year:
                continue
            value = float(score)
            if value > 0.0:
                candidates.append((chunk.chunk_id, value))
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        return candidates[:k]
