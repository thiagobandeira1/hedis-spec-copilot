"""Reciprocal Rank Fusion of the dense and BM25 legs (ADR-003). Pure function, no I/O."""

from collections.abc import Sequence


def rrf(dense: Sequence[str], bm25: Sequence[str], k: int = 60) -> list[tuple[str, float]]:
    """Fuse two ranked chunk-id lists: score(id) = sum over legs of 1 / (k + rank).

    Ranks are 1-based within each leg. Output is sorted by fused score descending with a
    deterministic ascending chunk_id tie-break, so identical inputs always fuse
    identically regardless of dict ordering or float coincidences.
    """
    scores: dict[str, float] = {}
    for leg in (dense, bm25):
        for rank, chunk_id in enumerate(leg, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
