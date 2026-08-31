"""Reranker seam. v1 ships a no-op; the eval harness is the instrument that would ever
justify a cross-encoder here (SPEC non-goal), so only the protocol exists."""

from collections.abc import Sequence
from typing import Protocol

from hedis_copilot.retrieval.types import ScoredChunk


class Reranker(Protocol):
    """Reorders fused candidates for a query; must not add or invent chunks."""

    def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]: ...


class NoopReranker:
    """Preserves fused order exactly — the v1 default."""

    def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return list(chunks)
