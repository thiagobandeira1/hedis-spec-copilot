"""Shared retrieval contracts — the seam between retrieval, answering, and evals."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from hedis_copilot.corpus.models import Chunk


class ScoredChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    """Fused RRF score (rank-derived; comparable only within one retrieval)."""
    dense_rank: int | None = None
    bm25_rank: int | None = None


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    chunks: list[ScoredChunk]
    measure_filter: list[str] | None = None
    """Measure ids the alias router applied (None = unfiltered)."""
    used_fallback: bool = False
    """True when the filtered pass starved (<4 chunks) and the unfiltered union kicked in."""


class RetrieverLike(Protocol):
    """What the answer layer and eval runner depend on."""

    def retrieve(self, query: str, *, plan_year: int | None = None) -> RetrievalResult: ...
