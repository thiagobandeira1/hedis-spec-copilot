"""Hybrid retriever: alias route -> filtered dense + BM25 legs -> RRF -> top final_k.

Implements :class:`~hedis_copilot.retrieval.types.RetrieverLike`. When the alias filter
starves retrieval (<4 fused chunks — a routed measure may not exist in the requested
year), both legs rerun unfiltered and the union keeps filtered hits first, flagged via
``used_fallback`` so the UI/evals can see the router overruled itself.
"""

import re
from collections import Counter
from collections.abc import Mapping

from langchain_core.embeddings import Embeddings

from hedis_copilot.config import Settings
from hedis_copilot.corpus.models import Chunk
from hedis_copilot.index.bm25 import Bm25Index
from hedis_copilot.index.embed import FastEmbedEmbeddings
from hedis_copilot.index.store import ChromaStore, chunks_path, load_chunks_jsonl, verify_stamp
from hedis_copilot.retrieval.aliases import AliasRouter, build_alias_table, entries_from_chunks
from hedis_copilot.retrieval.fusion import rrf
from hedis_copilot.retrieval.rerank import NoopReranker, Reranker
from hedis_copilot.retrieval.types import RetrievalResult, ScoredChunk

_FALLBACK_MIN_CHUNKS = 4

_YEAR_RE = re.compile(r"\b(20\d{2})\b")

#: Queries that compare across years must see every year — no default filter.
_CROSS_YEAR_RE = re.compile(
    r"\b(chang\w*|renam\w*|differ\w*|previous|prior|compar\w*|versus|vs\.?|"
    r"new in|retired|no longer|used to|year.over.year)\b",
    re.IGNORECASE,
)


class _Legs:
    """One dense+BM25 pass: fused ranking plus per-leg ranks for debug provenance."""

    def __init__(
        self,
        fused: list[tuple[str, float]],
        dense_ranks: dict[str, int],
        bm25_ranks: dict[str, int],
        best_dense_distance: float | None,
    ) -> None:
        self.fused = fused
        self.dense_ranks = dense_ranks
        self.bm25_ranks = bm25_ranks
        self.best_dense_distance = best_dense_distance


class HybridRetriever:
    """The retrieval engine behind `hedis ask`, the Streamlit app, and the eval runner."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: ChromaStore,
        bm25: Bm25Index,
        embedder: Embeddings,
        chunks: Mapping[str, Chunk],
        router: AliasRouter | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._bm25 = bm25
        self._embedder = embedder
        self._chunks = dict(chunks)
        self._router = router or AliasRouter(
            build_alias_table(entries_from_chunks(self._chunks.values()))
        )
        self._reranker: Reranker = reranker or NoopReranker()
        # The default year for unqualified queries: the most recent plan_year that carries a
        # substantial share of the corpus (a full Technical Notes), so a 3-chunk forward-look
        # doc (2027 measure list) can never become "the default year".
        year_counts = Counter(chunk.plan_year for chunk in self._chunks.values())
        total = sum(year_counts.values()) or 1
        substantial = [y for y, n in year_counts.items() if n / total >= 0.05]
        self._default_year: int | None = max(substantial) if substantial else None
        self._indexed_years = set(year_counts)

    @classmethod
    def from_settings(cls, settings: Settings) -> "HybridRetriever":
        """Open a built index, refusing loudly (StaleIndexError) if it is stale/missing."""
        verify_stamp(settings)
        chunks = load_chunks_jsonl(chunks_path(settings.index_dir))
        return cls(
            settings,
            store=ChromaStore(settings.index_dir),
            bm25=Bm25Index(chunks.values()),
            embedder=FastEmbedEmbeddings(settings.embedding_model),
            chunks=chunks,
        )

    def _run_legs(
        self,
        embedding: list[float],
        query: str,
        measure_ids: list[str] | None,
        plan_year: int | None,
    ) -> _Legs:
        dense = self._store.query(
            embedding, self._settings.dense_k, measure_ids=measure_ids, plan_year=plan_year
        )
        bm25 = self._bm25.query(
            query, self._settings.bm25_k, measure_ids=measure_ids, plan_year=plan_year
        )
        dense_ids = [chunk_id for chunk_id, _ in dense]
        bm25_ids = [chunk_id for chunk_id, _ in bm25]
        return _Legs(
            fused=rrf(dense_ids, bm25_ids, k=self._settings.rrf_k),
            dense_ranks={chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)},
            bm25_ranks={chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, start=1)},
            best_dense_distance=min((distance for _, distance in dense), default=None),
        )

    def _infer_year(self, query: str) -> int | None:
        """Explicit year in the query > cross-year smell (no filter) > index default."""
        mentioned = [int(y) for y in _YEAR_RE.findall(query) if int(y) in self._indexed_years]
        if len(mentioned) == 1:
            return mentioned[0]
        if mentioned or _CROSS_YEAR_RE.search(query):
            return None
        return self._default_year

    def retrieve(self, query: str, *, plan_year: int | None = None) -> RetrievalResult:
        effective_year = plan_year if plan_year is not None else self._infer_year(query)
        measure_ids = self._router.route(query, effective_year) or None
        embedding = self._embedder.embed_query(query)
        legs = self._run_legs(embedding, query, measure_ids, effective_year)
        fused = list(legs.fused)
        dense_ranks = dict(legs.dense_ranks)
        bm25_ranks = dict(legs.bm25_ranks)
        best_distance = legs.best_dense_distance

        used_fallback = False
        if measure_ids is not None and len(fused) < _FALLBACK_MIN_CHUNKS:
            # The measure filter starved the result; union in the unfiltered pass,
            # filtered hits first. The year filter stays applied — the year is a setting.
            used_fallback = True
            unfiltered = self._run_legs(embedding, query, None, effective_year)
            present = {chunk_id for chunk_id, _ in fused}
            fused.extend(pair for pair in unfiltered.fused if pair[0] not in present)
            for chunk_id, rank in unfiltered.dense_ranks.items():
                dense_ranks.setdefault(chunk_id, rank)
            for chunk_id, rank in unfiltered.bm25_ranks.items():
                bm25_ranks.setdefault(chunk_id, rank)
            if unfiltered.best_dense_distance is not None and (
                best_distance is None or unfiltered.best_dense_distance < best_distance
            ):
                best_distance = unfiltered.best_dense_distance

        candidates = [
            ScoredChunk(
                chunk=self._chunks[chunk_id],
                score=score,
                dense_rank=dense_ranks.get(chunk_id),
                bm25_rank=bm25_ranks.get(chunk_id),
            )
            for chunk_id, score in fused
            if chunk_id in self._chunks
        ]
        top = self._reranker.rerank(query, candidates)[: self._settings.final_k]
        return RetrievalResult(
            query=query,
            chunks=top,
            measure_filter=measure_ids,
            used_fallback=used_fallback,
            plan_year_applied=effective_year,
            best_dense_similarity=None if best_distance is None else 1.0 - best_distance,
        )
