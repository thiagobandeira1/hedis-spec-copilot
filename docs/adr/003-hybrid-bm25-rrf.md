# ADR-003: Hybrid dense + BM25 retrieval fused with Reciprocal Rank Fusion

## Status

Accepted

## Date

2026-08-31

## Context

HEDIS/Star Ratings questions mix two query styles that favor opposite retrievers.
Paraphrased natural-language questions ("who gets left out of the colonoscopy measure?")
need semantic matching; exact-token questions ("what is the 4-star CBP cut point?",
"C14", "SUPD") need lexical matching that a small 384-dim embedder demonstrably blurs
(ADR-002, SPEC §10). The corpus is small enough that both indexes are cheap: Chroma is
embedded and gitignored, and rank_bm25 builds in-memory at startup in well under a
second. A deterministic alias router (acronym table + Cxx/Dxx regex) already narrows most
queries to a measure filter before either retriever runs.

## Decision

Run both retrievers under the **identical measure filter** and fuse by rank
(`retrieval/hybrid.py`, `retrieval/fusion.py`; knobs in `config.Settings`):

1. Dense top-20 (`dense_k`) and BM25 top-20 (`bm25_k`) over `Chunk.embed_text`.
2. If the filtered passes return **fewer than 4 chunks combined**, automatically re-run
   unfiltered and union the results, flagging `RetrievalResult.used_fallback = True`.
3. Fuse with **RRF**: `score(c) = Σ 1/(60 + rank_leg(c))` with `rrf_k = 60`, ties broken
   deterministically by `chunk_id`.
4. Keep top-8 (`final_k`, ~3.5K prompt tokens), grouped by measure then canonical section
   order for the answer prompt.
5. Each `ScoredChunk` records `dense_rank` and `bm25_rank` so the Streamlit debug
   expander and the eval report can show per-leg behavior; the hybrid-vs-dense-only delta
   is a first-class metric in `hedis eval --retrieval`.

## Alternatives considered

- **Dense-only**: simplest, but gives up exact-token recall precisely where the domain is
  hardest (codes, cut points). Kept as the eval comparison arm rather than the product.
- **Weighted score fusion (e.g. 0.5·dense + 0.5·bm25)**: requires normalizing
  incommensurable score scales and tuning weights per corpus; RRF is scale-free,
  parameter-light, and robust — the standard choice when no tuning data exists yet.
- **Cross-encoder reranker**: deferred by design. A `Reranker` protocol seam exists
  (no-op in v1); the eval harness is the instrument that would justify paying its latency
  and complexity cost.

## Consequences

- Exact-token and paraphrase queries are both served, and the README reports the measured
  hybrid-vs-dense delta rather than asserting the architecture helps.
- **RRF scores are rank-derived**, comparable only within a single retrieval (documented
  on `ScoredChunk.score`). The pre-LLM refusal gate's `refusal_score_floor` therefore
  works on fused-rank evidence, not calibrated probabilities, and had to be calibrated
  empirically on labeled negatives — a weaker signal than a true relevance score.
- Two indexes must agree on the chunk set; the shared build stamp {embedding model,
  chunker version, manifest hash} refuses a stale index rather than serving skew.
- BM25 is rebuilt in-memory at every startup — fine at ~1,900 chunks, a rethink if the
  corpus grows orders of magnitude.
- The <4-chunk fallback trades filter precision for recall on alias-router misses; the
  `used_fallback` flag keeps that trade visible in evals and the UI.
