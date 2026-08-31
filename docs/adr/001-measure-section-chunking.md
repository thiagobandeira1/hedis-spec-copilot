# ADR-001: Measure/section-aware chunking with contextual headers

## Status

Accepted

## Date

2026-08-31

## Context

The corpus is a small set of CMS Star Ratings documents whose useful structure is rigid:
each measure (e.g. C14 COL) carries a description, numerator, denominator, exclusions,
timeline, and coding guidance. Questions target exactly those slices ("what are the
exclusions for COL?"), and the gold set labels answers at `(doc_id, measure_id, section)`
granularity. The embedder (ADR-002) has a 512-token context limit, and a mid-document
fragment like "Exclusions: members in hospice" is ambiguous without knowing which measure
and plan year it belongs to. Chunking must also be deterministic: the eval ratchet and the
index staleness stamp both assume identical input produces identical chunks.

## Decision

Chunk along document structure, never across it (`corpus/chunk.py`):

1. **Measure boundary first, then section blocks.** Parsing (ADR upstream of this file)
   yields `NormalizedDoc → MeasureBlock → DocSection(kind, heading, text, page)`. A chunk
   never spans two sections or two measures.
2. **A section stays one chunk when it fits** (≤ `chunk_max_tokens`, default 480);
   otherwise it splits *within* the section on sentence/bullet boundaries into 480-token
   windows with 64-token overlap. Delimiter-free tables and code lists are hard-split on
   word windows so no chunk can exceed the embedder's context.
3. **Contextual header on every chunk**:
   `[CMS 2026 Star Ratings Technical Notes | C14 COL ... | Exclusions | 2026]`. The header
   is prepended for embedding and shown to the LLM (`Chunk.embed_text`), restoring the
   identity a fragment loses.
4. **Content-hashed chunk ids** — `doc:measure:section:sha256(text)[:16]` — so
   re-chunking identical content yields identical ids; exact duplicates within one
   `(doc, measure, section)` collapse to a single chunk.
5. Token counts use a cheap heuristic: `max(words * 1.3, chars / 4)`, bounded both ways so
   dense numeric/tabular text is not undercounted.

## Alternatives considered

- **Fixed-size sliding windows** over raw text: simplest, but windows straddle measure
  boundaries, poisoning the measure filter and making section-level gold labels unusable.
- **LLM-driven semantic chunking**: non-deterministic, needs an API key at build time
  (breaks keyless CI, ADR-005), and adds nothing for documents this well structured.
- **One chunk per section, no splitting**: some sections exceed 512 tokens and would be
  silently truncated by the embedder.

## Consequences

- Retrieval, eval labels, and the alias-based measure filter all share the same
  `(doc, measure, section)` coordinate system; chunker refactors do not invalidate the
  gold set.
- Headers cost ~20–30 tokens per chunk of embedding and prompt budget — accepted, since
  they measurably help a small embedder disambiguate near-identical section text across
  measures and years.
- The token estimate is a heuristic, not the bge tokenizer. 480 + header leaves margin
  under 512, but pathological text could still estimate low; the chunker invariant tests
  (determinism, no boundary crossing, max size) are the guard, not the arithmetic.
- Overlapping split windows duplicate some sentences in the index; RRF fusion and the
  chunk-id tie-break keep this benign but it slightly inflates index size.
- Chunk ids change whenever text changes, so cross-revision chunk-id comparisons are
  meaningless by design — provenance comparisons must use `(doc_id, measure_id, section)`.
