# Architecture Decision Records

Decisions that shaped `hedis-spec-copilot`, in the order they were made. Each record is
self-contained: the context that forced a choice, the choice, the alternatives rejected,
and the trade-offs accepted. The project spec (`docs/SPEC.md`) states *what* the system
is; these records preserve *why* it is built this way.

## Index

| ADR | Decision | Enforced / implemented in |
|---|---|---|
| [001](001-measure-section-chunking.md) | Measure/section-aware chunking with contextual headers and content-hashed ids | `corpus/chunk.py`, `corpus/models.py` |
| [002](002-fastembed-bge-small.md) | FastEmbed BAAI/bge-small-en-v1.5 (384-dim ONNX, pinned) as the embedder | `index/embed.py`, `config.Settings.embedding_model` |
| [003](003-hybrid-bm25-rrf.md) | Hybrid dense + BM25 retrieval fused with RRF (k=60), measure-filtered with starvation fallback | `retrieval/hybrid.py`, `retrieval/fusion.py` |
| [004](004-two-tier-corpus-license-posture.md) | Two-tier corpus: committed US-gov public domain vs fetch-at-build ©NCQA excerpts, license posture on every chunk | `corpus/manifest.py`, pytest license gates |
| [005](005-keyless-ci-two-tier-evals.md) | Keyless CI retrieval evals with a ratcheted baseline; real-LLM faithfulness local-only with stamped artifacts | `evals/`, CI workflow, `hedis report` |
| [006](006-citation-fail-closed.md) | Fail-closed citation validation: parse → regenerate once → refuse; disclaimer appended by code | `answer/citations.py`, `answer/chain.py` |

## Conventions

- **Format**: Status (Accepted / Superseded by ADR-XXX / Deprecated), Date, Context,
  Decision, Alternatives considered, Consequences — with honest trade-offs, not sales
  copy.
- **Numbering**: sequential, zero-padded (`NNN-short-slug.md`). Numbers are never reused.
- **Lifecycle**: ADRs are immutable history. A changed decision gets a *new* ADR that
  references and supersedes the old one; the old record keeps the context that made the
  original choice sensible at the time.
- **Ratchet coupling**: moving the committed eval baseline (`evals/baseline.json`)
  requires a PR that includes an ADR note explaining the change (see ADR-005).
