# ADR-002: FastEmbed BAAI/bge-small-en-v1.5 as the embedding model

## Status

Accepted

## Date

2026-08-31

## Context

The project's hard constraints shape the embedder choice more than leaderboard rank does:

- **No API key in CI, ever** (ADR-005). CI rebuilds the index from `corpus/committed/` and
  recomputes retrieval metrics on every push, so embedding must run keyless, offline, and
  cheap on a GitHub-hosted ubuntu runner.
- **Determinism.** The eval ratchet compares recall/MRR against a committed baseline;
  embedding drift would make the gate meaningless. Hosted embedding APIs can change
  silently behind a model name.
- The corpus is small (~1,892 chunks from 4 documents), English-only, and heavy on exact
  domain tokens (measure codes, thresholds, ICD/CPT-like strings).

## Decision

Embed with **FastEmbed running BAAI/bge-small-en-v1.5** — 384-dimensional, ONNX,
CPU-only, with the model revision pinned. The model name is a `Settings` knob
(`embedding_model`) folded into `config_hash`, so switching models visibly re-stamps the
index and marks prior eval artifacts as computed under a different configuration. The bge
query/passage asymmetry (query prefix) is applied in the embedding layer and covered by a
unit test. Vectors live in embedded Chroma at `data/index/` (gitignored, rebuilt from
source). CI caches the ONNX weights via `actions/cache` so runs stay fast and hermetic.

## Alternatives considered

- **Hosted embeddings (Voyage, OpenAI, etc.)**: better raw quality, but requires a secret
  in CI — explicitly forbidden — and reproducibility depends on the vendor.
- **bge-base / bge-large or an instruct model**: measurably stronger, but slower on CPU
  runners and heavier to cache; for ~1,900 chunks with a measure filter in front, the
  small model plus BM25 was judged sufficient until the eval harness proves otherwise.
- **sentence-transformers runtime instead of FastEmbed**: pulls in torch (~GBs) for the
  same weights; FastEmbed's ONNX runtime is a fraction of the install and CI cost.

## Consequences

- Keyless CI can build the real index and run the real retrieval pipeline — the metrics
  in the README are produced by the same code path a user runs, not a mock.
- **A small embedder blurs exact tokens** (accepted risk in SPEC §10): "C14" vs "C15" or
  two nearby thresholds can embed near-identically. This is deliberately compensated
  twice — contextual headers (ADR-001) inject discriminating text, and hybrid BM25
  (ADR-003) matches exact tokens; the hybrid-vs-dense delta in the eval report is the
  measurement of how much this matters.
- 512-token model context is a hard ceiling that propagates backwards into the 480-token
  chunk budget (ADR-001).
- English-only model; acceptable because the corpus is English-only by construction.
- Pinning the revision means upstream model improvements arrive only via a deliberate PR
  that re-baselines the ratchet — slower to benefit, but never surprised.
