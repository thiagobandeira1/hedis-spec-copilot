# ADR-007: Gate A is a degenerate-query guard, not the refusal mechanism

Status: Accepted
Date: 2026-09-01

## Context

The spec assigned refusal behavior to two gates: gate A (pre-LLM, a score floor on
retrieval, giving CI an LLM-free refusal path) and gate B (the grounding prompt's contract,
measured by the full LLM eval). Gate A was originally keyed to the fused RRF score, and its
accuracy on the refusal-trap slice was a headline retrieval metric.

Two empirical findings from the dev split (15 items, 3 refusal traps) changed this:

1. **RRF scores carry no absolute relevance signal.** They are rank-derived: the best fused
   score for answerable questions (median 0.0325) and refusal traps (median 0.0320) were
   statistically indistinguishable, because BM25 always ranks *something* first.
2. **Dense cosine similarity does not separate the traps either.** With
   `BAAI/bge-small-en-v1.5`, dev answerables scored 0.64–0.79 best-similarity and dev traps
   scored 0.65–0.78 — fully interleaved. This is not an artifact: good refusal traps are
   *deliberately plausible* healthcare questions ("what is plan X's star rating?",
   "enumerate the full COL exclusion value set"), which land semantically close to corpus
   content by construction. No threshold exists that refuses traps without refusing real
   questions.

## Decision

- Gate A keys off `RetrievalResult.best_dense_similarity` (the only absolute signal in the
  pipeline) with a **coarse floor of 0.35** — it catches gibberish, empty-corpus states,
  and wildly off-domain queries, and nothing subtler.
- Substantive refusals (out-of-corpus and licensed-content questions) are **gate B's job**:
  the frozen grounding prompt instructs the model to emit the exact code-owned refusal
  sentence, and the full LLM eval measures trap accuracy on that path.
- The keyless retrieval eval still reports gate-A trap refusals, expected ≈ 0.0, labeled as
  a degenerate-input guard — an honest number with context beats a deleted metric.

## Consequences

- CI cannot measure substantive refusal quality; that number only exists after a local
  `hedis eval --full` run with a real key. The README table marks it accordingly.
- The 0.35 floor is essentially never hit by real English healthcare questions; its tests
  exercise it with synthetic low-similarity results.
- If a future embedding upgrade (or a trained verifier head) produces measurable
  answerable/trap separation on the dev split, this decision should be revisited — the
  calibration script pattern in the repo history shows how to check.
