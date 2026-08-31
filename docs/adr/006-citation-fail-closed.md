# ADR-006: Fail-closed citation validation

## Status

Accepted

## Date

2026-08-31

## Context

The product promise is "answers with inline citations resolving to real passages." An LLM
under a grounding prompt still sometimes emits no citation markers, cites a passage
number that was never provided, or drifts into unsupported prose. In a compliance-adjacent
domain (measure eligibility, exclusions, cut points) a fluent uncited answer is *worse*
than no answer: it looks authoritative and cannot be checked. Prompting alone is a
probabilistic control; the guarantee has to live in code that runs on every response,
including under the fake models CI injects.

## Decision

Validation is a deterministic, code-owned layer after generation (`answer/citations.py`,
wired in the LCEL chain of `answer/chain.py`):

1. Context is presented as numbered blocks [1..8], each headed
   `chunk_id | org | doc | measure | year | URL`, so a marker `[n]` maps mechanically to
   one retrieved chunk.
2. A **pure-function citation parser** (property-tested, no LLM involved) extracts every
   `[n]` marker and checks it resolves to a provided block.
3. **Any unresolvable marker, or zero citations on a factual answer → regenerate once**,
   feeding the specific validation error back as a hint.
4. If the retry also fails validation, the response is **converted to the code-owned
   refusal template**. No uncited answer ever renders — fail-closed.
5. The demo-grade disclaimer footer is appended **by code on every path** — answers,
   refusals, retries — never delegated to the model.
6. The whole ladder is proven in CI with a scripted misbehaving `GenericFakeChatModel`
   (hallucinated citation → regenerate-once-then-refuse), so the guarantee is a tested
   behavior, not an aspiration.

This complements the two upstream gates: gate A refuses pre-LLM when the best fused
retrieval score sits below `refusal_score_floor` (no API call spent), and gate B is the
frozen prompt contract whose adherence the eval slices measure.

## Alternatives considered

- **Trust the prompt** ("always cite [n]"): fails silently exactly when it matters;
  unverifiable in keyless CI.
- **LLM-judged validation at answer time**: doubles latency and cost per answer, is
  non-deterministic, and cannot run keyless; judging belongs in the offline eval tier
  (ADR-005).
- **Unlimited regeneration retries**: unbounded cost and latency for a model that is
  likely to keep failing; one hinted retry captures most recoveries.

## Consequences

- The invariant "every rendered factual sentence traces to a real passage" is enforced by
  a parser, so it holds for any injected model — real, fake, or misbehaving.
- **False refusals are the accepted cost**: a substantively correct answer with malformed
  or missing markers is discarded after one retry. The eval harness tracks refusal
  accuracy so over-refusal shows up as a measured number, not an anecdote.
- Each validation failure spends up to one extra generation of tokens and latency.
- Marker validity is **syntactic** grounding only — `[3]` resolving to block 3 does not
  prove block 3 supports the claim. Semantic faithfulness is deliberately measured
  offline by the judge tier (ADR-005) rather than gated at answer time.
- The regeneration hint slightly couples the chain to parser error phrasing; kept stable
  by unit tests on the hint format.
