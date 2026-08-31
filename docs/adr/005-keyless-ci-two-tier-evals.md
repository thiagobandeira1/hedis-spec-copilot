# ADR-005: Keyless CI with a two-tier eval harness

## Status

Accepted

## Date

2026-08-31

## Context

Portfolio RAG projects routinely undermine their own eval claims: numbers hand-typed into
a README, CI that needs a paid API secret (and therefore silently skips on forks), or
"evals" that only ever ran once on the author's machine. This project's stated goal is an
*honest* harness. Two facts make that achievable: retrieval quality metrics (hit@k,
recall@k, MRR, precision@k) are pure functions over ranked chunk lists and need no LLM;
and LangChain's model abstraction means every generation path can run under
`GenericFakeChatModel`. The expensive part — LLM-judged faithfulness — cannot be keyless,
so pretending otherwise was off the table.

## Decision

Split evaluation into two tiers with different execution homes:

- **Tier 1 — keyless, CI, every push.** No `ANTHROPIC_API_KEY` secret exists in the
  repository *at all* (`llm.py` raises a typed `ConfigError` if a real-model path is hit
  without one; CI injects fakes). CI builds the index from `corpus/committed/`, runs
  `hedis eval --retrieval` over the hand-labeled gold set, and **fails if recall@8 or MRR
  drops more than 0.02 absolute** below the committed, previously *measured*
  `evals/baseline.json`. The baseline is a ratchet: it moves only via a PR carrying an
  ADR note. Gate-A refusal accuracy on the trap slice runs in the same pass.
- **Tier 2 — real key, local, `hedis eval --full` (~$3–5/run).** A claude-opus-5 judge
  (deliberately stronger than and different from the sonnet answerer) scores per-claim
  faithfulness and citation validity. Results are committed as stamped artifacts:
  `evals/results/{date}-{gitsha}.json` carrying git_sha, both model ids,
  judge_prompt_sha256, embedding revision, `Settings.config_hash()`, dataset_hash,
  metrics, judge–human agreement, and token usage. `hedis review` forces a 15-item human
  spot-check; <80% judge–human agreement marks the run untrusted in the artifact.
- **README numbers are generated, never typed.** `hedis report` rewrites the README table
  from the latest artifact; CI verifies the table byte-matches it. If `config_hash` has
  drifted since the artifact was produced, LLM numbers render as STALE (visible) rather
  than blocking retrieval work.

## Alternatives considered

- **API secret in CI**: rejected outright (SPEC non-goal). Costs money per push, breaks
  on forks, and creates a secret-exfiltration surface in a public repo.
- **LLM-judged evals mocked in CI**: worse than nothing — green checks implying
  faithfulness coverage that does not exist.
- **Fixed absolute thresholds instead of a ratchet**: either too lax to catch regressions
  or an arbitrary bar nobody can justify; a measured baseline is self-calibrating.

## Consequences

- Anyone can fork, push, and get real retrieval metrics with zero secrets; the retrieval
  engineering claims are continuously re-verified.
- **CI cannot catch LLM-quality regressions.** A prompt change that hurts faithfulness
  surfaces only when someone pays for a full run — mitigated by config_hash staleness
  marking, but honestly a gap accepted in exchange for a zero-secret repo.
- Committed artifacts can lag the code; the stamp makes lag detectable, not impossible.
- The 0.02 ratchet tolerance absorbs embedding/ONNX jitter but will also absorb genuine
  regressions smaller than 0.02 — chosen as the noise floor, revisit if variance shrinks.
- Trust in Tier-2 numbers rests on the stamped provenance chain plus the human agreement
  check, i.e. on process, not on CI enforcement.
