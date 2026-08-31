# SPEC — hedis-spec-copilot (P2)

> Status: **approved v1** · Produced by a 3-lens design panel (RAG architecture, eval
> methodology, product/compliance) + two web-research agents that verified every corpus URL,
> merged by an adversarial synthesis. Key decisions land as ADRs in `docs/adr/`.

## 1. Purpose

A RAG assistant over **exclusively public** HEDIS / CMS Medicare Star Ratings measure
documents that answers eligibility, exclusion, coding, threshold, and timeline questions with
**inline citations resolving to real passages** (URL + doc + measure + section + retrieval
date visible in the UI). Portfolio goals: LangChain competence via a typed, testable LCEL
pipeline; retrieval-quality engineering (hybrid dense+BM25 with measured deltas); and an
**honest two-tier eval harness** — retrieval metrics recomputed keylessly in CI with a
ratcheted baseline, real-LLM faithfulness runs local-only with committed, stamped artifacts.
README numbers are generated from artifacts, never hand-typed.

## 2. Goals

1. Cited answers from a public corpus; citation cards carry full provenance.
2. Typed LCEL pipeline: retrieve → numbered context → ChatAnthropic → deterministic citation
   validation → code-appended disclaimer. The chat model is injected as `BaseChatModel`, so
   every path runs under `GenericFakeChatModel` in keyless CI.
3. Retrieval engineering: measure/section-aware chunking with contextual headers, hybrid
   dense+BM25 fused with RRF, deterministic measure-alias filter; hit@k / recall@k / MRR /
   precision@k on a hand-labeled gold set, hybrid vs dense-only reported side by side.
4. Two-tier evals: CI = keyless retrieval metrics + ratchet gate; local `hedis eval --full` =
   faithfulness + citation validity + refusal accuracy with a stronger judge model, committed
   as stamped JSON; `hedis report` regenerates the README table from the latest artifact.
5. Licensing hygiene as **tested behavior**: two-tier corpus (committed US-gov public domain
   vs fetch-at-build copyrighted), provenance on every chunk, pytest license gates, explicit
   refusal when a question requires NCQA's licensed full specification.
6. P6 quality bar: uv, src layout, mypy --strict, ruff, pytest, ADRs, hardened .gitignore,
   zero-secret ubuntu CI, LangSmith env-gated, 3-command quickstart (keyless mode works).

## 3. Non-goals

- No reproduction/vendoring/wholesale paraphrase of NCQA's licensed HEDIS Technical
  Specifications (Volume 2), value set directories, or PQA value sets. Requests for
  exhaustive exclusion/code lists → refusal + pointer to NCQA licensing.
- Not a measure-computation engine; no member-level verdicts; no patient data; every answer
  carries a machine-appended demo-grade disclaimer.
- No hosted deployment/auth/server vector DB — embedded Chroma, localhost Streamlit,
  gitignored index rebuilt from source.
- No trained/cross-encoder reranker in v1 (a `Reranker` protocol seam exists; the eval
  harness is the instrument that would justify one). No LLM chunking or LLM query routing.
- No agentic multi-hop, no history-based query rewriting (single-turn RAG), no query-time web
  search — corpus fixed at build time by an explicit URL whitelist.
- **No CI use of ANTHROPIC_API_KEY** — no repository secret exists at all.
- v1 corpus = Medicare Part C & D Star Ratings public documents + NCQA public summary
  excerpts. Payer quick-refs and Medicaid Core Set specs are manifest-ready v1.1 extensions.

## 4. Architecture

CLI-driven (`hedis` console script); offline steps deterministic.

1. **FETCH** — `hedis fetch` reads `corpus/manifest.yaml` (pydantic-validated), downloads
   only whitelisted URLs (httpx, browser User-Agent — cms.gov 403s generic bot UAs), verifies
   pinned sha256 (mismatch = hard fail → deliberate manifest update when CMS revises a PDF),
   captures Content-Disposition filename + retrieval date. fetch_at_build files land only in
   gitignored `corpus/fetched/`.
2. **PARSE** — pypdf for CMS PDFs (508-compliant Technical Notes extract cleanly), CSV
   renderer for cut-point tables, BeautifulSoup for NCQA summary HTML → normalized MeasureDoc
   JSON with heading breadcrumbs + provenance, committed under `corpus/committed/` (Tier A).
3. **CHUNK** — measure-boundary-first, then section blocks (description | numerator |
   denominator | exclusions | timeline | coding | general); a section stays one chunk ≤480
   tokens else recursive split 480/64 within the section, never across boundaries (bge
   512-token limit). Contextual header per chunk
   (`[CMS 2026 Star Ratings Technical Notes | C14 COL … | Exclusions | 2026]`) +
   content-hashed chunk_id.
4. **INDEX** — `hedis build`: FastEmbed BAAI/bge-small-en-v1.5 (384-dim ONNX CPU, pinned
   revision) → embedded Chroma at `data/index/` (gitignored); BM25 (rank_bm25) built
   in-memory at startup. Index stamped {embedding model, chunker version, manifest hash};
   refused if stale.
5. **RETRIEVE** — deterministic alias router (acronym table + Cxx/Dxx regex) → measure
   filter; dense top-20 + BM25 top-20 under identical filter (auto unfiltered fallback+union
   if <4 chunks); RRF fusion (k=60, chunk_id tie-break) → top-8 (~3.5K tokens), grouped by
   measure then canonical section order. plan_year defaults latest (UI-overridable).
6. **GENERATE** — LCEL: numbered blocks [1..8] headed chunk_id | org | doc | measure | year |
   URL → ChatAnthropic (default claude-sonnet-5, temperature 0, env-overridable) under a
   frozen grounding prompt requiring inline [n] citations.
7. **VALIDATE** — pure-function citation parser; unresolvable marker or zero citations →
   regenerate once with error hint → else convert to refusal (**fail-closed: no uncited
   answer ever renders**). Disclaimer footer appended by code on every path.

Refusal gate A (pre-LLM): best fused score below a floor calibrated on labeled negatives →
code-owned refusal template, no API call. Gate B: the prompt contract, measured by eval
slices. Streamlit and CLI `ask` share one `AnswerService`.

## 5. Corpus (verified sources)

Two commit tiers in `corpus/manifest.yaml`
(`{doc_id, title, publisher, source_url, sha256, content_disposition_filename,
retrieval_date, plan_year, doc_type, license_posture, commit_policy, measures_covered,
parser_id}`), enforced by pytest license gates.

**Tier A — COMMITTED (US-gov public domain, 17 U.S.C. §105; normalized text committed, raw
PDFs cached gitignored, sha256-pinned):**

| Doc | URL | Why |
|---|---|---|
| Medicare 2026 Part C & D Star Ratings Technical Notes (211pp) | cms.gov/files/document/2026-star-ratings-technical-notes.pdf | **Backbone.** Verified per-measure description/numerator/denominator/exclusions/timeframe for every target measure (CBP p56, EED p52, GSD p53, BCS p38, COL pp39–40, SPC p63, SUPD p104, readmissions p62, adherence pp95–99) |
| Medicare 2025 Technical Notes (214pp) | cms.gov/files/document/2025-star-ratings-technical-notes.pdf | Prior year → cross-year answers (HBD→GSD rename etc.) |
| 2027 Star Ratings Measures & Weights | cms.gov/files/document/2027-star-ratings-measures.pdf | Forward changes incl. COL respecification |
| 2026 Star Ratings Fact Sheet | cms.gov/files/document/2026-star-ratings-fact-sheet.pdf | Program context |

**Tier B — fetch-at-build / excerpt-only:** NCQA public measure summary pages (HTML,
©NCQA — ≤75-word excerpts, amber-badged, cite-by-URL). Payer quick-refs + Medicaid Core Set
specs: manifest-ready v1.1.

Enforcement: pytest gates (committed files ↔ committable manifest posture; no fetch_at_build
doc in `git ls-files`), .gitignore hard-excludes, license_posture rides on every chunk (answer
layer caps rendered snippets ~40 words for excerpt-only). CI + gold set reference Tier A only
→ CI hermetic, zero network.

## 6. Eval design

**Gold set** `evals/dataset/questions.jsonl`: 60 hand-authored items over Tier-A docs —
~48 answerable (eligibility 12, exclusions 10, coding 8, thresholds 8, timelines 6,
cross-year 4) + 12 refusal traps (6 out-of-corpus, 6 licensed-spec-only). Labels at
**(doc_id, measure_id, section)** granularity — survives chunker refactors; a resolver maps
to chunk ids at eval time, failing loudly if a labeled section vanished. Split 15 dev / 45
test; tune on dev, README reports test. Construction protocol documented in `evals/README.md`
(doc-first authoring, ≥50% paraphrased, second-day verification pass);
`scripts/validate_dataset.py` runs in CI.

**Retrieval metrics** (pure functions, keyless, CI on every push): hit@1, recall@5, recall@8,
MRR, precision@8 — overall, per category, dev/test, hybrid vs dense-only — plus gate-A
refusal accuracy on the trap slice. CI builds the index from `corpus/committed/`, runs
`hedis eval --retrieval`, fails if recall@8 or MRR drops >0.02 absolute below the committed
**measured** `evals/baseline.json` (ratchet; moves only via PR + ADR note).

**Faithfulness** (local, real key, `hedis eval --full`, ~$3–5/run): claude-opus-5 judge
(deliberately stronger + different from the sonnet answerer) scores per-claim
supported/unsupported/contradicted → faithfulness ratio + citation validity; refusal traps
scored deterministically by template match. **Human spot-check** every full run:
`hedis review` renders a stratified 15-item sample; judge–human agreement <80% flags the run
untrusted in the artifact and README.

**Artifacts**: `evals/results/{date}-{gitsha}.json` stamped {git_sha, answer_model,
judge_model, judge_prompt_sha256, embedding revision, config_hash, dataset_hash, metrics,
agreement, token usage}. `hedis report` regenerates the README table between markers; CI
checks the table byte-matches the latest artifact. config_hash drift marks LLM numbers STALE
in the README (visible) rather than red-blocking retrieval work.

## 7. Answer & UI design

- Frozen system prompt (unit-tested, cache_control ephemeral): answer only from numbered
  passages; every factual sentence cites [n]; state plan year; thresholds/ages/dates verbatim
  from context; refuse out-of-corpus with the exact code-owned template; licensed-spec
  questions → point to NCQA licensing.
- Inline [n] markers, post-validated (property-tested). Fail-closed regeneration path proven
  in CI with a scripted misbehaving fake model.
- Disclaimer footer appended by code on every path incl. refusals: *"Summarizes public
  CMS/NCQA-published documents. Not the NCQA HEDIS technical specification; not clinical,
  coding, or billing advice."*
- Streamlit chat: citation cards (license badge green "CMS — public domain" / amber "©NCQA —
  brief excerpt", doc, measure chip, year chip, page, snippet, retrieval date, link);
  refusals render as distinct info cards; sidebar measure/year filters + corpus provenance
  panel + retrieval-debug expander (per-leg ranks, RRF scores, hybrid toggle).
- **Keyless degraded mode**: without a key the app runs retrieval-only with full citation
  cards — any repo visitor experiences the retrieval engineering at zero cost.
- `ANTHROPIC_API_KEY` via pydantic-settings SecretStr from .env; absence raises typed
  ConfigError only on real-model paths.

## 8. Module layout

`src/hedis_copilot/`: `config.py` · `llm.py` (the only module importing langchain_anthropic)
· `corpus/` {manifest, fetch, parse, chunk} · `index/` {embed, store, bm25, build} ·
`retrieval/` {aliases, hybrid, fusion, rerank(noop protocol)} · `answer/` {prompts,
citations, chain, service, models} · `evals/` {dataset, retrieval_metrics, judge, runner,
report} · `cli.py` (fetch | build | ask | eval | review | report). Top level:
`app/streamlit_app.py`, `corpus/manifest.yaml`, `corpus/committed/`, `evals/dataset|baseline|
results`, `tests/` (unit, integration, fixtures/corpus with 3 fake measures), `docs/adr/`
(001 chunking · 002 embedding model · 003 hybrid+RRF · 004 license posture · 005 keyless CI
eval design · 006 citation fail-closed).

Deps: langchain-core, langchain-anthropic, langchain-community (BM25 retriever), chromadb,
fastembed, rank-bm25, streamlit, pydantic-settings, httpx, pypdf, beautifulsoup4, typer.
Dev: pytest, pytest-cov, ruff, mypy, pre-commit.

## 9. Testing strategy

Markers: default (pure unit) · `embed` (cached FastEmbed — runs in CI) · `llm` (real key,
auto-skipped in CI). Unit: golden-file parser tests per doc family; chunker invariants
(determinism, no boundary crossing, 512 max, headers); alias-router incl. negatives; RRF math
fixtures; citation parser property tests; metric math fixtures; embedder query-prefix
asymmetry; dataset validator on a broken file; disclaimer on every path. License gates as
tests. Fake-LLM E2E in CI: fixture-corpus Chroma+BM25 in tmp_path; happy path, hallucinated
citation → regenerate-once-then-refuse, gate-A refusal, retrieval-only mode; real-model paths
raise ConfigError keyless. CI (ubuntu, zero secrets): lint/typecheck → unit+fake-E2E →
hermetic retrieval-eval ratchet gate → license/hygiene gates; FastEmbed model via
actions/cache. Weekly windows-latest smoke job.

## 10. Risks (accepted)

- pypdf extraction of the Technical Notes may need per-family parser tuning; golden-file
  tests pin behavior per doc family.
- Small embedder blurs exact tokens — mitigated by hybrid BM25 (the measured delta is the
  point) and contextual headers.
- Solo-authored gold set carries author bias — mitigated by the documented protocol,
  paraphrase quota, and second-day verification pass; judge–human agreement is tracked.
- CMS silently revises PDFs — sha256 pin fails loudly; manifest update is a reviewed PR.
- Public corpus ≠ full NCQA spec — the copilot's refusal behavior makes that boundary a
  feature, and every answer says which year's public documents it summarizes.
