# Build plan — hedis-spec-copilot

- [ ] **A. Scaffold**: pyproject, CI (zero secrets), pre-commit, config.py, llm.py factory.
- [ ] **B. Corpus**: manifest schema + fetch (sha256-pinned, browser UA) + Technical-Notes
      parser (golden-file tested) + chunker (measure/section-aware, contextual headers) →
      committed Tier-A normalized docs.
- [ ] **C. Index/retrieval**: FastEmbed adapter, embedded Chroma + stamp, BM25, alias router,
      hybrid + RRF fusion, Reranker noop seam.
- [ ] **D. Answer layer**: frozen prompts, citation parse/validate (fail-closed), LCEL chain,
      AnswerService, refusal gates, disclaimer.
- [ ] **E. Evals**: 60-item gold set (doc-first, protocolled), retrieval metrics + ratchet
      baseline, judge + runner + report, dataset validator.
- [ ] **F. UI + CLI**: Streamlit (citation cards, keyless degraded mode), `hedis` commands.
- [ ] **G. Review gate**: adversarial multi-agent review; fix confirmed findings via PR.
- [ ] **H. Ship**: README (generated eval table), ADRs 001–006, .env.example, repo + CI green.
