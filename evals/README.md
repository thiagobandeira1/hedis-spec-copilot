# Eval harness — gold set construction protocol

This directory holds the hand-authored gold set (`dataset/questions.jsonl`), the committed
retrieval baseline (`baseline.json`), and stamped result artifacts (`results/`). The code
lives in `src/hedis_copilot/evals/`; this file is the **construction protocol** required by
SPEC §6. The gold-set *content* is authored in a later wave — the framework, validator, and
this protocol land first so authoring happens against fixed rules.

## Two tiers, honestly separated

| Tier | What | Where | Cost |
|---|---|---|---|
| Retrieval metrics | hit@1, recall@5, recall@8, precision@8, MRR + gate-A refusal accuracy | CI, every push, keyless | $0 |
| Faithfulness | per-claim supported/unsupported/contradicted + citation validity, judged by claude-opus-5 | local only (`hedis eval --full`), artifact committed | ~$3–5/run |

CI builds the index from `corpus/committed/` (Tier A only — hermetic, zero network, zero
secrets) and fails if `recall@8` or `mrr` drops more than **0.02 absolute** below
`evals/baseline.json`. The baseline is a **measured ratchet**: it only moves via a PR that
includes an ADR note explaining why.

## Dataset shape

`dataset/questions.jsonl` — one JSON object per line:

```json
{"item_id": "elig-col-001", "question": "Which ages are eligible for colorectal cancer screening in the 2026 Star Ratings?", "category": "eligibility", "split": "test", "gold": [{"doc_id": "cms-tn-2026", "measure_name_contains": "colorectal", "section": "denominator"}], "reference_answer": "Adults 45-75 ..."}
{"item_id": "refuse-vsd-001", "question": "List every ICD-10 code in the COL exclusion value set.", "category": "refusal_licensed_only", "split": "test", "gold": []}
```

- `category` ∈ `eligibility | exclusions | coding | thresholds | timelines | cross_year |
  refusal_out_of_corpus | refusal_licensed_only`.
- `split` ∈ `dev | test`.
- Refusal items carry **no** gold labels; answerable items carry **≥ 1** gold label and a
  `reference_answer`.

### Label granularity (survives chunker refactors)

Labels live at **(doc_id, measure-by-name, section)** — never chunk ids, never measure ids,
never plan years. `measure_name_contains` is a case-insensitive substring of the measure
name (`"colorectal"`, not `"C14"` — ids and years shift across CMS documents; names are the
stable handle). `measure_name_contains: null` targets a document's general (non-measure)
sections. At eval time `resolve_labels()` expands each label to every current chunk with
that doc + section whose measure name contains the substring, and **fails loudly**
(`LabelResolutionError`) if any label matches zero chunks — a labeled section silently
vanishing from the corpus must break the build, not deflate a metric.

## Composition targets (60 items)

| Category | Count | Split |
|---|---|---|
| eligibility | 12 | 60 items total: **15 dev / 45 test** |
| exclusions | 10 | |
| coding | 8 | |
| thresholds | 8 | |
| timelines | 6 | |
| cross_year | 4 | |
| refusal_out_of_corpus | 6 | |
| refusal_licensed_only | 6 | |

Tune retrieval knobs on **dev only**; the README reports **test**. Every gold label
references a Tier-A (committed, US-gov public domain) document, so the whole retrieval tier
stays keyless and network-free in CI.

## Authoring rules

1. **Doc-first authoring.** Open the Tier-A document, pick a real passage, write the
   question *from the passage* — never from memory of what HEDIS "usually" says. Record the
   (doc, measure, section) label at the same moment.
2. **≥ 50 % paraphrased.** At least half the questions must avoid the document's own key
   phrasing (synonyms, reordered constraints, colloquial framing) so BM25 does not get free
   exact-match wins and the hybrid-vs-dense comparison stays meaningful. Mark paraphrase
   status while authoring to prove the quota.
3. **Refusal traps are realistic.** `refusal_out_of_corpus`: plausible measure questions the
   corpus genuinely cannot answer (e.g. Medicaid Core Set, commercial HEDIS). 
   `refusal_licensed_only`: questions only NCQA's licensed full specification can answer
   (exhaustive code/value-set lists, exact spec logic).
4. **Second-day verification pass.** Re-read every item at least one day after authoring:
   confirm the labeled section actually answers the question, the reference answer matches
   the passage verbatim on numbers/ages/dates, and the category fits. Fix or drop; never
   ship an unverified item.
5. **Validation gate.** `uv run python scripts/validate_dataset.py` must exit 0: schema,
   unique ids, refusal/gold invariants, and full label resolution against the committed
   corpus. CI runs it on every push.

Known bias: the gold set is solo-authored. The paraphrase quota, the second-day pass, and
the tracked judge–human agreement number (below) are the mitigations; they are documented
rather than hidden.

## Faithfulness tier

`hedis eval --full` answers every item with the real pipeline, then a **claude-opus-5**
judge (stronger than and different from the claude-sonnet-5 answerer) applies the frozen
per-claim rubric in `judge.py`: split the answer into atomic claims, verdict each as
supported / unsupported / contradicted against the retrieved passages only, and check that
cited passages actually support their claims. Refusal traps are scored deterministically by
template match, never by the judge. Every full run includes a **human spot-check**
(`hedis review`, stratified 15-item sample); judge–human agreement < 80 % flags the run as
untrusted in the artifact and the README.

## Artifacts

`results/{date}-{gitsha}.json`, written with sorted keys + trailing newline, stamped with
`{git_sha, answer_model, judge_model, judge_prompt_sha256, embedding revision, config_hash,
dataset_hash, metrics, agreement, token_usage}`. `hedis report` regenerates the README
table between `<!-- EVAL:BEGIN -->` / `<!-- EVAL:END -->` from the latest artifact; CI
checks the table byte-matches. A `config_hash` drift marks LLM numbers **STALE** in the
README instead of blocking retrieval work.
