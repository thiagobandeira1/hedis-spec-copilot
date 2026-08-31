# ADR-004: Two-tier corpus with license posture as tested behavior

## Status

Accepted

## Date

2026-08-31

## Context

HEDIS is a trademarked NCQA program and the full Technical Specifications (Volume 2) are
licensed, paid content. A public portfolio repo must not commit, vendor, or wholesale
paraphrase that material — but CMS's own Star Ratings documents (Technical Notes, fact
sheets, measure lists) are US-government works in the public domain under 17 U.S.C. §105
and describe every target measure in verifiable detail. The risk to manage is drift:
licensing intent stated only in a README erodes as files get added. It has to be
enforceable by machines, in CI, on every push.

## Decision

Split the corpus into two tiers declared per document in `corpus/manifest.yaml` and typed
in `corpus/manifest.py`:

- **Tier A — committed.** US-gov public-domain docs (the four CMS Star Ratings documents).
  Normalized JSON is committed under `corpus/committed/`; raw PDFs are cached gitignored
  and sha256-pinned so a silent CMS revision fails loudly.
- **Tier B — fetch-at-build / excerpt-only.** NCQA public summary pages (©NCQA). Never
  committed; fetched into gitignored `corpus/fetched/` only when a user runs
  `hedis fetch`; the answer layer caps rendered snippets (~40 words) and citation cards
  carry an amber "©NCQA — brief excerpt" badge that links out rather than reproducing.

Enforcement is layered and tested:

1. `load_manifest` **hard-fails** if `commit_policy: commit_normalized` is paired with a
   non-committable `license_posture` (the `LicensePosture.committable` property).
2. Pytest license gates assert no fetch-at-build doc ever appears in `git ls-files` and
   that every committed file maps to a committable manifest posture.
3. `license_posture` rides on **every chunk** (`corpus/models.py`), so the render layer
   can enforce excerpt caps without re-consulting the manifest.
4. Questions answerable only from the licensed full spec get an explicit refusal that
   points to NCQA licensing — the boundary is a product behavior, not a gap.

## Alternatives considered

- **Public-domain-only corpus** (drop Tier B): simplest legally, but loses NCQA's own
  public summaries, which are the best source for measure intent; cite-only excerpts with
  hard caps keep them usable within fair-use bounds.
- **Commit everything and rely on repo privacy**: not viable for a public portfolio and
  precisely the failure mode this ADR exists to prevent.
- **Policy documented but untested**: rejected — every rule above exists as a failing
  test or a hard ValueError, not prose.

## Consequences

- CI and the gold set reference Tier A only, so CI is hermetic with zero network.
- The copilot **cannot answer exhaustive code-list questions** — the public corpus is not
  the spec, and the refusal template says so. This is the honest posture, but it means
  some genuinely useful HEDIS questions are out of scope by design.
- Tier B freshness depends on users running `hedis fetch`; a fresh clone in keyless mode
  runs on Tier A alone, so NCQA-summary context is absent until fetched.
- sha256 pinning means any upstream CMS revision breaks `hedis fetch` until a reviewed
  manifest PR updates the pin — deliberate friction, accepted.
