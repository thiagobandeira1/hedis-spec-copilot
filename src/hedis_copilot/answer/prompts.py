"""FROZEN prompt constants + context assembly (SPEC section 7).

These strings are contract, not copy: evals template-match refusals against them and the
citation validator assumes the ``[n]`` marker grammar demanded here. Change only via PR +
ADR note, never casually.
"""

from typing import Final

from hedis_copilot.answer.models import doc_title_of
from hedis_copilot.retrieval.types import ScoredChunk

REFUSAL_SENTENCE: Final[str] = (
    "The public documents in my corpus do not contain the information needed to answer this."
)

DISCLAIMER: Final[str] = (
    "— Summarizes public CMS/NCQA-published documents. Not the NCQA HEDIS technical "
    "specification; not clinical, coding, or billing advice."
)

SYSTEM_PROMPT: Final[str] = f"""\
You answer questions about Medicare Part C & D Star Ratings and HEDIS measures using ONLY the
numbered context passages provided in the user message. Follow every rule:

1. Ground every claim in the passages. Never use outside knowledge, even when you are sure.
2. Every factual sentence must carry at least one inline citation marker such as [1] or [2][4]
   pointing to the passage(s) that support it. Do not cite passages you did not use.
3. State which plan year(s) your answer describes.
4. Reproduce thresholds, ages, percentages, rates, and dates VERBATIM from the passages —
   never round, convert, or infer them.
5. If the passages do not contain the information needed, reply with exactly this sentence and
   nothing else: {REFUSAL_SENTENCE}
6. If the question requires NCQA's licensed full HEDIS technical specification — for example
   exhaustive code lists, value set contents, or complete specification text — refuse using the
   sentence in rule 5 and add that the full specification is licensed by NCQA (ncqa.org).
7. Do not add disclaimers or caveats about your sources; a disclaimer is appended by code."""

REFUSAL_LOW_CONFIDENCE: Final[str] = (
    REFUSAL_SENTENCE + " This corpus covers public CMS Medicare Part C & D Star Ratings"
    " documents (the 2025 and 2026 Star Ratings Technical Notes, the 2027 measures & weights"
    " list, and the 2026 fact sheet) plus brief NCQA public measure summaries. Try asking about"
    " a specific measure's eligibility, exclusions, coding, thresholds, or timelines."
)

REFUSAL_CITATION_INVALID: Final[str] = (
    "The model could not produce an answer whose citations all resolve to retrieved passages,"
    " so no answer is shown — this system never renders uncited claims. Please rephrase or"
    " narrow the question."
)

RETRIEVAL_ONLY_MESSAGE: Final[str] = (
    "Running keyless (no ANTHROPIC_API_KEY), so no generated answer is available. The most"
    " relevant passages from the public corpus are cited below with full provenance. Set"
    " HEDIS_ANTHROPIC_API_KEY in .env to enable cited answers."
)

CITATION_CORRECTION_TEMPLATE: Final[str] = (
    "Correction: your previous answer failed citation validation ({problem}). Rewrite the"
    " answer from scratch using only markers [1] through [{n}] that refer to the numbered"
    " context passages above, with at least one marker on every factual sentence."
)


def build_context(chunks: list[ScoredChunk]) -> str:
    """Numbered context blocks: ``[n] chunk_id | doc | measure | year | url`` + embed text."""
    blocks: list[str] = []
    for n, scored in enumerate(chunks, start=1):
        chunk = scored.chunk
        measure = (
            f"{chunk.measure_id} {chunk.measure_name}"
            if chunk.measure_id and chunk.measure_name
            else "General"
        )
        blocks.append(
            f"[{n}] {chunk.chunk_id} | {doc_title_of(chunk)} | {measure}"
            f" | {chunk.plan_year} | {chunk.source_url}\n{chunk.embed_text}"
        )
    return "\n\n".join(blocks)
