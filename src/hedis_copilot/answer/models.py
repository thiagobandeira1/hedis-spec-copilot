"""Answer-layer data models: citation cards with full provenance + the Answer envelope."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict

from hedis_copilot.corpus.manifest import LicensePosture
from hedis_copilot.corpus.models import Chunk, SectionKind
from hedis_copilot.retrieval.types import RetrievalResult

AnswerKind = Literal[
    "answered",
    "refused_low_confidence",
    "refused_by_model",
    "refused_citation_invalid",
    "retrieval_only",
]

SNIPPET_WORD_LIMIT = 40
"""Rendered-snippet cap (in words) for non-committable license postures (SPEC section 5)."""


def doc_title_of(chunk: Chunk) -> str:
    """Document title recovered from the contextual header's first ``|`` segment.

    :class:`Chunk` carries ``doc_id`` but not the human title; the chunker writes the title
    as the first segment of every header (``[<title> | <measure> | <heading> | <year>]``),
    so we parse it back out. Falls back to ``doc_id`` if the header is malformed.
    """
    inner = chunk.header.strip().removeprefix("[").removesuffix("]")
    title = inner.split(" | ", 1)[0].strip()
    return title or chunk.doc_id


def snippet_of(chunk: Chunk) -> str:
    """License-aware snippet: full text for committable postures, ~40 words + ellipsis else."""
    if chunk.license_posture.committable:
        return chunk.text
    words = chunk.text.split()
    if len(words) <= SNIPPET_WORD_LIMIT:
        return chunk.text
    return " ".join(words[:SNIPPET_WORD_LIMIT]) + " …"


class Citation(BaseModel):
    """One citation card — everything the UI renders, license posture included."""

    model_config = ConfigDict(frozen=True)

    marker: int
    chunk_id: str
    doc_title: str
    measure_id: str | None
    measure_name: str | None
    section: SectionKind
    plan_year: int
    source_url: str
    page: int | None
    snippet: str
    license_posture: LicensePosture

    @classmethod
    def from_chunk(cls, marker: int, chunk: Chunk) -> Self:
        return cls(
            marker=marker,
            chunk_id=chunk.chunk_id,
            doc_title=doc_title_of(chunk),
            measure_id=chunk.measure_id,
            measure_name=chunk.measure_name,
            section=chunk.section,
            plan_year=chunk.plan_year,
            source_url=chunk.source_url,
            page=chunk.page,
            snippet=snippet_of(chunk),
            license_posture=chunk.license_posture,
        )


class Answer(BaseModel):
    """The final envelope every surface (CLI, Streamlit, evals) renders.

    ``text`` is fully rendered — the code-appended disclaimer is always already present.
    """

    model_config = ConfigDict(frozen=True)

    kind: AnswerKind
    text: str
    citations: list[Citation]
    retrieval: RetrievalResult
    model_id: str | None
