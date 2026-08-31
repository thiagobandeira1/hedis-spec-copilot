"""Normalized document + chunk models. Everything carries provenance."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from hedis_copilot.corpus.manifest import LicensePosture

SectionKind = Literal[
    "description", "numerator", "denominator", "exclusions", "timeline", "coding", "general"
]


class DocSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: SectionKind
    heading: str
    text: str
    page: int | None = None
    """1-based page where the section starts in the source PDF (None for HTML)."""


class MeasureBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    measure_id: str
    """Star Ratings id, e.g. 'C02'."""
    measure_name: str
    sections: list[DocSection]


class NormalizedDoc(BaseModel):
    """One corpus document, parsed and normalized — the committed Tier-A artifact."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    publisher: str
    source_url: str
    plan_year: int
    doc_type: str
    license_posture: LicensePosture
    retrieval_date: str
    measures: list[MeasureBlock] = []
    general_sections: list[DocSection] = []


class Chunk(BaseModel):
    """One retrievable unit; the contextual header travels with the text everywhere."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    measure_id: str | None
    measure_name: str | None
    section: SectionKind
    header: str
    text: str
    plan_year: int
    license_posture: LicensePosture
    source_url: str
    page: int | None = None

    @property
    def embed_text(self) -> str:
        """What gets embedded and shown to the LLM: header + body."""
        return f"{self.header}\n{self.text}"
