"""Chunker invariants (ADR-001) on synthetic :class:`NormalizedDoc` fixtures.

``_token_estimate`` is the module's own budget heuristic; importing it keeps these
invariants phrased in the chunker's own units rather than re-deriving a rival estimate.
"""

import re

from hedis_copilot.corpus.chunk import _token_estimate, chunk_document
from hedis_copilot.corpus.manifest import LicensePosture
from hedis_copilot.corpus.models import (
    DocSection,
    MeasureBlock,
    NormalizedDoc,
    SectionKind,
)

_CHUNK_ID_RE = re.compile(r"^[a-z0-9-]+:[A-Za-z0-9-]+:[a-z]+:[0-9a-f]{16}$")


def _section(kind: SectionKind, heading: str, text: str, page: int | None = 7) -> DocSection:
    return DocSection(kind=kind, heading=heading, text=text, page=page)


def _doc(
    measures: list[MeasureBlock] | None = None,
    general_sections: list[DocSection] | None = None,
) -> NormalizedDoc:
    return NormalizedDoc(
        doc_id="fake-tn-2026",
        title="Fake 2026 Technical Notes",
        publisher="CMS",
        source_url="https://www.cms.gov/files/document/fake.pdf",
        plan_year=2026,
        doc_type="technical_notes",
        license_posture=LicensePosture.US_GOV_PUBLIC_DOMAIN,
        retrieval_date="2026-08-30",
        measures=measures or [],
        general_sections=general_sections or [],
    )


def _prose(n_sentences: int) -> str:
    return " ".join(
        f"Sentence number {i} explains the denominator criteria for this measure in detail."
        for i in range(n_sentences)
    )


def _two_measure_doc() -> NormalizedDoc:
    return _doc(
        measures=[
            MeasureBlock(
                measure_id="C01",
                measure_name="Breast Cancer Screening",
                sections=[
                    _section("description", "Metric", "The percentage of women screened."),
                    _section("exclusions", "Exclusions", "Hospice at any time. " + _prose(60)),
                ],
            ),
            MeasureBlock(
                measure_id="C02",
                measure_name="Colorectal Cancer Screening",
                sections=[
                    _section("description", "Metric", "The percentage of adults screened."),
                    _section("timeline", "Data Time Frame", _prose(80)),
                ],
            ),
        ],
        general_sections=[_section("general", "Introduction", "Star Ratings overview.", page=1)],
    )


class TestDeterminismAndIds:
    def test_chunking_is_deterministic(self) -> None:
        doc = _two_measure_doc()
        assert chunk_document(doc) == chunk_document(doc)

    def test_chunk_ids_are_unique(self) -> None:
        chunks = chunk_document(_two_measure_doc())
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_format(self) -> None:
        for chunk in chunk_document(_two_measure_doc()):
            assert _CHUNK_ID_RE.match(chunk.chunk_id), chunk.chunk_id
        general = [c for c in chunk_document(_two_measure_doc()) if c.measure_id is None]
        assert general and all(c.chunk_id.split(":")[1] == "doc" for c in general)

    def test_identical_content_yields_identical_ids_across_runs(self) -> None:
        first = {c.chunk_id for c in chunk_document(_two_measure_doc())}
        second = {c.chunk_id for c in chunk_document(_two_measure_doc())}
        assert first == second


class TestTokenBudget:
    def test_small_section_stays_one_chunk_verbatim(self) -> None:
        doc = _doc(
            measures=[
                MeasureBlock(
                    measure_id="C14",
                    measure_name="Controlling Blood Pressure",
                    sections=[_section("numerator", "Numerator", "Members whose BP was <140/90.")],
                )
            ]
        )
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Members whose BP was <140/90."

    def test_oversized_prose_section_splits_within_budget(self) -> None:
        max_tokens, overlap = 480, 64
        doc = _doc(
            measures=[
                MeasureBlock(
                    measure_id="C02",
                    measure_name="Colorectal Cancer Screening",
                    sections=[_section("exclusions", "Exclusions", _prose(200))],
                )
            ]
        )
        chunks = chunk_document(doc, max_tokens=max_tokens, overlap_tokens=overlap)
        assert len(chunks) > 1
        for chunk in chunks:
            # A window may carry up to overlap_tokens of trailing context from its
            # predecessor before its own units, so the ceiling is max + overlap.
            assert _token_estimate(chunk.text) <= max_tokens + overlap, len(chunk.text)

    def test_oversized_delimiter_free_text_hard_splits(self) -> None:
        """Code tables carry no sentence delimiters; the word-window hard split kicks in."""
        max_tokens = 480
        text = " ".join(f"G{i:04d}" for i in range(2000))  # no '.', ';', or bullets
        doc = _doc(
            measures=[
                MeasureBlock(
                    measure_id="D08",
                    measure_name="Medication Adherence",
                    sections=[_section("coding", "Data Source", text)],
                )
            ]
        )
        chunks = chunk_document(doc, max_tokens=max_tokens, overlap_tokens=64)
        assert len(chunks) > 1
        max_words = int(max_tokens / 1.3)
        for chunk in chunks:
            assert len(chunk.text.split()) <= max_words
        joined = " ".join(c.text for c in chunks)
        assert "G0000" in joined and "G1999" in joined

    def test_all_words_survive_prose_split(self) -> None:
        text = _prose(150)
        doc = _doc(
            general_sections=[_section("general", "Introduction", text, page=1)],
        )
        chunks = chunk_document(doc)
        covered = " ".join(c.text for c in chunks)
        assert set(text.split()) <= set(covered.split())


class TestContextualHeaders:
    def test_measure_chunk_header_format(self) -> None:
        chunks = chunk_document(_two_measure_doc())
        bcs = [c for c in chunks if c.measure_id == "C01" and c.section == "description"]
        assert bcs
        expected = "[Fake 2026 Technical Notes | C01 Breast Cancer Screening | Metric | 2026]"
        assert bcs[0].header == expected

    def test_general_chunk_header_uses_general_marker(self) -> None:
        chunks = chunk_document(_two_measure_doc())
        general = [c for c in chunks if c.measure_id is None]
        assert general
        assert general[0].header == "[Fake 2026 Technical Notes | General | Introduction | 2026]"

    def test_every_split_piece_of_a_section_shares_the_header(self) -> None:
        chunks = chunk_document(_two_measure_doc())
        timeline = [c for c in chunks if c.section == "timeline"]
        assert len(timeline) > 1
        assert len({c.header for c in timeline}) == 1

    def test_embed_text_prepends_header(self) -> None:
        chunk = chunk_document(_two_measure_doc())[0]
        assert chunk.embed_text == f"{chunk.header}\n{chunk.text}"


class TestDuplicateCollapse:
    def test_identical_sections_within_one_measure_collapse(self) -> None:
        section = _section("exclusions", "Exclusions", "Hospice at any time.")
        doc = _doc(
            measures=[
                MeasureBlock(
                    measure_id="C01",
                    measure_name="Breast Cancer Screening",
                    sections=[section, section],
                )
            ]
        )
        chunks = chunk_document(doc)
        assert len(chunks) == 1

    def test_same_text_under_different_measures_is_kept_for_both(self) -> None:
        text = "Members in hospice are excluded."
        doc = _doc(
            measures=[
                MeasureBlock(
                    measure_id="C01",
                    measure_name="Breast Cancer Screening",
                    sections=[_section("exclusions", "Exclusions", text)],
                ),
                MeasureBlock(
                    measure_id="C02",
                    measure_name="Colorectal Cancer Screening",
                    sections=[_section("exclusions", "Exclusions", text)],
                ),
            ]
        )
        chunks = chunk_document(doc)
        assert len(chunks) == 2
        assert len({c.chunk_id for c in chunks}) == 2


class TestProvenance:
    def test_every_chunk_carries_doc_provenance(self) -> None:
        for chunk in chunk_document(_two_measure_doc()):
            assert chunk.doc_id == "fake-tn-2026"
            assert chunk.plan_year == 2026
            assert chunk.source_url == "https://www.cms.gov/files/document/fake.pdf"
            assert chunk.license_posture is LicensePosture.US_GOV_PUBLIC_DOMAIN

    def test_page_travels_from_section_to_chunk(self) -> None:
        chunks = chunk_document(_two_measure_doc())
        assert all(c.page == 7 for c in chunks if c.measure_id is not None)
        assert all(c.page == 1 for c in chunks if c.measure_id is None)
