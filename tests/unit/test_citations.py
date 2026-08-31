"""Unit tests for the deterministic citation parser/validator + license-aware snippets."""

import pytest

from hedis_copilot.answer.citations import extract_markers, to_citations, validate
from hedis_copilot.answer.models import SNIPPET_WORD_LIMIT, doc_title_of, snippet_of
from hedis_copilot.answer.prompts import build_context
from hedis_copilot.corpus.manifest import LicensePosture
from hedis_copilot.corpus.models import Chunk, SectionKind
from hedis_copilot.retrieval.types import ScoredChunk

DOC_TITLE = "CMS 2026 Star Ratings Technical Notes"


def make_chunk(
    chunk_id: str,
    *,
    text: str = "Women 50-74 who had a mammogram during the measurement year.",
    section: SectionKind = "description",
    posture: LicensePosture = LicensePosture.US_GOV_PUBLIC_DOMAIN,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="tn2026",
        measure_id="C01",
        measure_name="Breast Cancer Screening",
        section=section,
        header=f"[{DOC_TITLE} | C01 Breast Cancer Screening | Description | 2026]",
        text=text,
        plan_year=2026,
        license_posture=posture,
        source_url="https://www.cms.gov/files/document/2026-star-ratings-technical-notes.pdf",
        page=38,
    )


def scored(chunks: list[Chunk]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=c, score=0.5) for c in chunks]


# --- extract_markers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("sequence", "joiner"),
    [
        ([1], " "),
        ([1, 2, 3], " "),
        ([2, 1, 2], " "),
        ([8, 8, 8], ""),
        ([10, 3], "\n"),
    ],
)
def test_extract_markers_round_trip(sequence: list[int], joiner: str) -> None:
    """Property-style round trip: markers written into text come back in appearance order."""
    text = joiner.join(f"Fact number {m}. [{m}]" for m in sequence)
    assert extract_markers(text) == sequence


def test_extract_markers_adjacent_and_embedded() -> None:
    assert extract_markers("The rate is 74% [1][2]. See page 5 [10].") == [1, 2, 10]


def test_extract_markers_none() -> None:
    assert extract_markers("No markers here, not even [brackets] with words.") == []


# --- validate ----------------------------------------------------------------------


def test_validate_happy() -> None:
    result = validate("A [1]. B [2].", n_chunks=2)
    assert result.valid
    assert result.invalid_markers == []
    assert result.has_any


def test_validate_out_of_range_marker() -> None:
    result = validate("A [1]. B [9].", n_chunks=2)
    assert not result.valid
    assert result.invalid_markers == [9]
    assert result.has_any


def test_validate_marker_zero_is_invalid() -> None:
    result = validate("A [0].", n_chunks=2)
    assert not result.valid
    assert result.invalid_markers == [0]


def test_validate_zero_markers() -> None:
    result = validate("An answer with no citations at all.", n_chunks=2)
    assert not result.valid
    assert not result.has_any
    assert result.invalid_markers == []


def test_validate_invalid_markers_sorted_and_deduped() -> None:
    result = validate("[9] and [7] and [9] again.", n_chunks=2)
    assert result.invalid_markers == [7, 9]


# --- to_citations ------------------------------------------------------------------


def test_to_citations_first_appearance_order_and_dedup() -> None:
    chunks = scored([make_chunk("c1"), make_chunk("c2"), make_chunk("c3")])
    citations = to_citations("B [2]. A [1]. B again [2]. C [3].", chunks)
    assert [c.marker for c in citations] == [2, 1, 3]
    assert [c.chunk_id for c in citations] == ["c2", "c1", "c3"]


def test_to_citations_skips_out_of_range() -> None:
    chunks = scored([make_chunk("c1")])
    citations = to_citations("A [1]. Ghost [9]. Zero [0].", chunks)
    assert [c.marker for c in citations] == [1]


def test_to_citations_carries_provenance() -> None:
    chunks = scored([make_chunk("c1")])
    (citation,) = to_citations("A [1].", chunks)
    assert citation.doc_title == DOC_TITLE
    assert citation.measure_id == "C01"
    assert citation.measure_name == "Breast Cancer Screening"
    assert citation.section == "description"
    assert citation.plan_year == 2026
    assert citation.page == 38
    assert citation.source_url.endswith("2026-star-ratings-technical-notes.pdf")
    assert citation.license_posture is LicensePosture.US_GOV_PUBLIC_DOMAIN


# --- license-aware snippets + header-derived titles --------------------------------


def test_snippet_full_text_for_committable_posture() -> None:
    long_text = " ".join(f"w{i}" for i in range(60))
    chunk = make_chunk("c1", text=long_text, posture=LicensePosture.US_GOV_PUBLIC_DOMAIN)
    assert snippet_of(chunk) == long_text


def test_snippet_truncated_for_cite_only_posture() -> None:
    words = [f"w{i}" for i in range(60)]
    chunk = make_chunk("c1", text=" ".join(words), posture=LicensePosture.PUBLIC_WEB_CITE_ONLY)
    assert snippet_of(chunk) == " ".join(words[:SNIPPET_WORD_LIMIT]) + " …"


def test_snippet_short_cite_only_text_untruncated() -> None:
    chunk = make_chunk(
        "c1", text="Only five words right here.", posture=LicensePosture.PUBLIC_WEB_CITE_ONLY
    )
    assert snippet_of(chunk) == "Only five words right here."


def test_doc_title_parsed_from_header() -> None:
    assert doc_title_of(make_chunk("c1")) == DOC_TITLE


def test_doc_title_falls_back_to_doc_id_on_malformed_header() -> None:
    chunk = make_chunk("c1").model_copy(update={"header": "[]"})
    assert doc_title_of(chunk) == "tn2026"


# --- build_context (marker grammar the validator depends on) -----------------------


def test_build_context_numbers_blocks_from_one() -> None:
    chunks = scored([make_chunk("c1"), make_chunk("c2")])
    context = build_context(chunks)
    first, second = context.split("\n\n")
    assert first.startswith(f"[1] c1 | {DOC_TITLE} | C01 Breast Cancer Screening | 2026 | ")
    assert second.startswith(f"[2] c2 | {DOC_TITLE} | C01 Breast Cancer Screening | 2026 | ")
    assert chunks[0].chunk.embed_text in first
