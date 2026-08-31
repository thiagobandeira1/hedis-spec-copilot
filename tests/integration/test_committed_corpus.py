"""Golden-ish gates over the committed Tier-A corpus — runs in CI on the committed JSON.

Measure ids are derived by name substring, never hardcoded: CMS renumbers measures between
plan years, and these tests must survive that.
"""

import re
from pathlib import Path

import pytest

from hedis_copilot.corpus.chunk import chunk_document
from hedis_copilot.corpus.models import Chunk, MeasureBlock, NormalizedDoc
from hedis_copilot.corpus.normalize import load_committed

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_DIR = REPO_ROOT / "corpus" / "committed"

_MEASURE_ID_RE = re.compile(r"^[CD]\d{2}$")


@pytest.fixture(scope="module")
def docs() -> list[NormalizedDoc]:
    return load_committed(COMMITTED_DIR)


@pytest.fixture(scope="module")
def tn_2026(docs: list[NormalizedDoc]) -> NormalizedDoc:
    return next(d for d in docs if d.doc_id == "cms-tn-2026")


@pytest.fixture(scope="module")
def all_chunks(docs: list[NormalizedDoc]) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc)]


def _measures_named(doc: NormalizedDoc, name_fragment: str) -> list[MeasureBlock]:
    return [m for m in doc.measures if name_fragment.lower() in m.measure_name.lower()]


class TestDocuments:
    def test_all_four_docs_load(self, docs: list[NormalizedDoc]) -> None:
        assert {d.doc_id for d in docs} == {
            "cms-tn-2026",
            "cms-tn-2025",
            "cms-measures-2027",
            "cms-factsheet-2026",
        }

    def test_tn_2026_has_at_least_45_measures(self, tn_2026: NormalizedDoc) -> None:
        assert len(tn_2026.measures) >= 45

    @pytest.mark.parametrize(
        "name_fragment",
        ["Breast Cancer Screening", "Colorectal Cancer Screening", "Blood Pressure"],
    )
    def test_key_measures_present_with_derived_star_ids(
        self, tn_2026: NormalizedDoc, name_fragment: str
    ) -> None:
        matches = _measures_named(tn_2026, name_fragment)
        assert matches, f"no 2026 measure named like {name_fragment!r}"
        for measure in matches:
            assert _MEASURE_ID_RE.match(measure.measure_id), measure.measure_id

    def test_key_measure_ids_are_distinct(self, tn_2026: NormalizedDoc) -> None:
        ids = {
            _measures_named(tn_2026, fragment)[0].measure_id
            for fragment in (
                "Breast Cancer Screening",
                "Colorectal Cancer Screening",
                "Blood Pressure",
            )
        }
        assert len(ids) == 3

    def test_colorectal_exclusions_mention_hospice(self, tn_2026: NormalizedDoc) -> None:
        col = _measures_named(tn_2026, "Colorectal Cancer Screening")[0]
        exclusion_sections = [s for s in col.sections if s.kind == "exclusions"]
        assert exclusion_sections, f"{col.measure_id} has no exclusions section"
        combined = " ".join(s.text for s in exclusion_sections)
        assert "hospice" in combined.lower()


class TestChunks:
    def test_more_than_1500_unique_chunks(self, all_chunks: list[Chunk]) -> None:
        assert len(all_chunks) > 1500
        assert len({c.chunk_id for c in all_chunks}) == len(all_chunks)

    def test_chunking_is_deterministic_across_runs(
        self, docs: list[NormalizedDoc], all_chunks: list[Chunk]
    ) -> None:
        rechunked = [chunk for doc in docs for chunk in chunk_document(doc)]
        assert rechunked == all_chunks

    def test_no_chunk_exceeds_3200_chars(self, all_chunks: list[Chunk]) -> None:
        oversized = [c.chunk_id for c in all_chunks if len(c.text) > 3200]
        assert not oversized, oversized[:5]

    def test_every_chunk_carries_full_provenance(self, all_chunks: list[Chunk]) -> None:
        for chunk in all_chunks:
            assert chunk.source_url.startswith("https://www.cms.gov"), chunk.chunk_id
            assert chunk.plan_year in {2025, 2026, 2027}, chunk.chunk_id
            assert chunk.license_posture.committable, chunk.chunk_id
            assert chunk.doc_id
            assert chunk.header.startswith("[") and chunk.header.endswith("]")
            assert chunk.text.strip()
