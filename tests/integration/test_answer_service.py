"""Fake-model E2E for AnswerService: every path proven in CI, no key anywhere."""

from collections.abc import Callable, Iterator

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from hedis_copilot.answer.prompts import (
    DISCLAIMER,
    REFUSAL_SENTENCE,
)
from hedis_copilot.answer.service import AnswerService
from hedis_copilot.config import Settings
from hedis_copilot.corpus.manifest import LicensePosture
from hedis_copilot.corpus.models import Chunk, SectionKind
from hedis_copilot.retrieval.types import RetrievalResult, ScoredChunk


def make_chunk(
    chunk_id: str,
    *,
    section: SectionKind = "description",
    text: str = "Women 50-74 who had a mammogram during the measurement year.",
    posture: LicensePosture = LicensePosture.US_GOV_PUBLIC_DOMAIN,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="tn2026",
        measure_id="C01",
        measure_name="Breast Cancer Screening",
        section=section,
        header="[CMS 2026 Star Ratings Technical Notes | C01 Breast Cancer Screening"
        " | Description | 2026]",
        text=text,
        plan_year=2026,
        license_posture=posture,
        source_url="https://www.cms.gov/files/document/2026-star-ratings-technical-notes.pdf",
        page=38,
    )


def make_result(
    chunks: list[Chunk],
    *,
    score: float = 0.5,
    query: str = "q",
    dense_similarity: float | None = 0.8,
) -> RetrievalResult:
    """Gate A keys off best_dense_similarity (cosine), not the rank-derived RRF score."""
    return RetrievalResult(
        query=query,
        chunks=[ScoredChunk(chunk=c, score=score) for c in chunks],
        best_dense_similarity=dense_similarity,
    )


class StubRetriever:
    """Minimal RetrieverLike: returns a canned result, records calls."""

    def __init__(self, result: RetrievalResult) -> None:
        self._result = result
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, *, plan_year: int | None = None) -> RetrievalResult:
        self.calls.append((query, plan_year))
        return self._result


def make_settings() -> Settings:
    return Settings(_env_file=None, answer_model="fake-answerer")


def factory_of(model: BaseChatModel) -> Callable[[], BaseChatModel]:
    return lambda: model


def exploding_factory() -> BaseChatModel:
    raise AssertionError("the model factory must not be called on this path")


def scripted(*replies: str) -> tuple[GenericFakeChatModel, Iterator[str]]:
    """A GenericFakeChatModel over an iterator we can also probe for exhaustion."""
    it = iter(replies)
    return GenericFakeChatModel(messages=it), it


def test_happy_path_validated_answer_with_ordered_citations() -> None:
    retriever = StubRetriever(make_result([make_chunk("c1"), make_chunk("c2")]))
    model, _ = scripted(
        "For the 2026 plan year, women 50-74 are eligible [1]. Exclusions apply [2]."
    )
    service = AnswerService(retriever, factory_of(model), make_settings())

    answer = service.ask("Who is eligible for BCS?", plan_year=2026)

    assert answer.kind == "answered"
    assert "[1]" in answer.text and "[2]" in answer.text
    assert answer.text.endswith(DISCLAIMER)
    assert [c.marker for c in answer.citations] == [1, 2]
    assert [c.chunk_id for c in answer.citations] == ["c1", "c2"]
    assert answer.model_id == "fake-answerer"
    assert answer.retrieval.query == "q"
    assert retriever.calls == [("Who is eligible for BCS?", 2026)]


def test_hallucinated_marker_corrected_on_regenerate() -> None:
    retriever = StubRetriever(make_result([make_chunk("c1"), make_chunk("c2")]))
    model, it = scripted("Made-up citation [9].", "Grounded for 2026 [1].")
    service = AnswerService(retriever, factory_of(model), make_settings())

    answer = service.ask("q")

    assert answer.kind == "answered"
    assert [c.marker for c in answer.citations] == [1]
    assert answer.text.endswith(DISCLAIMER)
    with pytest.raises(StopIteration):  # both scripted replies were consumed
        next(it)


def test_zero_markers_corrected_on_regenerate() -> None:
    retriever = StubRetriever(make_result([make_chunk("c1")]))
    model, _ = scripted("An answer with no citations.", "Now cited [1].")
    service = AnswerService(retriever, factory_of(model), make_settings())

    answer = service.ask("q")

    assert answer.kind == "answered"
    assert [c.marker for c in answer.citations] == [1]


def test_hallucinated_twice_fails_closed() -> None:
    retriever = StubRetriever(make_result([make_chunk("c1"), make_chunk("c2")]))
    model, it = scripted("Wrong [9].", "Still wrong [9].")
    service = AnswerService(retriever, factory_of(model), make_settings())

    answer = service.ask("q")

    assert answer.kind == "refused_citation_invalid"
    assert answer.citations == []
    assert "[9]" not in answer.text  # the invalid model text never renders
    assert answer.text.endswith(DISCLAIMER)
    assert answer.model_id == "fake-answerer"
    with pytest.raises(StopIteration):
        next(it)


def test_model_refusal_sentence_becomes_refused_by_model() -> None:
    retriever = StubRetriever(make_result([make_chunk("c1")]))
    model, _ = scripted(REFUSAL_SENTENCE)
    service = AnswerService(retriever, factory_of(model), make_settings())

    answer = service.ask("q")

    assert answer.kind == "refused_by_model"
    assert answer.text == f"{REFUSAL_SENTENCE}\n\n{DISCLAIMER}"
    assert answer.citations == []
    assert answer.model_id == "fake-answerer"


def test_gate_a_low_score_refuses_without_touching_the_model() -> None:
    retriever = StubRetriever(make_result([make_chunk("c1")], dense_similarity=0.05))
    service = AnswerService(retriever, exploding_factory, make_settings())

    answer = service.ask("q")

    assert answer.kind == "refused_low_confidence"
    assert answer.text.startswith(REFUSAL_SENTENCE)
    assert answer.text.endswith(DISCLAIMER)
    assert answer.citations == []
    assert answer.model_id is None


def test_gate_a_empty_retrieval_refuses_without_touching_the_model() -> None:
    retriever = StubRetriever(RetrievalResult(query="q", chunks=[]))
    service = AnswerService(retriever, exploding_factory, make_settings())

    answer = service.ask("q")

    assert answer.kind == "refused_low_confidence"
    assert answer.model_id is None


def test_keyless_mode_returns_retrieval_only_citation_cards() -> None:
    committable = make_chunk("c1")
    excerpt_only = make_chunk(
        "c2",
        text=" ".join(f"w{i}" for i in range(60)),
        posture=LicensePosture.PUBLIC_WEB_CITE_ONLY,
    )
    retriever = StubRetriever(make_result([committable, excerpt_only]))
    service = AnswerService(retriever, None, make_settings())

    answer = service.ask("q")

    assert answer.kind == "retrieval_only"
    assert answer.text.endswith(DISCLAIMER)
    assert [c.marker for c in answer.citations] == [1, 2]
    assert answer.citations[0].snippet == committable.text  # committable: full text
    assert answer.citations[1].snippet.endswith("…")  # excerpt-only: capped snippet
    assert len(answer.citations[1].snippet.split()) <= 41
    assert answer.model_id is None


def test_gate_a_beats_keyless_mode_when_nothing_relevant() -> None:
    retriever = StubRetriever(RetrievalResult(query="q", chunks=[]))
    service = AnswerService(retriever, None, make_settings())

    answer = service.ask("q")

    assert answer.kind == "refused_low_confidence"
    assert answer.citations == []
