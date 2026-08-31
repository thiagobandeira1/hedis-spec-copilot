"""AnswerService — the one orchestrator CLI, Streamlit, and evals all share.

Flow (SPEC sections 4 and 7): retrieve -> gate A (score floor, no model call) ->
keyless retrieval-only mode -> generate -> deterministic citation validation with one
correction retry -> fail-closed refusal. The disclaimer is appended by CODE on every path.
"""

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from hedis_copilot.answer.chain import generate
from hedis_copilot.answer.citations import ValidationResult, to_citations, validate
from hedis_copilot.answer.models import Answer, AnswerKind, Citation
from hedis_copilot.answer.prompts import (
    CITATION_CORRECTION_TEMPLATE,
    DISCLAIMER,
    REFUSAL_CITATION_INVALID,
    REFUSAL_LOW_CONFIDENCE,
    REFUSAL_SENTENCE,
    RETRIEVAL_ONLY_MESSAGE,
)
from hedis_copilot.config import Settings
from hedis_copilot.retrieval.types import RetrievalResult, RetrieverLike


def _with_disclaimer(text: str) -> str:
    return f"{text}\n\n{DISCLAIMER}"


def _correction_hint(check: ValidationResult, n_chunks: int) -> str:
    if check.invalid_markers:
        cited = ", ".join(f"[{m}]" for m in check.invalid_markers)
        problem = f"it cited nonexistent passage(s) {cited}"
    else:
        problem = "it contained no citation markers at all"
    return CITATION_CORRECTION_TEMPLATE.format(problem=problem, n=n_chunks)


class AnswerService:
    """Single-turn RAG answering over an injected retriever and (optional) chat model.

    ``model_factory`` returning the model lazily keeps this constructible keyless: gate-A
    refusals and retrieval-only mode never touch it.
    """

    def __init__(
        self,
        retriever: RetrieverLike,
        model_factory: Callable[[], BaseChatModel] | None,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._model_factory = model_factory
        self._settings = settings

    def ask(self, query: str, plan_year: int | None = None) -> Answer:
        result = self._retriever.retrieve(query, plan_year=plan_year)

        # Gate A: nothing retrieved, or best fused score under the calibrated floor ->
        # code-owned refusal with zero model involvement (works with model_factory=None).
        best = max((scored.score for scored in result.chunks), default=0.0)
        if not result.chunks or best < self._settings.refusal_score_floor:
            return self._envelope("refused_low_confidence", REFUSAL_LOW_CONFIDENCE, result)

        # Keyless degraded mode: no model, but full citation cards for every retrieved chunk.
        if self._model_factory is None:
            citations = [
                Citation.from_chunk(n, scored.chunk)
                for n, scored in enumerate(result.chunks, start=1)
            ]
            return self._envelope(
                "retrieval_only", RETRIEVAL_ONLY_MESSAGE, result, citations=citations
            )

        model = self._model_factory()
        model_id = self._settings.answer_model

        text = generate(model, query, result.chunks)
        if REFUSAL_SENTENCE in text:
            return self._envelope("refused_by_model", REFUSAL_SENTENCE, result, model_id=model_id)

        check = validate(text, len(result.chunks))
        if not check.valid:
            # Regenerate ONCE with an explicit correction, then fail closed.
            correction = _correction_hint(check, len(result.chunks))
            text = generate(model, query, result.chunks, correction=correction)
            if REFUSAL_SENTENCE in text:
                return self._envelope(
                    "refused_by_model", REFUSAL_SENTENCE, result, model_id=model_id
                )
            check = validate(text, len(result.chunks))
            if not check.valid:
                return self._envelope(
                    "refused_citation_invalid", REFUSAL_CITATION_INVALID, result, model_id=model_id
                )

        return self._envelope(
            "answered", text, result, citations=to_citations(text, result.chunks), model_id=model_id
        )

    @staticmethod
    def _envelope(
        kind: AnswerKind,
        body: str,
        result: RetrievalResult,
        *,
        citations: list[Citation] | None = None,
        model_id: str | None = None,
    ) -> Answer:
        return Answer(
            kind=kind,
            text=_with_disclaimer(body),
            citations=citations or [],
            retrieval=result,
            model_id=model_id,
        )
