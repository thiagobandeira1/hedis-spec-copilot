"""Streamlit chat UI over the shared AnswerService (SPEC section 7).

Run (after ``uv run hedis build``)::

    uv run streamlit run app/streamlit_app.py

Keyless degraded mode is auto-detected: without ``HEDIS_ANTHROPIC_API_KEY`` the app serves
retrieval-only answers with full citation cards — the retrieval engineering works at zero
cost for any visitor. Importing this module has no Streamlit runtime side effects; the UI
only renders under ``streamlit run`` (which executes it as ``__main__``).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel

from hedis_copilot.answer.models import Answer, Citation
from hedis_copilot.answer.service import AnswerService
from hedis_copilot.config import Settings, get_settings
from hedis_copilot.corpus.manifest import LicensePosture, Manifest, load_manifest
from hedis_copilot.corpus.models import Chunk
from hedis_copilot.index.store import (
    IndexStamp,
    StaleIndexError,
    chunks_path,
    load_chunks_jsonl,
    load_stamp,
)
from hedis_copilot.retrieval.hybrid import HybridRetriever
from hedis_copilot.retrieval.types import ScoredChunk

_REFUSAL_LABELS: dict[str, str] = {
    "refused_low_confidence": (
        "Refused — nothing in the public corpus matched this question confidently."
    ),
    "refused_by_model": (
        "Refused — answering needs material outside the public corpus "
        "(e.g. the licensed NCQA HEDIS specification)."
    ),
    "refused_citation_invalid": (
        "Refused — the model could not ground its answer in the retrieved passages "
        "(fail-closed: no uncited answer ever renders)."
    ),
}


@dataclass(frozen=True)
class AppResources:
    """Everything expensive, loaded once per process via ``st.cache_resource``."""

    retriever: HybridRetriever
    stamp: IndexStamp
    chunks: list[Chunk]
    manifest: Manifest | None
    retrieval_dates: dict[str, str]
    """source_url -> manifest retrieval_date (empty when no manifest is present)."""


def _load_resources_impl() -> AppResources:
    settings = get_settings()
    retriever = HybridRetriever.from_settings(settings)  # raises StaleIndexError when stale
    stamp = load_stamp(settings.index_dir)
    chunks = list(load_chunks_jsonl(chunks_path(settings.index_dir)).values())
    manifest_path = settings.corpus_dir / "manifest.yaml"
    manifest = load_manifest(manifest_path) if manifest_path.is_file() else None
    retrieval_dates = (
        {str(doc.source_url): doc.retrieval_date for doc in manifest.documents}
        if manifest is not None
        else {}
    )
    return AppResources(
        retriever=retriever,
        stamp=stamp,
        chunks=chunks,
        manifest=manifest,
        retrieval_dates=retrieval_dates,
    )


load_resources: Callable[[], AppResources] = st.cache_resource(
    show_spinner="Loading index, BM25, and retriever…"
)(_load_resources_impl)


def _license_badge(posture: LicensePosture) -> str:
    if posture is LicensePosture.US_GOV_PUBLIC_DOMAIN:
        return ":green-badge[CMS — public domain]"
    if posture.committable:
        return ":green-badge[redistributable]"
    return ":orange-badge[©NCQA — brief excerpt, cite by URL]"


def _measure_index(chunks: Sequence[Chunk]) -> dict[str, list[str]]:
    """measure name -> sorted measure ids across years (C14 in 2026 may be C11 in 2025)."""
    mapping: dict[str, set[str]] = {}
    for chunk in chunks:
        if chunk.measure_id is not None and chunk.measure_name is not None:
            mapping.setdefault(chunk.measure_name, set()).add(chunk.measure_id)
    return {name: sorted(ids) for name, ids in sorted(mapping.items())}


def _augment_query(query: str, selected: Sequence[str], measures: dict[str, list[str]]) -> str:
    """Append selected measures' ids so the deterministic alias router filters to them."""
    ids = sorted({measure_id for name in selected for measure_id in measures.get(name, [])})
    return f"{query} {' '.join(ids)}" if ids else query


def _model_factory(settings: Settings) -> Callable[[], BaseChatModel]:
    def factory() -> BaseChatModel:
        from hedis_copilot.llm import answer_model

        return answer_model(settings)

    return factory


def _render_citation(
    citation: Citation,
    resources: AppResources,
    scored: ScoredChunk | None,
    debug: bool,
) -> None:
    with st.container(border=True):
        st.markdown(
            f"**[{citation.marker}] {citation.doc_title}** "
            f"{_license_badge(citation.license_posture)}"
        )
        measure = (
            f"`{citation.measure_id} {citation.measure_name}` · "
            if citation.measure_id is not None and citation.measure_name is not None
            else ""
        )
        page = f" · p.{citation.page}" if citation.page is not None else ""
        st.markdown(f"{measure}`{citation.section}` · plan year {citation.plan_year}{page}")
        st.caption(citation.snippet)
        retrieved = resources.retrieval_dates.get(citation.source_url)
        date_note = (
            f"retrieved {retrieved}"
            if retrieved is not None
            else f"index built {resources.stamp.built_at[:10]}"
        )
        st.markdown(f"[source]({citation.source_url}) · {date_note}")
        if debug and scored is not None:
            st.caption(
                f"debug: dense_rank={scored.dense_rank} · bm25_rank={scored.bm25_rank} · "
                f"rrf_score={scored.score:.5f}"
            )


def _render_answer(answer: Answer, resources: AppResources, debug: bool) -> None:
    if answer.kind == "answered":
        st.write(answer.text)
    elif answer.kind == "retrieval_only":
        st.info(answer.text, icon="🔎")
    else:  # refusal kinds render as distinct info cards
        st.info(f"**{_REFUSAL_LABELS.get(answer.kind, 'Refused.')}**\n\n{answer.text}", icon="🚫")

    if answer.retrieval.measure_filter:
        st.caption(f"measure router filtered to: {', '.join(answer.retrieval.measure_filter)}")
    if answer.retrieval.used_fallback:
        st.caption(
            "router note: the measure filter starved retrieval — unfiltered results "
            "were unioned in (filtered hits first)."
        )

    if answer.citations:
        scored_by_id = {sc.chunk.chunk_id: sc for sc in answer.retrieval.chunks}
        st.markdown("**Citations**")
        for citation in answer.citations:
            _render_citation(citation, resources, scored_by_id.get(citation.chunk_id), debug)


def _sidebar(
    settings: Settings, resources: AppResources, measures: dict[str, list[str]]
) -> tuple[list[str], int | None, bool]:
    """Render the sidebar; returns (selected measure names, plan year or None, debug flag)."""
    with st.sidebar:
        st.header("Filters")
        selected = st.multiselect(
            "Measures",
            options=list(measures),
            help="Hints the deterministic measure router by appending the measures' ids "
            "(Cxx/Dxx) to the query.",
        )
        years = sorted({chunk.plan_year for chunk in resources.chunks}, reverse=True)
        year_labels = ["All years (retriever default)"] + [str(year) for year in years]
        year_choice = str(st.selectbox("Plan year", year_labels, index=0))
        plan_year = int(year_choice) if year_choice.isdigit() else None

        st.header("Corpus provenance")
        st.caption(
            f"Index built {resources.stamp.built_at[:19]}Z · "
            f"{resources.stamp.chunk_count} chunks · "
            f"config {resources.stamp.config_hash} · "
            f"manifest {resources.stamp.manifest_version} · "
            f"{resources.stamp.embedding_model}"
        )
        with st.expander("Index stamp"):
            st.json(resources.stamp.model_dump())
        if resources.manifest is not None:
            with st.expander("Source documents"):
                for doc in resources.manifest.documents:
                    st.markdown(
                        f"**{doc.title}** {_license_badge(doc.license_posture)}  \n"
                        f"{doc.publisher} · plan year {doc.plan_year} · "
                        f"retrieved {doc.retrieval_date} · [source]({doc.source_url})"
                    )

        with st.expander("Advanced"):
            st.slider(
                "top-k (final_k)",
                min_value=1,
                max_value=20,
                value=settings.final_k,
                disabled=True,
                help="Display-only: retrieval knobs are configuration (HEDIS_FINAL_K). "
                "Changing one re-stamps the index — rebuild with `hedis build`.",
            )
            debug = bool(
                st.toggle(
                    "Retrieval debug",
                    value=False,
                    help="Show dense/BM25 ranks and RRF score per citation.",
                )
            )
    return list(selected), plan_year, debug


def main() -> None:
    st.set_page_config(page_title="HEDIS Spec Copilot", page_icon="📋", layout="wide")
    settings = get_settings()

    st.title("HEDIS Spec Copilot")
    st.caption(
        "Cited answers over exclusively public CMS Star Ratings / HEDIS measure documents. "
        "Summarizes public CMS/NCQA-published documents — not the NCQA HEDIS technical "
        "specification; not clinical, coding, or billing advice."
    )

    try:
        resources = load_resources()
    except StaleIndexError as exc:
        st.error(f"{exc}\n\nRun `uv run hedis build`, then reload this page.")
        st.stop()

    keyless = settings.anthropic_api_key is None
    if keyless:
        st.warning(
            "**Keyless mode** — no `HEDIS_ANTHROPIC_API_KEY` configured: answers are "
            "retrieval-only with full citation cards (generation disabled). CI runs this "
            "same mode.",
            icon="🔑",
        )

    measures = _measure_index(resources.chunks)
    selected, plan_year, debug = _sidebar(settings, resources, measures)

    if "history" not in st.session_state:
        st.session_state["history"] = []
    history: list[tuple[str, Answer]] = st.session_state["history"]

    for question, answer in history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            _render_answer(answer, resources, debug)

    prompt = st.chat_input("Ask about eligibility, exclusions, coding, thresholds, timelines…")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        service = AnswerService(
            retriever=resources.retriever,
            model_factory=None if keyless else _model_factory(settings),
            settings=settings,
        )
        query = _augment_query(prompt, selected, measures)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving…" if keyless else "Retrieving + generating…"):
                answer = service.ask(query, plan_year=plan_year)
            _render_answer(answer, resources, debug)
        history.append((prompt, answer))


if __name__ == "__main__":
    main()
