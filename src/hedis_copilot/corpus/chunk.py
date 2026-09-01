"""Measure/section-aware chunking with contextual headers (ADR-001).

A section stays one chunk when it fits; otherwise it splits within the section — never
across a section or measure boundary. Every chunk carries a contextual header giving the
small embedder the measure identity a mid-document fragment loses, and a deterministic
content-hashed id so re-chunking identical content yields identical ids.
"""

import hashlib
import re

from hedis_copilot.corpus.models import Chunk, DocSection, NormalizedDoc

_WORD_RE = re.compile(r"\S+")


def _token_estimate(text: str) -> int:
    """Cheap proxy bounded two ways: ~1.3 tokens/word for prose, ~4 chars/token for dense
    numeric/tabular text where word counts undercount."""
    return max(int(len(_WORD_RE.findall(text)) * 1.3), len(text) // 4)


def _split_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split on sentence/bullet boundaries into windows <= max_tokens with overlap."""
    units = [u.strip() for u in re.split(r"(?<=[.;])\s+|\n(?=[•-])", text) if u.strip()]
    if not units:
        return []
    # Tables and code lists carry no sentence delimiters: hard-split any oversized unit
    # into word windows so no chunk can ever exceed the embedder's context.
    hardened: list[str] = []
    max_words = max(int(max_tokens / 1.3), 1)
    for unit in units:
        words = _WORD_RE.findall(unit)
        if len(words) <= max_words:
            hardened.append(unit)
        else:
            step = max(max_words - int(overlap_tokens / 1.3), 1)
            hardened.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), step))
    units = hardened
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = _token_estimate(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            windows.append(" ".join(current))
            # Overlap: carry trailing units up to overlap_tokens into the next window.
            carried: list[str] = []
            carried_tokens = 0
            for prev in reversed(current):
                prev_tokens = _token_estimate(prev)
                if carried_tokens + prev_tokens > overlap_tokens:
                    break
                carried.insert(0, prev)
                carried_tokens += prev_tokens
            current = carried
            current_tokens = carried_tokens
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        windows.append(" ".join(current))
    # Final guarantee: _token_estimate charges dense numeric text by characters (chars/4),
    # which the word-count hardening above cannot see — a 369-word coefficient table can
    # still estimate >512 tokens and would be silently truncated by the embedder. Char-split
    # any window that still exceeds the budget.
    max_chars = max_tokens * 4
    bounded: list[str] = []
    for window in windows:
        if _token_estimate(window) <= max_tokens:
            bounded.append(window)
            continue
        for start in range(0, len(window), max_chars):
            piece = window[start : start + max_chars].strip()
            if piece:
                bounded.append(piece)
    return bounded


def _chunk_id(doc_id: str, measure_id: str | None, section: str, text: str) -> str:
    digest = hashlib.sha256(f"{doc_id}|{measure_id}|{section}|{text}".encode()).hexdigest()[:16]
    return f"{doc_id}:{measure_id or 'doc'}:{section}:{digest}"


def chunk_document(
    doc: NormalizedDoc, *, max_tokens: int = 480, overlap_tokens: int = 64
) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(section: DocSection, measure_id: str | None, measure_name: str | None) -> None:
        header_measure = (
            f"{measure_id} {measure_name}" if measure_id and measure_name else "General"
        )
        header = f"[{doc.title} | {header_measure} | {section.heading} | {doc.plan_year}]"
        pieces = (
            [section.text]
            if _token_estimate(section.text) <= max_tokens
            else _split_text(section.text, max_tokens, overlap_tokens)
        )
        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.doc_id, measure_id, section.kind, piece),
                    doc_id=doc.doc_id,
                    measure_id=measure_id,
                    measure_name=measure_name,
                    section=section.kind,
                    header=header,
                    text=piece,
                    plan_year=doc.plan_year,
                    license_posture=doc.license_posture,
                    source_url=doc.source_url,
                    page=section.page,
                )
            )

    for measure in doc.measures:
        for section in measure.sections:
            emit(section, measure.measure_id, measure.measure_name)
    for section in doc.general_sections:
        emit(section, None, None)

    # Duplicate content within one (doc, measure, section) collapses to one chunk.
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            unique.append(chunk)
    return unique
