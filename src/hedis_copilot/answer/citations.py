"""Deterministic citation parsing and validation — the fail-closed gate (pure functions)."""

import re

from pydantic import BaseModel, ConfigDict

from hedis_copilot.answer.models import Citation
from hedis_copilot.retrieval.types import ScoredChunk

_MARKER_RE = re.compile(r"\[(\d+)\]")


def extract_markers(text: str) -> list[int]:
    """Every ``[n]`` marker in appearance order, duplicates preserved."""
    return [int(m) for m in _MARKER_RE.findall(text)]


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool
    """True only when at least one marker exists and all markers resolve to a passage."""
    invalid_markers: list[int]
    """Markers outside 1..n_chunks (sorted, deduplicated)."""
    has_any: bool


def validate(text: str, n_chunks: int) -> ValidationResult:
    """Check that ``text`` cites at least once and only passages ``[1]..[n_chunks]``."""
    markers = extract_markers(text)
    invalid = sorted({m for m in markers if not 1 <= m <= n_chunks})
    has_any = bool(markers)
    return ValidationResult(valid=has_any and not invalid, invalid_markers=invalid, has_any=has_any)


def to_citations(text: str, chunks: list[ScoredChunk]) -> list[Citation]:
    """Citation cards in first-appearance order, deduplicated by marker.

    Out-of-range markers are skipped defensively; every service path runs
    :func:`validate` first, so in practice none survive to this point.
    """
    seen: set[int] = set()
    citations: list[Citation] = []
    for marker in extract_markers(text):
        if marker in seen or not 1 <= marker <= len(chunks):
            continue
        seen.add(marker)
        citations.append(Citation.from_chunk(marker, chunks[marker - 1].chunk))
    return citations
