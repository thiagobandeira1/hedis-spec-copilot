"""Parsers: CMS PDF -> :class:`NormalizedDoc`.

``technical_notes`` understands the Star Ratings Technical Notes layout: per-measure blocks
headed ``Measure: C02 - Colorectal Cancer Screening`` whose bodies are labeled fields
(``Metric:``, ``Exclusions:``, ``Data Time Frame:`` ...). ``generic_pdf`` produces page-wise
general sections for list/fact-sheet documents. Golden-file tests pin behavior per family.
"""

import re
from pathlib import Path

from pypdf import PdfReader

from hedis_copilot.corpus.manifest import ManifestDoc
from hedis_copilot.corpus.models import DocSection, MeasureBlock, NormalizedDoc, SectionKind

_PAGE_MARK = "\x00PAGE:{n}\x00"
_PAGE_MARK_RE = re.compile("\x00PAGE:(\\d+)\x00")

#: Body measure heading (TOC entries carry dot leaders and are excluded). Layout-mode
#: extraction indents headings and pads runs of spaces, hence the tolerant whitespace.
_MEASURE_HEAD_RE = re.compile(r"^[ \t]*Measure:\s*([CD]\d{2})\s*[-–—:]\s*(.+?)\s*$", re.MULTILINE)
_TOC_DOTS_RE = re.compile(r"\.{5,}")

#: Field label -> section kind. Labels not listed become 'general' sections keyed by label.
_FIELD_KINDS: dict[str, SectionKind] = {
    "Metric": "description",
    "HEDIS Label": "description",
    "Measure Reference": "description",
    "Label for Stars": "description",
    "Title": "description",
    "Description": "description",
    "Numerator": "numerator",
    "Denominator": "denominator",
    "Exclusions": "exclusions",
    "Data Time Frame": "timeline",
    "Data Display": "general",
    "Data Source": "coding",
    "Primary Data Source": "coding",
    "Data Source Category": "coding",
    "General Trend": "general",
    "Statistical Method": "general",
    "Weighting Category": "general",
    "Weighting Value": "general",
    "Improvement Measure": "general",
    "Reporting Requirements": "general",
    "General Notes": "general",
    "Historical Notes": "general",
    "NQF": "general",
    "Metric Notes": "general",
}

#: Labels sit indented in layout mode and may carry doubled internal spaces
#: ("Measure  Reference:"); build each alternative with \s+ between its words.
_FIELD_LABEL_RE = re.compile(
    r"^[ \t]*("
    + "|".join(r"\s+".join(re.escape(w) for w in label.split()) for label in _FIELD_KINDS)
    + r")\s*:\s*",
    re.MULTILINE,
)


def _canonical_label(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def _extract_marked_text(pdf_path: Path) -> str:
    """Concatenate page texts with sentinel page markers for later attribution.

    Layout mode preserves in-word spacing (plain mode splits words across text runs:
    "laborator\ny claims"); the cost is heavy indentation, which normalization collapses.
    """
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        parts.append(_PAGE_MARK.format(n=i))
        parts.append(page.extract_text(extraction_mode="layout") or "")
    return "\n".join(parts)


def _normalize_block(text: str) -> str:
    """Flatten one logical block: join wrapped lines, keep bullet structure, collapse runs."""
    text = _PAGE_MARK_RE.sub(" ", text)
    joined: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("•", "-", "–")):
            joined.append("\n" + line)
        else:
            joined.append(" " + line)
    flat = "".join(joined)
    flat = re.sub(r"[ \t]{2,}", " ", flat)
    flat = re.sub(r"•\s+", "• ", flat)
    # Rejoin hyphenated tokens the layout extractor spaced out ("Patient - level",
    # "2025- 2026"). Bullet dashes are safe: they follow a newline, not an alnum.
    flat = re.sub(r"(?<=[A-Za-z0-9]) ?- ?(?=[A-Za-z0-9])", "-", flat)
    return flat.strip()


def _page_at(marked: str, pos: int) -> int | None:
    last = None
    for mark in _PAGE_MARK_RE.finditer(marked, 0, pos):
        last = int(mark.group(1))
    return last


def _split_fields(block: str) -> list[tuple[str, str]]:
    """Split a measure body into (label, text) fields on recognized labels."""
    matches = list(_FIELD_LABEL_RE.finditer(block))
    if not matches:
        return [("Body", block)]
    fields: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        fields.append(("Body", block[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        fields.append((_canonical_label(m.group(1)), block[m.end() : end]))
    return fields


def parse_technical_notes(pdf_path: Path, doc: ManifestDoc) -> NormalizedDoc:
    marked = _extract_marked_text(pdf_path)

    # Body headings only: drop TOC lines (dot leaders).
    headings = [m for m in _MEASURE_HEAD_RE.finditer(marked) if not _TOC_DOTS_RE.search(m.group(2))]
    measures: list[MeasureBlock] = []
    for i, head in enumerate(headings):
        start = head.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(marked)
        body = marked[start:end]
        page = _page_at(marked, head.start())
        sections: list[DocSection] = []
        for label, raw in _split_fields(body):
            text = _normalize_block(raw)
            if not text:
                continue
            kind = _FIELD_KINDS.get(label, "general")
            sections.append(DocSection(kind=kind, heading=label, text=text, page=page))
        if sections:
            measures.append(
                MeasureBlock(
                    measure_id=head.group(1),
                    measure_name=_normalize_block(head.group(2)),
                    sections=sections,
                )
            )

    # Front/back matter before the first heading becomes general context.
    general: list[DocSection] = []
    if headings:
        preamble = marked[: headings[0].start()]
        # The preamble contains the TOC; keep only the introductory prose before it.
        toc_start = preamble.find("Measure: C01")
        if toc_start > 0:
            preamble = preamble[:toc_start]
        text = _normalize_block(preamble)
        if text:
            general.append(DocSection(kind="general", heading="Introduction", text=text, page=1))

    return NormalizedDoc(
        doc_id=doc.doc_id,
        title=doc.title,
        publisher=doc.publisher,
        source_url=str(doc.source_url),
        plan_year=doc.plan_year,
        doc_type=doc.doc_type,
        license_posture=doc.license_posture,
        retrieval_date=doc.retrieval_date,
        measures=measures,
        general_sections=general,
    )


def parse_generic_pdf(pdf_path: Path, doc: ManifestDoc) -> NormalizedDoc:
    """Fact sheets / measure lists: one general section per page with text."""
    reader = PdfReader(str(pdf_path))
    sections: list[DocSection] = []
    for i, page in enumerate(reader.pages, start=1):
        text = _normalize_block(page.extract_text(extraction_mode="layout") or "")
        if text:
            sections.append(DocSection(kind="general", heading=f"Page {i}", text=text, page=i))
    return NormalizedDoc(
        doc_id=doc.doc_id,
        title=doc.title,
        publisher=doc.publisher,
        source_url=str(doc.source_url),
        plan_year=doc.plan_year,
        doc_type=doc.doc_type,
        license_posture=doc.license_posture,
        retrieval_date=doc.retrieval_date,
        general_sections=sections,
    )


def parse_document(pdf_path: Path, doc: ManifestDoc) -> NormalizedDoc:
    if doc.parser_id == "technical_notes":
        return parse_technical_notes(pdf_path, doc)
    if doc.parser_id == "generic_pdf":
        return parse_generic_pdf(pdf_path, doc)
    raise ValueError(f"no PDF parser for parser_id={doc.parser_id}")
