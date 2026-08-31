"""Deterministic measure-alias router (no LLM query routing — ADR-003).

Measure numbering shifts between Star Ratings years (Controlling Blood Pressure is C14 in
2026 but C11 in 2025), so measure ids are never hand-maintained here. Instead a small
curated map of *name fragments -> alias tokens* is matched against the measure names in
the actual corpus, deriving an ``alias -> {plan_year: [measure_ids]}`` table from whatever
corpus it is given. Literal ``Cxx``/``Dxx`` mentions always route, table or not.

Generic program questions ("star rating cut points") match no alias and route to ``[]``,
leaving retrieval unfiltered.
"""

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from hedis_copilot.config import get_settings
from hedis_copilot.corpus.models import Chunk, NormalizedDoc
from hedis_copilot.corpus.normalize import load_committed

AliasTable = dict[str, dict[int, list[str]]]
"""alias token/phrase -> plan_year -> sorted measure ids (derived, never hand-written)."""

MeasureEntry = tuple[int, str, str]
"""(plan_year, measure_id, measure_name) — the unit the table is derived from."""

_MEASURE_ID_RE = re.compile(r"\b([CD]\d{2})\b", re.IGNORECASE)
_TOKEN_SPLIT_RE = re.compile(r"\W+")

# Curated, corpus-agnostic: lowercase measure-name fragment -> alias tokens/phrases.
# Single-word aliases match query tokens exactly; multi-word aliases match as phrases.
# Deliberately NO generic tokens (rating, plan, star, cut) — those must route to [].
_NAME_FRAGMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "breast cancer screening": ("bcs", "breast cancer", "mammogram", "mammography"),
    "colorectal cancer screening": ("col", "colorectal", "colonoscopy", "colorectal cancer"),
    "annual flu vaccine": ("flu", "influenza", "flu vaccine", "flu shot"),
    "monitoring physical activity": ("pao", "physical activity"),
    "special needs plan": ("snp",),
    "care for older adults": ("coa", "older adults"),
    "osteoporosis management": ("omw", "osteoporosis"),
    "eye exam": ("eed", "eye exam", "retinal exam", "diabetic eye"),
    "blood sugar controlled": ("gsd", "hbd", "glycemic", "blood sugar", "hba1c", "a1c"),
    "kidney health evaluation": ("ked", "kidney"),
    "controlling blood pressure": ("cbp", "blood pressure"),
    "reducing the risk of falling": ("frm", "falling", "falls", "fall risk"),
    "improving bladder control": ("mui", "bladder", "incontinence"),
    "medication reconciliation post-discharge": ("mrp", "medication reconciliation"),
    "plan all-cause readmissions": ("pcr", "readmission", "readmissions"),
    "statin therapy for patients with cardiovascular disease": ("spc", "statin"),
    "transitions of care": ("trc", "transitions of care"),
    "follow-up after emergency department visit": ("fmc", "emergency department", "ed visit"),
    "medication adherence for diabetes": ("mad", "adherence for diabetes"),
    "medication adherence for hypertension": ("mah", "ras antagonists"),
    "medication adherence for cholesterol": ("mac", "statins"),
    "mtm program completion": ("mtm", "cmr", "comprehensive medication review"),
    "statin use in persons with diabetes": ("supd", "statin"),
}


def entries_from_docs(docs: Iterable[NormalizedDoc]) -> list[MeasureEntry]:
    return [
        (doc.plan_year, measure.measure_id, measure.measure_name)
        for doc in docs
        for measure in doc.measures
    ]


def entries_from_chunks(chunks: Iterable[Chunk]) -> list[MeasureEntry]:
    return [
        (chunk.plan_year, chunk.measure_id, chunk.measure_name)
        for chunk in chunks
        if chunk.measure_id is not None and chunk.measure_name is not None
    ]


def build_alias_table(entries: Iterable[MeasureEntry]) -> AliasTable:
    """Derive the alias table by matching curated name fragments against real names."""
    collected: dict[str, dict[int, set[str]]] = {}
    for plan_year, measure_id, measure_name in entries:
        lowered = measure_name.lower()
        for fragment, aliases in _NAME_FRAGMENT_ALIASES.items():
            if fragment in lowered:
                for alias in aliases:
                    collected.setdefault(alias, {}).setdefault(plan_year, set()).add(measure_id)
    return {
        alias: {year: sorted(ids) for year, ids in sorted(by_year.items())}
        for alias, by_year in sorted(collected.items())
    }


@lru_cache(maxsize=8)
def load_alias_table(committed_dir: Path) -> AliasTable:
    """Table derived from the committed corpus on disk (cached per directory)."""
    return build_alias_table(entries_from_docs(load_committed(committed_dir)))


class AliasRouter:
    """Routes a query to measure ids using a derived alias table + literal id regex."""

    def __init__(self, table: AliasTable) -> None:
        self._table = table

    def route(self, query: str, year: int | None = None) -> list[str]:
        """Measure ids the query names — [] means "no filter", never "no results"."""
        lowered = query.lower()
        tokens = {token for token in _TOKEN_SPLIT_RE.split(lowered) if token}
        matched: set[str] = set()
        for alias, by_year in self._table.items():
            hit = alias in lowered if " " in alias else alias in tokens
            if not hit:
                continue
            years = [year] if year is not None else list(by_year)
            for candidate_year in years:
                matched.update(by_year.get(candidate_year, []))
        matched.update(match.group(1).upper() for match in _MEASURE_ID_RE.finditer(query))
        return sorted(matched)


def route(query: str, year: int | None = None) -> list[str]:
    """Route against the committed corpus at ``settings.corpus_dir / 'committed'``."""
    committed_dir = get_settings().corpus_dir / "committed"
    return AliasRouter(load_alias_table(committed_dir)).route(query, year)


def __getattr__(name: str) -> AliasTable:
    # MEASURE_ALIASES is derived from the on-disk corpus; PEP 562 keeps the module
    # import-safe (no I/O at import time) while still exposing the table as an attribute.
    if name == "MEASURE_ALIASES":
        return load_alias_table(get_settings().corpus_dir / "committed")
    raise AttributeError(name)
