"""Gold-set models, JSONL loading with validation, and label → chunk-id resolution.

Labels live at **(doc_id, measure-by-name, section)** granularity (SPEC §6) so they survive
chunker refactors: the resolver matches measures by case-insensitive *name substring* — no
chunk ids, measure ids, or plan years ever appear inside a label. Resolution fails loudly
(:class:`LabelResolutionError`) when a labeled section vanishes from the corpus.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from hedis_copilot.corpus.models import Chunk, SectionKind

Category = Literal[
    "eligibility",
    "exclusions",
    "coding",
    "thresholds",
    "timelines",
    "cross_year",
    "refusal_out_of_corpus",
    "refusal_licensed_only",
]
Split = Literal["dev", "test"]

REFUSAL_CATEGORIES: frozenset[str] = frozenset({"refusal_out_of_corpus", "refusal_licensed_only"})


class DatasetError(ValueError):
    """The gold-set file is malformed or violates a dataset invariant."""


class GoldLabel(BaseModel):
    """One relevant (doc, measure, section) region.

    ``measure_name_contains`` is a case-insensitive substring of the measure name
    (e.g. ``"colorectal"``); ``None`` targets a document's *general* (non-measure) sections.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    measure_name_contains: str | None = None
    section: SectionKind

    def describe(self) -> str:
        return (
            f"doc_id={self.doc_id} measure_name_contains={self.measure_name_contains!r} "
            f"section={self.section}"
        )


class LabelResolutionError(RuntimeError):
    """A gold label matched zero chunks in the current corpus — fail loudly (SPEC §6)."""

    def __init__(self, unresolved: Sequence[tuple[str, GoldLabel]]) -> None:
        self.unresolved: list[tuple[str, GoldLabel]] = list(unresolved)
        lines = "\n".join(f"  {item_id}: {label.describe()}" for item_id, label in self.unresolved)
        super().__init__(
            f"{len(self.unresolved)} gold label(s) resolved to zero chunks "
            f"(did a labeled section vanish from the corpus?):\n{lines}"
        )


class EvalItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    question: str
    category: Category
    split: Split
    gold: list[GoldLabel] = []
    """Empty for refusal_* items; >= 1 label for answerable items."""
    reference_answer: str | None = None

    @property
    def is_refusal(self) -> bool:
        return self.category in REFUSAL_CATEGORIES


def validate_items(items: Sequence[EvalItem]) -> None:
    """Enforce dataset invariants; raises :class:`DatasetError` listing every violation."""
    problems: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            problems.append(f"{item.item_id}: duplicate item_id")
        seen.add(item.item_id)
        if item.is_refusal:
            if item.gold:
                problems.append(f"{item.item_id}: refusal item must carry no gold labels")
        else:
            if not item.gold:
                problems.append(f"{item.item_id}: answerable item needs >= 1 gold label")
            if not (item.reference_answer or "").strip():
                problems.append(f"{item.item_id}: answerable item needs a reference_answer")
    if problems:
        raise DatasetError("gold set invalid:\n  " + "\n  ".join(problems))


def load_dataset(path: Path) -> list[EvalItem]:
    """Load and validate ``questions.jsonl`` (one :class:`EvalItem` object per line)."""
    if not path.exists():
        raise FileNotFoundError(f"gold set not found at {path}")
    items: list[EvalItem] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name}:{lineno}: invalid JSON: {exc}") from exc
        try:
            items.append(EvalItem.model_validate(raw))
        except ValidationError as exc:
            raise DatasetError(f"{path.name}:{lineno}: {exc}") from exc
    validate_items(items)
    return items


def _matches(label: GoldLabel, chunk: Chunk) -> bool:
    if chunk.doc_id != label.doc_id or chunk.section != label.section:
        return False
    if label.measure_name_contains is None:
        return chunk.measure_name is None
    return (
        chunk.measure_name is not None
        and label.measure_name_contains.lower() in chunk.measure_name.lower()
    )


def resolve_labels(items: Sequence[EvalItem], chunks: Sequence[Chunk]) -> dict[str, set[str]]:
    """Map every item to the chunk ids its labels denote in the *current* chunking.

    A label matches every chunk sharing its doc_id + section whose measure name contains the
    (case-insensitive) substring. Refusal items map to an empty set. Any label matching zero
    chunks raises :class:`LabelResolutionError` naming all unresolvable labels.
    """
    resolved: dict[str, set[str]] = {}
    unresolved: list[tuple[str, GoldLabel]] = []
    for item in items:
        ids: set[str] = set()
        for label in item.gold:
            matched = {chunk.chunk_id for chunk in chunks if _matches(label, chunk)}
            if not matched:
                unresolved.append((item.item_id, label))
            ids |= matched
        resolved[item.item_id] = ids
    if unresolved:
        raise LabelResolutionError(unresolved)
    return resolved
