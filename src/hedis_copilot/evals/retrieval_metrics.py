"""Pure retrieval metrics over (ranked chunk ids, gold chunk-id set).

Keyless, deterministic, no I/O — these run in CI on every push (SPEC §6). All functions
raise ``ValueError`` on an empty gold set (the metric would be undefined; refusal items are
scored separately by the runner, never through these).
"""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

from pydantic import BaseModel, ConfigDict


def _check(gold: AbstractSet[str], k: int) -> None:
    if not gold:
        raise ValueError("gold set is empty — metric undefined (refusal items score elsewhere)")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def hit_at(ranked: Sequence[str], gold: AbstractSet[str], k: int) -> float:
    """1.0 if any gold chunk appears in the top-k, else 0.0."""
    _check(gold, k)
    return 1.0 if any(chunk_id in gold for chunk_id in ranked[:k]) else 0.0


def recall_at(ranked: Sequence[str], gold: AbstractSet[str], k: int) -> float:
    """Fraction of gold chunks present in the top-k."""
    _check(gold, k)
    return len(set(ranked[:k]) & gold) / len(gold)


def precision_at(ranked: Sequence[str], gold: AbstractSet[str], k: int) -> float:
    """Fraction of the top-k slots occupied by gold chunks (denominator is k, not len)."""
    _check(gold, k)
    return len(set(ranked[:k]) & gold) / k


def mrr(ranked: Sequence[str], gold: AbstractSet[str]) -> float:
    """Reciprocal rank of the first gold chunk; 0.0 when no gold chunk is retrieved."""
    if not gold:
        raise ValueError("gold set is empty — metric undefined (refusal items score elsewhere)")
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


class AggregateMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    per_category: dict[str, dict[str, float]]
    overall: dict[str, float]


def _means(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    means: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        means[key] = sum(values) / len(values)
    return means


def aggregate(per_item: Sequence[tuple[str, Mapping[str, float]]]) -> AggregateMetrics:
    """Mean each metric per category and overall from ``(category, metrics)`` pairs.

    A metric key is averaged over the items that carry it, so mixed slices (e.g. refusal
    items scored on ``refused`` only) never dilute retrieval means. Output dicts are
    key-sorted — byte-identical across runs for identical inputs.
    """
    by_category: dict[str, list[Mapping[str, float]]] = {}
    for category, metrics in per_item:
        by_category.setdefault(category, []).append(metrics)
    per_category = {cat: _means(rows) for cat, rows in sorted(by_category.items())}
    overall = _means([metrics for _, metrics in per_item])
    return AggregateMetrics(per_category=per_category, overall=overall)
