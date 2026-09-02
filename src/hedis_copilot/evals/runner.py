"""Retrieval-eval runner + ratchet comparison — keyless by construction.

The retriever arrives as a plain ``Callable[[str], list[str]]`` (query → ranked chunk ids)
and the gate-A refusal probe as ``Callable[[str], bool]``, so this module never touches an
index, an embedder, or a network. Scoring is an adapter over
:func:`clinevals.runner.score_items`; the ratchet delegates to
:func:`clinevals.ratchet.compare_to_baseline`. ``run_full_eval`` (faithfulness tier) is a
deliberate loud stub until the answer service and the authored gold set land.
"""

from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import NoReturn

from clinevals.dataset import EvalItemBase
from clinevals.ranking import hit_at, mrr, precision_at, recall_at
from clinevals.ratchet import Gate
from clinevals.ratchet import compare_to_baseline as _compare_flat
from clinevals.runner import ItemScore, score_items
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from hedis_copilot.evals.dataset import EvalItem
from hedis_copilot.retrieval.types import RetrieverLike

RetrieverFn = Callable[[str], list[str]]
"""query -> ranked chunk ids (best first)."""
GateAFn = Callable[[str], bool]
"""query -> True when the pre-LLM refusal gate would refuse it."""

#: Metrics the CI ratchet gates on (SPEC §6: fail if either drops >0.02 below baseline).
GATED_METRICS: tuple[str, ...] = ("recall@8", "mrr")
_GATES: tuple[Gate, ...] = tuple(Gate(metric) for metric in GATED_METRICS)


class ItemResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    category: str
    split: str
    metrics: dict[str, float]
    """Answerable items: hit@1/recall@5/recall@8/precision@8/mrr. Refusal items: refused."""


class RetrievalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    per_item: list[ItemResult]
    per_category: dict[str, dict[str, float]]
    overall: dict[str, float]
    """Means over ALL answerable items (dev+test) — diagnostic only, never published."""
    per_split: dict[str, dict[str, float]] = {}
    """Answerable-item means per split. ``per_split['test']`` is the publishable number:
    tuning happens on dev, so combined metrics would overstate rigor (review finding)."""
    refusal_trap_accuracy: float | None = None
    """Fraction of refusal_* items gate A refused; None when no gate_a_fn was supplied."""

    @property
    def test_overall(self) -> dict[str, float]:
        """The headline block: test-split means, falling back to overall if no splits."""
        return self.per_split.get("test", self.overall)


class _ScoredItem(EvalItemBase):
    """clinevals-typed view of an :class:`EvalItem` (P2's model predates ``EvalItemBase``)."""

    source: EvalItem


def run_retrieval_eval(
    retriever_fn: RetrieverFn,
    items: Sequence[EvalItem],
    resolved_gold: Mapping[str, AbstractSet[str]],
    *,
    gate_a_fn: GateAFn | None = None,
) -> RetrievalReport:
    """Score every item: retrieval metrics for answerable items, gate A for refusal traps.

    ``resolved_gold`` comes from :func:`hedis_copilot.evals.dataset.resolve_labels`. Refusal
    items are skipped entirely when ``gate_a_fn`` is None (their gold set is empty, so no
    retrieval metric is defined for them).
    """

    def score(row: _ScoredItem) -> ItemScore | None:
        item = row.source
        if item.is_refusal:
            if gate_a_fn is None:
                return None
            return ItemScore(metrics={"refused": 1.0 if gate_a_fn(item.question) else 0.0})
        gold = resolved_gold.get(item.item_id)
        if not gold:
            raise KeyError(
                f"no resolved gold chunks for answerable item {item.item_id!r} — "
                "run resolve_labels over the same corpus first"
            )
        ranked = retriever_fn(item.question)
        return ItemScore(
            metrics={
                "hit@1": hit_at(ranked, gold, 1),
                "recall@5": recall_at(ranked, gold, 5),
                "recall@8": recall_at(ranked, gold, 8),
                "precision@8": precision_at(ranked, gold, 8),
                "mrr": mrr(ranked, gold),
            }
        )

    rows = [
        _ScoredItem(item_id=item.item_id, category=item.category, split=item.split, source=item)
        for item in items
    ]
    report = score_items(rows, score)

    # Byte-compat with the committed artifacts: clinevals' sparse-key means carry the refusal
    # rows' ``refused`` key in overall/per_split, while P2 publishes answerable-only blocks
    # there (refusal traps live in per_category + refusal_trap_accuracy). Every other mean is
    # computed over the same carriers in the same order, so the floats are identical.
    overall = {key: value for key, value in report.overall.items() if key != "refused"}
    per_split: dict[str, dict[str, float]] = {}
    for split, block in report.per_split.items():
        kept = {key: value for key, value in block.items() if key != "refused"}
        if kept:  # a split holding only refusal traps has no answerable block in P2
            per_split[split] = kept
    refusal_scores = [row.metrics["refused"] for row in report.per_item if "refused" in row.metrics]
    return RetrievalReport(
        per_item=[
            ItemResult(item_id=r.item_id, category=r.category, split=r.split, metrics=r.metrics)
            for r in report.per_item
        ],
        per_category=report.per_category,
        overall=overall,
        per_split=per_split,
        refusal_trap_accuracy=(
            sum(refusal_scores) / len(refusal_scores) if refusal_scores else None
        ),
    )


def compare_to_baseline(
    report: RetrievalReport, baseline: Mapping[str, float], tolerance: float = 0.02
) -> list[str]:
    """Ratchet gate: return one message per gated metric worse than baseline - tolerance.

    Only ``recall@8`` and ``mrr`` are gated (SPEC §6), and only when the committed
    ``evals/baseline.json`` carries them; the ratchet guards published (test-split) numbers.
    An empty list means the gate passes.
    """
    regressions = _compare_flat(report.test_overall, baseline, gates=_GATES, tolerance=tolerance)
    # P2's pre-adoption wording for a gated metric absent from the report is part of the
    # shim's compatibility surface (pinned by tests/unit/test_retrieval_metrics.py).
    return [
        message.replace("missing from current results", "missing from report", 1)
        for message in regressions
    ]


def run_full_eval(
    items: Sequence[EvalItem],
    retriever: RetrieverLike,
    answerer: BaseChatModel,
    judge: BaseChatModel,
) -> NoReturn:
    """Faithfulness + citation-validity tier (``hedis eval --full``) — NOT YET WIRED.

    This stub exists so the CLI and artifact plumbing can typecheck against the final
    signature now, but it cannot be implemented in this wave: producing answers requires
    the AnswerService (answer wave), and scoring them requires the authored gold set with
    reference answers and refusal templates (eval-content wave). A loud
    ``NotImplementedError`` beats a silently half-wired eval that could mint fake numbers.
    """
    raise NotImplementedError("wired in the eval-content wave")
