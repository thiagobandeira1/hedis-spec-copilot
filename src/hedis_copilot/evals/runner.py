"""Retrieval-eval runner + ratchet comparison — keyless by construction.

The retriever arrives as a plain ``Callable[[str], list[str]]`` (query → ranked chunk ids)
and the gate-A refusal probe as ``Callable[[str], bool]``, so this module never touches an
index, an embedder, or a network. ``run_full_eval`` (faithfulness tier) is a deliberate
loud stub until the answer service and the authored gold set land.
"""

from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import NoReturn

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from hedis_copilot.evals.dataset import EvalItem
from hedis_copilot.evals.retrieval_metrics import aggregate, hit_at, mrr, precision_at, recall_at
from hedis_copilot.retrieval.types import RetrieverLike

RetrieverFn = Callable[[str], list[str]]
"""query -> ranked chunk ids (best first)."""
GateAFn = Callable[[str], bool]
"""query -> True when the pre-LLM refusal gate would refuse it."""

#: Metrics the CI ratchet gates on (SPEC §6: fail if either drops >0.02 below baseline).
GATED_METRICS: tuple[str, ...] = ("recall@8", "mrr")


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
    per_item: list[ItemResult] = []
    answerable_rows: list[tuple[str, dict[str, float]]] = []
    split_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}
    refusal_rows: list[tuple[str, dict[str, float]]] = []
    for item in items:
        if item.is_refusal:
            if gate_a_fn is None:
                continue
            refused = 1.0 if gate_a_fn(item.question) else 0.0
            metrics = {"refused": refused}
            refusal_rows.append((item.category, metrics))
        else:
            gold = resolved_gold.get(item.item_id)
            if not gold:
                raise KeyError(
                    f"no resolved gold chunks for answerable item {item.item_id!r} — "
                    "run resolve_labels over the same corpus first"
                )
            ranked = retriever_fn(item.question)
            metrics = {
                "hit@1": hit_at(ranked, gold, 1),
                "recall@5": recall_at(ranked, gold, 5),
                "recall@8": recall_at(ranked, gold, 8),
                "precision@8": precision_at(ranked, gold, 8),
                "mrr": mrr(ranked, gold),
            }
            answerable_rows.append((item.category, metrics))
            split_rows.setdefault(item.split, []).append((item.category, metrics))
        per_item.append(
            ItemResult(
                item_id=item.item_id, category=item.category, split=item.split, metrics=metrics
            )
        )

    answerable_agg = aggregate(answerable_rows)
    per_category = dict(answerable_agg.per_category)
    if refusal_rows:
        per_category.update(aggregate(refusal_rows).per_category)
    refusal_scores = [metrics["refused"] for _, metrics in refusal_rows]
    return RetrievalReport(
        per_item=per_item,
        per_category=dict(sorted(per_category.items())),
        overall=answerable_agg.overall,
        per_split={split: aggregate(rows).overall for split, rows in sorted(split_rows.items())},
        refusal_trap_accuracy=(
            sum(refusal_scores) / len(refusal_scores) if refusal_scores else None
        ),
    )


def compare_to_baseline(
    report: RetrievalReport, baseline: Mapping[str, float], tolerance: float = 0.02
) -> list[str]:
    """Ratchet gate: return one message per gated metric worse than baseline - tolerance.

    Only ``recall@8`` and ``mrr`` are gated (SPEC §6), and only when the committed
    ``evals/baseline.json`` carries them. An empty list means the gate passes.
    """
    regressions: list[str] = []
    gated_block = report.test_overall  # the ratchet guards published (test-split) numbers
    for metric in GATED_METRICS:
        if metric not in baseline:
            continue
        expected = baseline[metric]
        actual = gated_block.get(metric)
        if actual is None:
            regressions.append(f"{metric}: missing from report (baseline {expected:.4f})")
        elif actual < expected - tolerance:
            regressions.append(
                f"{metric}: {actual:.4f} regressed below baseline {expected:.4f} "
                f"- tolerance {tolerance:.2f}"
            )
    return regressions


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
