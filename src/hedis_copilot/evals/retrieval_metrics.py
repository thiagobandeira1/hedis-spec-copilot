"""Pure retrieval metrics — re-exported from :mod:`clinevals` (P5 adoption shim).

The implementations moved verbatim into ``clinevals.ranking`` (hit@k / recall@k /
precision@k / MRR, all raising ``ValueError`` on an empty gold set) and ``clinevals.runner``
(sparse-key ``aggregate``). This module keeps the ``hedis_copilot.evals.retrieval_metrics``
import surface intact so call sites and tests see zero churn.
"""

from clinevals.ranking import hit_at, mrr, precision_at, recall_at
from clinevals.runner import AggregateMetrics, aggregate

__all__ = ["AggregateMetrics", "aggregate", "hit_at", "mrr", "precision_at", "recall_at"]
