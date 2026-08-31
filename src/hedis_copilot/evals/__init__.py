"""Two-tier eval harness (SPEC §6).

Tier 1 — keyless retrieval metrics, recomputed in CI on every push and gated against a
ratcheted baseline. Tier 2 — local-only LLM faithfulness runs whose stamped artifacts are
committed; README numbers are regenerated from artifacts, never hand-typed.
"""

from hedis_copilot.evals.dataset import (
    DatasetError,
    EvalItem,
    GoldLabel,
    LabelResolutionError,
    load_dataset,
    resolve_labels,
)
from hedis_copilot.evals.judge import (
    JUDGE_PROMPT,
    JudgeParseError,
    JudgeVerdict,
    build_judge_input,
    parse_judge_output,
)
from hedis_copilot.evals.report import render_readme_table, sync_readme, write_artifact
from hedis_copilot.evals.retrieval_metrics import (
    AggregateMetrics,
    aggregate,
    hit_at,
    mrr,
    precision_at,
    recall_at,
)
from hedis_copilot.evals.runner import (
    ItemResult,
    RetrievalReport,
    compare_to_baseline,
    run_full_eval,
    run_retrieval_eval,
)

__all__ = [
    "JUDGE_PROMPT",
    "AggregateMetrics",
    "DatasetError",
    "EvalItem",
    "GoldLabel",
    "ItemResult",
    "JudgeParseError",
    "JudgeVerdict",
    "LabelResolutionError",
    "RetrievalReport",
    "aggregate",
    "build_judge_input",
    "compare_to_baseline",
    "hit_at",
    "load_dataset",
    "mrr",
    "parse_judge_output",
    "precision_at",
    "recall_at",
    "render_readme_table",
    "resolve_labels",
    "run_full_eval",
    "run_retrieval_eval",
    "sync_readme",
    "write_artifact",
]
