"""Stamped eval artifacts + README table sync — via :mod:`clinevals.artifacts` (P5 shim).

``write_artifact`` serializes deterministically (sorted keys, trailing newline, LF on every
OS, refuses NaN); ``sync_readme`` rewrites only the region between the EVAL markers and is
idempotent. ``render_readme_table`` / ``sync_readme`` keep their one-artifact P2 shape: the
hybrid-vs-dense two-column layout is clinevals' ``comparison`` option, which reproduces the
committed README region byte-for-byte.
"""

from collections.abc import Mapping
from pathlib import Path

from clinevals import artifacts as _artifacts
from clinevals.artifacts import EVAL_BEGIN, EVAL_END, ReportError, write_artifact

#: The SPEC-promised side-by-side: hybrid (published) vs dense-only (BM25's justification).
COMPARISON: tuple[str, str] = ("Dense-only", "dense_only_overall")
PRIMARY_LABEL = "Hybrid"


def render_readme_table(artifact: Mapping[str, object]) -> str:
    """Render the artifact's overall metrics as a markdown table plus a stamp line.

    Expects ``artifact["metrics"]`` to be a mapping — either ``{"overall": {...}, ...}`` or
    a flat ``{metric: value}`` mapping. A non-empty ``metrics.dense_only_overall`` adds the
    Dense-only column; recognized top-level stamps (git_sha, config_hash, dataset_hash,
    answer_model, judge_model) are appended as an italic line.
    """
    return _artifacts.render_metrics_table(
        artifact, comparison=COMPARISON, primary_label=PRIMARY_LABEL
    )


def sync_readme(readme_path: Path, artifact: Mapping[str, object]) -> str:
    """Replace the README region between the EVAL markers with the rendered table.

    Returns the full updated README text (also written back to ``readme_path``). Raises
    :class:`ReportError` when the markers are missing or out of order — the table must
    never be silently appended anywhere else.
    """
    return _artifacts.sync_readme(
        readme_path, artifact, comparison=COMPARISON, primary_label=PRIMARY_LABEL
    )


__all__ = [
    "COMPARISON",
    "EVAL_BEGIN",
    "EVAL_END",
    "PRIMARY_LABEL",
    "ReportError",
    "render_readme_table",
    "sync_readme",
    "write_artifact",
]
