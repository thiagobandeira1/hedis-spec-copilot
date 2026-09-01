"""Stamped eval artifacts + README table sync (SPEC §6: numbers generated, never typed).

``write_artifact`` serializes deterministically (sorted keys, trailing newline) so CI can
byte-compare; ``sync_readme`` rewrites only the region between the EVAL markers and is
idempotent — syncing the same artifact twice yields byte-identical output.
"""

import json
from collections.abc import Mapping
from pathlib import Path

EVAL_BEGIN = "<!-- EVAL:BEGIN -->"
EVAL_END = "<!-- EVAL:END -->"


class ReportError(RuntimeError):
    """The artifact or README does not have the shape reporting requires."""


def write_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Write an eval artifact deterministically: sorted keys, 2-space indent, trailing \\n.

    ``payload`` must already include its stamps (git_sha, models, judge_prompt_sha256,
    config_hash, dataset_hash, metrics, ... per SPEC §6) — this function never invents data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _format_value(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    return f"{value:.3f}"


def render_readme_table(artifact: Mapping[str, object]) -> str:
    """Render the artifact's overall metrics as a markdown table plus a stamp line.

    Expects ``artifact["metrics"]`` to be a mapping — either ``{"overall": {...}, ...}``
    or a flat ``{metric: value}`` mapping. Recognized top-level stamps (git_sha,
    config_hash, dataset_hash, answer_model, judge_model) are appended as an italic line.
    """
    metrics = artifact.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ReportError("artifact has no 'metrics' mapping")
    overall_raw = metrics.get("overall", metrics)
    if not isinstance(overall_raw, Mapping) or not overall_raw:
        raise ReportError("artifact metrics carry no overall values")
    dense_raw = metrics.get("dense_only_overall")
    if isinstance(dense_raw, Mapping) and dense_raw:
        lines = ["| Metric (test split) | Hybrid | Dense-only |", "|---|---|---|"]
        lines.extend(
            f"| {key} | {_format_value(overall_raw[key])} | "
            f"{_format_value(dense_raw.get(key, '—'))} |"
            for key in sorted(overall_raw)
        )
    else:
        lines = ["| Metric | Value |", "|---|---|"]
        lines.extend(
            f"| {key} | {_format_value(overall_raw[key])} |" for key in sorted(overall_raw)
        )
    stamp_keys = ("git_sha", "config_hash", "dataset_hash", "answer_model", "judge_model")
    stamps = [f"{key}={artifact[key]}" for key in stamp_keys if key in artifact]
    if stamps:
        lines.extend(["", f"_Stamps: {' · '.join(stamps)}_"])
    return "\n".join(lines)


def sync_readme(readme_path: Path, artifact: Mapping[str, object]) -> str:
    """Replace the README region between the EVAL markers with the rendered table.

    Returns the full updated README text (also written back to ``readme_path``). Raises
    :class:`ReportError` when the markers are missing or out of order — the table must
    never be silently appended anywhere else.
    """
    text = readme_path.read_text(encoding="utf-8")
    begin = text.find(EVAL_BEGIN)
    end = text.find(EVAL_END)
    if begin == -1 or end == -1 or end < begin:
        raise ReportError(
            f"README markers {EVAL_BEGIN} / {EVAL_END} missing or out of order in {readme_path}"
        )
    table = render_readme_table(artifact)
    updated = text[: begin + len(EVAL_BEGIN)] + "\n" + table + "\n" + text[end:]
    readme_path.write_text(updated, encoding="utf-8")
    return updated
