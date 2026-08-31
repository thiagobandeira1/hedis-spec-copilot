"""LLM-judge prompt contract + tolerant output parsing. **No network code lives here.**

The judge model is injected by the caller as a ``BaseChatModel`` (SPEC §6: claude-opus-5,
deliberately stronger than and different from the answerer); this module owns only the
frozen per-claim rubric and the deterministic parsing of whatever the judge returns.
"""

import hashlib
import json
from collections.abc import Sequence
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

JUDGE_PROMPT = """\
You are grading a RAG answer about Medicare Star Ratings / HEDIS measures for faithfulness
to its retrieved passages. Judge ONLY against the numbered passages provided — outside
knowledge must never rescue an unsupported claim.

Procedure:
1. Split the ANSWER into atomic factual claims (one independently checkable fact each).
   Ignore the machine-appended disclaimer footer and pure hedging language.
2. Assign each claim exactly one verdict:
   - "supported": stated by, or directly entailed by, at least one passage.
   - "unsupported": not present in any passage.
   - "contradicted": at least one passage states the opposite.
3. Numbers, ages, dates, and thresholds count as supported only when they match a passage
   verbatim.
4. For each claim carrying inline [n] citation markers, set "citation_valid" to true only
   if at least one *cited* passage [n] itself supports the claim, false otherwise. Use
   null for a claim with no citation marker.

Output STRICT JSON only — no markdown fences, no prose — exactly this shape:
{"claims": [{"claim": "<text>", "verdict": "supported|unsupported|contradicted",
"citation_valid": true|false|null}], "notes": "<one line on anything odd, or empty>"}
"""


def judge_prompt_sha256() -> str:
    """SHA-256 of the frozen rubric — stamped into eval artifacts so drift is visible."""
    return hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()


class JudgeParseError(ValueError):
    """The judge's output carried no parseable, structurally valid verdict JSON."""


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    claims_total: int
    claims_supported: int
    claims_unsupported: int
    claims_contradicted: int
    citation_valid_ratio: float | None
    """Mean of citation_valid over cited claims; None when no claim carried a citation."""
    notes: str = ""

    @property
    def faithfulness(self) -> float | None:
        """supported / total; None when the judge found no claims to grade."""
        if self.claims_total == 0:
            return None
        return self.claims_supported / self.claims_total


def build_judge_input(question: str, answer_text: str, passages: Sequence[str]) -> str:
    """Format the human-turn payload: numbered passages, then question, then answer."""
    numbered = "\n\n".join(f"[{i}] {passage}" for i, passage in enumerate(passages, start=1))
    return f"PASSAGES:\n{numbered}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer_text}"


def _extract_json(raw: str) -> dict[str, Any]:
    """Return the first valid JSON object embedded anywhere in ``raw``.

    Tolerates prose or markdown fences around the JSON by attempting a decode at every
    ``{`` until one parses. Raises :class:`JudgeParseError` when nothing does.
    """
    decoder = json.JSONDecoder()
    idx = raw.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            idx = raw.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            return cast(dict[str, Any], obj)
        idx = raw.find("{", idx + 1)
    raise JudgeParseError("no JSON object found in judge output")


def parse_judge_output(raw: str) -> JudgeVerdict:
    """Parse the judge's raw text into a :class:`JudgeVerdict`; tolerant of surrounding prose.

    Raises :class:`JudgeParseError` on garbage: no JSON at all, a missing/`non-list`
    ``claims`` field, an unknown verdict value, or a non-boolean ``citation_valid``.
    """
    payload = _extract_json(raw)
    claims_raw = payload.get("claims")
    if not isinstance(claims_raw, list):
        raise JudgeParseError("judge output JSON lacks a 'claims' list")
    supported = unsupported = contradicted = 0
    citation_flags: list[bool] = []
    for i, entry in enumerate(claims_raw):
        if not isinstance(entry, dict):
            raise JudgeParseError(f"claims[{i}] is not an object")
        verdict = entry.get("verdict")
        if verdict == "supported":
            supported += 1
        elif verdict == "unsupported":
            unsupported += 1
        elif verdict == "contradicted":
            contradicted += 1
        else:
            raise JudgeParseError(f"claims[{i}] has invalid verdict {verdict!r}")
        citation_valid = entry.get("citation_valid")
        if isinstance(citation_valid, bool):
            citation_flags.append(citation_valid)
        elif citation_valid is not None:
            raise JudgeParseError(
                f"claims[{i}] citation_valid must be true/false/null, got {citation_valid!r}"
            )
    notes_raw = payload.get("notes")
    ratio = sum(citation_flags) / len(citation_flags) if citation_flags else None
    return JudgeVerdict(
        claims_total=len(claims_raw),
        claims_supported=supported,
        claims_unsupported=unsupported,
        claims_contradicted=contradicted,
        citation_valid_ratio=ratio,
        notes=notes_raw if isinstance(notes_raw, str) else "",
    )
