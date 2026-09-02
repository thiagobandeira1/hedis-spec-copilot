"""LLM-judge prompt contract + tolerant output parsing — via :mod:`clinevals` (P5 shim).

**No network code lives here.** The judge model is injected by the caller as a
``BaseChatModel`` (SPEC §6: claude-opus-5, deliberately stronger than and different from the
answerer). The frozen per-claim rubric now comes from
:func:`clinevals.grounding.faithfulness_rubric` parameterized with the HEDIS domain phrase;
it reproduces the pre-adoption ``JUDGE_PROMPT`` byte-for-byte, which the golden sha test in
``tests/unit/test_judge_provenance.py`` (and its twin in clinical-agent-evals) enforces.
"""

from clinevals.grounding import (
    GroundingVerdict,
    build_judge_input,
    faithfulness_rubric,
    parse_grounding_verdict,
)
from clinevals.judge import JudgeParseError, Rubric

HEDIS_DOMAIN = "Medicare Star Ratings / HEDIS measures"
RUBRIC: Rubric = faithfulness_rubric(HEDIS_DOMAIN, name="hedis-grounding-v1")
JUDGE_PROMPT: str = RUBRIC.text

JudgeVerdict = GroundingVerdict
parse_judge_output = parse_grounding_verdict


def judge_prompt_sha256() -> str:
    """SHA-256 of the frozen rubric — stamped into eval artifacts so drift is visible."""
    return RUBRIC.sha256


__all__ = [
    "HEDIS_DOMAIN",
    "JUDGE_PROMPT",
    "RUBRIC",
    "JudgeParseError",
    "JudgeVerdict",
    "build_judge_input",
    "judge_prompt_sha256",
    "parse_judge_output",
]
