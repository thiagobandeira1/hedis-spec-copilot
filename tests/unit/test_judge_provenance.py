"""Golden: the HEDIS judge rubric's sha256 is frozen at its pre-clinevals-adoption value.

A twin test lives in clinical-agent-evals (``faithfulness_rubric`` for the HEDIS domain
phrase). Editing the rubric text — in either repo — breaks both repos' CI by design: stamped
``judge_prompt_sha256`` provenance in committed artifacts must never move silently. Rubric
evolution is a NEW rubric name, not an edit.
"""

from hedis_copilot.evals.judge import HEDIS_DOMAIN, JUDGE_PROMPT, RUBRIC, judge_prompt_sha256

#: Recorded from `hedis_copilot.evals.judge.judge_prompt_sha256()` on main before adoption.
GOLDEN_SHA256 = "31946648b81673911d373c4784496f672dc83f105e3a73d397a9e73f79915409"


def test_judge_prompt_sha_is_golden() -> None:
    assert judge_prompt_sha256() == GOLDEN_SHA256


def test_rubric_is_the_parameterized_hedis_faithfulness_rubric() -> None:
    assert RUBRIC.name == "hedis-grounding-v1"
    assert RUBRIC.text == JUDGE_PROMPT
    assert HEDIS_DOMAIN in JUDGE_PROMPT
