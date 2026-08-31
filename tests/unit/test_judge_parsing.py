"""Judge prompt contract + tolerant verdict parsing (no model, no network)."""

import hashlib
import json

import pytest

from hedis_copilot.evals.judge import (
    JUDGE_PROMPT,
    JudgeParseError,
    JudgeVerdict,
    build_judge_input,
    judge_prompt_sha256,
    parse_judge_output,
)

GOOD_JSON = json.dumps(
    {
        "claims": [
            {"claim": "Ages 45-75 are eligible.", "verdict": "supported", "citation_valid": True},
            {
                "claim": "The measure excludes hospice.",
                "verdict": "supported",
                "citation_valid": False,
            },
            {"claim": "The threshold is 90%.", "verdict": "contradicted", "citation_valid": None},
            {"claim": "It applies to Medicaid.", "verdict": "unsupported", "citation_valid": None},
        ],
        "notes": "one uncited claim",
    }
)


class TestParseGoodOutput:
    def test_counts_and_ratio(self) -> None:
        verdict = parse_judge_output(GOOD_JSON)
        assert verdict == JudgeVerdict(
            claims_total=4,
            claims_supported=2,
            claims_unsupported=1,
            claims_contradicted=1,
            citation_valid_ratio=0.5,
            notes="one uncited claim",
        )
        assert verdict.faithfulness == 0.5

    def test_json_surrounded_by_prose(self) -> None:
        raw = f"Sure, here is my grading:\n\n{GOOD_JSON}\n\nLet me know if you need more."
        assert parse_judge_output(raw).claims_total == 4

    def test_json_inside_markdown_fence(self) -> None:
        raw = f"```json\n{GOOD_JSON}\n```"
        assert parse_judge_output(raw).claims_supported == 2

    def test_invalid_brace_blob_before_real_json_is_skipped(self) -> None:
        raw = "{this is not json} but this is: " + GOOD_JSON
        assert parse_judge_output(raw).claims_total == 4

    def test_ratio_none_when_no_claim_carries_citation_flag(self) -> None:
        raw = json.dumps(
            {
                "claims": [{"claim": "c", "verdict": "supported", "citation_valid": None}],
                "notes": "",
            }
        )
        verdict = parse_judge_output(raw)
        assert verdict.citation_valid_ratio is None
        assert verdict.claims_supported == 1

    def test_missing_optional_fields_tolerated(self) -> None:
        # No citation_valid keys at all and no notes: still a valid per-claim verdict.
        raw = json.dumps({"claims": [{"claim": "c", "verdict": "unsupported"}]})
        verdict = parse_judge_output(raw)
        assert verdict.claims_unsupported == 1
        assert verdict.citation_valid_ratio is None
        assert verdict.notes == ""

    def test_empty_claims_list(self) -> None:
        verdict = parse_judge_output('{"claims": [], "notes": "refusal answer"}')
        assert verdict.claims_total == 0
        assert verdict.faithfulness is None


class TestParseGarbage:
    def test_pure_prose_raises(self) -> None:
        with pytest.raises(JudgeParseError, match="no JSON object"):
            parse_judge_output("I cannot grade this answer, sorry.")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(JudgeParseError):
            parse_judge_output("")

    def test_json_without_claims_raises(self) -> None:
        with pytest.raises(JudgeParseError, match="'claims' list"):
            parse_judge_output('{"verdicts": []}')

    def test_claims_not_a_list_raises(self) -> None:
        with pytest.raises(JudgeParseError, match="'claims' list"):
            parse_judge_output('{"claims": "all supported"}')

    def test_claim_entry_not_object_raises(self) -> None:
        with pytest.raises(JudgeParseError, match=r"claims\[0\] is not an object"):
            parse_judge_output('{"claims": ["supported"]}')

    def test_unknown_verdict_raises(self) -> None:
        raw = json.dumps({"claims": [{"claim": "c", "verdict": "plausible"}]})
        with pytest.raises(JudgeParseError, match="invalid verdict 'plausible'"):
            parse_judge_output(raw)

    def test_non_boolean_citation_valid_raises(self) -> None:
        raw = json.dumps(
            {"claims": [{"claim": "c", "verdict": "supported", "citation_valid": "yes"}]}
        )
        with pytest.raises(JudgeParseError, match="citation_valid"):
            parse_judge_output(raw)


class TestPromptContract:
    def test_rubric_names_all_three_verdicts_and_strict_json(self) -> None:
        for token in (
            "supported",
            "unsupported",
            "contradicted",
            "citation_valid",
            "STRICT JSON",
            "verbatim",
        ):
            assert token in JUDGE_PROMPT

    def test_prompt_sha_matches_frozen_prompt(self) -> None:
        expected = hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()
        assert judge_prompt_sha256() == expected

    def test_build_judge_input_numbers_passages_in_order(self) -> None:
        text = build_judge_input("Which ages?", "Ages 45-75 [1].", ["passage one", "passage two"])
        assert "[1] passage one" in text
        assert "[2] passage two" in text
        assert text.index("PASSAGES:") < text.index("QUESTION:") < text.index("ANSWER:")
        assert "Which ages?" in text
        assert "Ages 45-75 [1]." in text
