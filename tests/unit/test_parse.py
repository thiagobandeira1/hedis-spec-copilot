"""White-box tests for the layout-mode text normalizer and field splitter.

``_normalize_block`` and ``_split_fields`` are private helpers of
``hedis_copilot.corpus.parse``; importing them directly is deliberate — these fixtures pin
the exact text-cleanup semantics the committed corpus depends on. Fixtures are shaped like
pypdf layout-mode extractor output: heavy indentation, padded space runs, wrapped lines.
"""

from hedis_copilot.corpus.parse import _normalize_block, _split_fields


class TestNormalizeBlock:
    def test_wrapped_lines_join_with_single_spaces(self) -> None:
        block = (
            "      The percentage of plan members\n"
            "      who received a breast cancer screening\n"
            "      during the measurement period.\n"
        )
        assert _normalize_block(block) == (
            "The percentage of plan members who received a breast cancer screening "
            "during the measurement period."
        )

    def test_space_runs_collapse(self) -> None:
        block = "Data   collected     from    claims."
        assert _normalize_block(block) == "Data collected from claims."

    def test_page_markers_are_stripped(self) -> None:
        block = "before the break\n\x00PAGE:39\x00\nafter the break"
        assert _normalize_block(block) == "before the break after the break"

    def test_bullet_lines_keep_their_own_lines(self) -> None:
        block = (
            "   Exclusions include the following:\n"
            "   • Members in hospice at any time\n"
            "   • Members with end-stage renal disease\n"
        )
        assert _normalize_block(block) == (
            "Exclusions include the following:\n"
            "• Members in hospice at any time\n"
            "• Members with end-stage renal disease"
        )

    def test_bullet_continuation_lines_join_into_the_bullet(self) -> None:
        block = "• Members in hospice care\n  at any time during the year\n• Second item\n"
        assert _normalize_block(block) == (
            "• Members in hospice care at any time during the year\n• Second item"
        )

    def test_spaced_hyphen_rejoin_in_year_range(self) -> None:
        assert _normalize_block("the 2025- 2026 plan period") == "the 2025-2026 plan period"
        assert _normalize_block("the 2025 -2026 plan period") == "the 2025-2026 plan period"

    def test_spaced_hyphen_rejoin_in_compound_word(self) -> None:
        assert _normalize_block("Patient - level data") == "Patient-level data"
        # A wrap right after a hyphenated token half also rejoins.
        assert _normalize_block("member- level\n   detail") == "member-level detail"

    def test_dash_bullets_are_not_swallowed_by_hyphen_rejoin(self) -> None:
        """Bullet dashes follow a newline, never an alnum — they must survive rejoining."""
        block = "Exclusions:\n- hospice care during the year\n- end-stage renal disease\n"
        assert _normalize_block(block) == (
            "Exclusions:\n- hospice care during the year\n- end-stage renal disease"
        )

    def test_blank_lines_vanish_and_result_is_stripped(self) -> None:
        assert _normalize_block("\n\n   text   \n\n") == "text"
        assert _normalize_block("   \n \n") == ""


class TestSplitFields:
    def test_labels_split_into_ordered_fields(self) -> None:
        block = (
            "   Metric:   The percentage of members screened.\n"
            "   Exclusions:  Hospice at any time.\n"
            "   Data Time Frame:  01/01/2024 - 12/31/2024\n"
        )
        fields = _split_fields(block)
        labels = [label for label, _ in fields]
        assert labels == ["Metric", "Exclusions", "Data Time Frame"]
        assert fields[0][1].strip() == "The percentage of members screened."
        assert fields[1][1].strip() == "Hospice at any time."

    def test_doubled_internal_spaces_in_label_still_match_and_canonicalize(self) -> None:
        """Layout mode pads label words: 'Measure  Reference:' must map to one field."""
        block = (
            "   Measure  Reference:  HEDIS Volume 2\n   Data  Time  Frame :  the measurement year\n"
        )
        fields = _split_fields(block)
        assert [label for label, _ in fields] == ["Measure Reference", "Data Time Frame"]

    def test_text_before_first_label_becomes_body(self) -> None:
        block = "Introductory sentence about the measure.\n   Metric:  The percentage.\n"
        fields = _split_fields(block)
        assert fields[0][0] == "Body"
        assert "Introductory sentence" in fields[0][1]
        assert fields[1][0] == "Metric"

    def test_block_without_labels_is_one_body_field(self) -> None:
        block = "Just prose with no recognized labels at all."
        assert _split_fields(block) == [("Body", block)]

    def test_unrecognized_label_stays_inside_previous_field(self) -> None:
        block = "   Metric:  The percentage.\n   Random Label:  should not split.\n"
        fields = _split_fields(block)
        assert [label for label, _ in fields] == ["Metric"]
        assert "Random Label" in fields[0][1]

    def test_longer_label_wins_over_its_prefix(self) -> None:
        """'Metric Notes:' must not be misread as 'Metric' + leftover text."""
        block = "   Metric Notes:  a note about the metric.\n"
        fields = _split_fields(block)
        assert [label for label, _ in fields] == ["Metric Notes"]

    def test_label_mid_line_does_not_split(self) -> None:
        block = "   Metric:  See the Exclusions: field below for details.\n"
        fields = _split_fields(block)
        assert [label for label, _ in fields] == ["Metric"]
