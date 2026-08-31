"""Alias router against the REAL committed corpus — ids are derived, never hand-kept."""

from pathlib import Path

import pytest

from hedis_copilot.retrieval import aliases
from hedis_copilot.retrieval.aliases import AliasRouter, build_alias_table, load_alias_table

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_DIR = REPO_ROOT / "corpus" / "committed"


@pytest.fixture(scope="module")
def router() -> AliasRouter:
    return AliasRouter(load_alias_table(COMMITTED_DIR))


def test_col_routes_to_colorectal_measure_each_year(router: AliasRouter) -> None:
    assert router.route("What are the COL exclusions?", 2026) == ["C02"]
    assert router.route("What are the COL exclusions?", 2025) == ["C02"]
    assert router.route("who qualifies for colorectal cancer screening", 2026) == ["C02"]
    assert router.route("does a colonoscopy count", 2026) == ["C02"]


def test_measure_ids_differ_across_years(router: AliasRouter) -> None:
    # Star Ratings renumbered between years: CBP is C14 in 2026 but C11 in 2025.
    assert router.route("What is the CBP threshold?", 2026) == ["C14"]
    assert router.route("What is the CBP threshold?", 2025) == ["C11"]
    # Diabetes eye exam: C11 in 2026, C09 in 2025.
    assert router.route("diabetes eye exam requirements", 2026) == ["C11"]
    assert router.route("diabetes eye exam requirements", 2025) == ["C09"]


def test_year_none_unions_both_years(router: AliasRouter) -> None:
    assert router.route("cbp cut point", None) == ["C11", "C14"]


def test_literal_measure_id_mentions_route(router: AliasRouter) -> None:
    assert router.route("What changed in C14 this year?", 2026) == ["C14"]
    assert router.route("what changed in c14 this year?", 2026) == ["C14"]
    # Literal ids route even without any alias-table entry (e.g. display measures).
    assert router.route("Tell me about D07 accuracy", 2026) == ["D07"]


def test_generic_program_queries_route_to_nothing(router: AliasRouter) -> None:
    assert router.route("star rating cut points", 2026) == []
    assert router.route("How are overall star ratings calculated?", 2025) == []
    assert router.route("what is the weighting methodology", 2026) == []


def test_single_word_alias_matches_tokens_not_substrings(router: AliasRouter) -> None:
    # "col" must not fire inside words like "protocol".
    assert router.route("what protocol applies to appeals", 2026) == []


def test_table_is_derived_from_measure_names_not_hand_kept() -> None:
    table = build_alias_table([(2030, "C99", "Colorectal Cancer Screening")])
    assert table["col"] == {2030: ["C99"]}
    assert table["colonoscopy"] == {2030: ["C99"]}
    # A name outside the curated map contributes nothing.
    assert build_alias_table([(2030, "C50", "Rating of Health Plan")]) == {}


def test_module_level_route_and_measure_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    assert aliases.route("COL exclusions", 2026) == ["C02"]
    table = aliases.MEASURE_ALIASES
    assert table["col"][2026] == ["C02"]
    assert set(table["col"]) >= {2025, 2026}
    with pytest.raises(AttributeError):
        _ = aliases.NOPE  # type: ignore[attr-defined]
