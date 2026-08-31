"""RRF math on hand-computed fixtures, including the deterministic tie-break."""

import pytest

from hedis_copilot.retrieval.fusion import rrf


def test_rrf_hand_computed_scores_and_order() -> None:
    fused = rrf(["a", "b", "c"], ["b", "c", "d"], k=60)
    scores = dict(fused)
    # dense ranks: a=1 b=2 c=3; bm25 ranks: b=1 c=2 d=3
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [chunk_id for chunk_id, _ in fused] == ["b", "c", "a", "d"]


def test_rrf_small_k_hand_computed() -> None:
    # k=1 keeps the arithmetic trivially checkable: rank 1 -> 1/2, rank 2 -> 1/3.
    fused = rrf(["x", "y"], ["y"], k=1)
    scores = dict(fused)
    assert scores["x"] == pytest.approx(1 / 2)
    assert scores["y"] == pytest.approx(1 / 3 + 1 / 2)
    assert [chunk_id for chunk_id, _ in fused] == ["y", "x"]


def test_rrf_tie_breaks_on_chunk_id_ascending() -> None:
    # Both ids score exactly 1/(k+1); order must come from the id, not dict order.
    fused = rrf(["b"], ["a"], k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(fused[1][1])
    # Same tie with insertion order reversed still breaks the same way.
    assert [chunk_id for chunk_id, _ in rrf(["a"], ["b"], k=60)] == ["a", "b"]


def test_rrf_empty_legs() -> None:
    assert rrf([], []) == []


def test_rrf_single_leg_preserves_leg_order() -> None:
    fused = rrf(["x", "y", "z"], [], k=10)
    assert [chunk_id for chunk_id, _ in fused] == ["x", "y", "z"]
    assert dict(fused)["x"] == pytest.approx(1 / 11)


def test_rrf_overlap_outranks_single_leg_top() -> None:
    # An id ranked mid-list in BOTH legs beats an id at rank 1 in only one leg (k=60).
    fused = rrf(["solo", "both"], ["both"], k=60)
    assert fused[0][0] == "both"


def test_rrf_deterministic() -> None:
    dense = ["c", "a", "b"]
    bm25 = ["b", "d"]
    assert rrf(dense, bm25) == rrf(list(dense), list(bm25))
