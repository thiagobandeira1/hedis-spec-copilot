"""Metric math fixtures + runner/ratchet/report behavior (all keyless, all deterministic).

Runner (`runner.py`) and report (`report.py`) tests are colocated here — the eval-core
wave's test files are fixed at three, and these modules form one cluster with the metrics.
"""

import json
from pathlib import Path

import pytest

from hedis_copilot.evals.dataset import EvalItem, GoldLabel
from hedis_copilot.evals.report import (
    EVAL_BEGIN,
    EVAL_END,
    ReportError,
    render_readme_table,
    sync_readme,
    write_artifact,
)
from hedis_copilot.evals.retrieval_metrics import aggregate, hit_at, mrr, precision_at, recall_at
from hedis_copilot.evals.runner import RetrievalReport, compare_to_baseline, run_retrieval_eval

GOLD = {"g1"}
RANKED = ["miss-a", "g1", "miss-b"]  # the canonical fixture: gold at rank 2


class TestMetricMath:
    def test_mrr_gold_at_rank_two(self) -> None:
        assert mrr(RANKED, GOLD) == 0.5

    def test_recall_at_1_misses(self) -> None:
        assert recall_at(RANKED, GOLD, 1) == 0.0

    def test_recall_at_3_hits(self) -> None:
        assert recall_at(RANKED, GOLD, 3) == 1.0

    def test_hit_at(self) -> None:
        assert hit_at(RANKED, GOLD, 1) == 0.0
        assert hit_at(RANKED, GOLD, 2) == 1.0

    def test_precision_uses_k_as_denominator(self) -> None:
        assert precision_at(RANKED, GOLD, 2) == 0.5
        assert precision_at(RANKED, GOLD, 3) == pytest.approx(1 / 3)

    def test_precision_k_beyond_list_still_divides_by_k(self) -> None:
        assert precision_at(["g1"], GOLD, 8) == pytest.approx(1 / 8)

    def test_multiple_gold_partial_recall(self) -> None:
        gold = {"g1", "g2"}
        assert recall_at(["g1", "x", "y"], gold, 3) == 0.5
        assert mrr(["x", "g2", "g1"], gold) == 0.5

    def test_mrr_zero_when_gold_never_retrieved(self) -> None:
        assert mrr(["x", "y"], GOLD) == 0.0

    def test_empty_ranked_list(self) -> None:
        assert hit_at([], GOLD, 5) == 0.0
        assert recall_at([], GOLD, 5) == 0.0
        assert mrr([], GOLD) == 0.0

    def test_empty_gold_raises(self) -> None:
        empty: set[str] = set()
        with pytest.raises(ValueError, match="gold set is empty"):
            hit_at(RANKED, empty, 1)
        with pytest.raises(ValueError, match="gold set is empty"):
            recall_at(RANKED, empty, 1)
        with pytest.raises(ValueError, match="gold set is empty"):
            precision_at(RANKED, empty, 1)
        with pytest.raises(ValueError, match="gold set is empty"):
            mrr(RANKED, empty)

    def test_bad_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            hit_at(RANKED, GOLD, 0)


class TestAggregate:
    def test_per_category_and_overall_means(self) -> None:
        agg = aggregate(
            [
                ("eligibility", {"mrr": 1.0, "recall@8": 1.0}),
                ("eligibility", {"mrr": 0.0, "recall@8": 0.5}),
                ("coding", {"mrr": 0.5, "recall@8": 1.0}),
            ]
        )
        assert agg.per_category["eligibility"] == {"mrr": 0.5, "recall@8": 0.75}
        assert agg.per_category["coding"] == {"mrr": 0.5, "recall@8": 1.0}
        assert agg.overall == {"mrr": 0.5, "recall@8": pytest.approx(2.5 / 3)}

    def test_metric_averaged_only_over_carriers(self) -> None:
        agg = aggregate([("a", {"mrr": 1.0}), ("b", {"refused": 0.0})])
        assert agg.overall == {"mrr": 1.0, "refused": 0.0}

    def test_deterministic_key_order(self) -> None:
        agg = aggregate([("zeta", {"b": 1.0, "a": 0.0}), ("alpha", {"a": 1.0})])
        assert list(agg.per_category) == ["alpha", "zeta"]
        assert list(agg.overall) == ["a", "b"]

    def test_empty_input(self) -> None:
        agg = aggregate([])
        assert agg.per_category == {}
        assert agg.overall == {}


def _answerable(item_id: str, category: str = "eligibility") -> EvalItem:
    return EvalItem.model_validate(
        {
            "item_id": item_id,
            "question": f"question for {item_id}",
            "category": category,
            "split": "dev",
            "gold": [GoldLabel(doc_id="d1", measure_name_contains="x", section="exclusions")],
            "reference_answer": "ref",
        }
    )


def _refusal(item_id: str, category: str = "refusal_out_of_corpus") -> EvalItem:
    return EvalItem.model_validate(
        {
            "item_id": item_id,
            "question": f"question for {item_id}",
            "category": category,
            "split": "test",
            "gold": [],
        }
    )


class TestRunner:
    def test_scores_answerable_and_refusal_items(self) -> None:
        items = [
            _answerable("a1"),
            _answerable("a2", category="coding"),
            _refusal("r1"),
            _refusal("r2", category="refusal_licensed_only"),
        ]
        resolved = {"a1": {"g1"}, "a2": {"g2"}, "r1": set[str](), "r2": set[str]()}
        rankings = {
            "question for a1": ["g1", "x", "y"],  # mrr 1.0
            "question for a2": ["x", "g2", "y"],  # mrr 0.5
        }
        report = run_retrieval_eval(
            rankings.__getitem__,
            items,
            resolved,
            gate_a_fn=lambda q: q == "question for r1",
        )
        assert report.overall["mrr"] == 0.75
        assert report.overall["hit@1"] == 0.5
        assert report.overall["recall@8"] == 1.0
        assert report.per_category["eligibility"]["mrr"] == 1.0
        assert report.per_category["coding"]["mrr"] == 0.5
        assert report.per_category["refusal_out_of_corpus"]["refused"] == 1.0
        assert report.per_category["refusal_licensed_only"]["refused"] == 0.0
        assert report.refusal_trap_accuracy == 0.5
        assert len(report.per_item) == 4
        # Refusal traps never leak into retrieval means.
        assert "refused" not in report.overall

    def test_without_gate_a_refusals_are_skipped(self) -> None:
        items = [_answerable("a1"), _refusal("r1")]
        report = run_retrieval_eval(lambda _q: ["g1"], items, {"a1": {"g1"}})
        assert report.refusal_trap_accuracy is None
        assert [r.item_id for r in report.per_item] == ["a1"]

    def test_missing_resolved_gold_raises(self) -> None:
        with pytest.raises(KeyError, match="a1"):
            run_retrieval_eval(lambda _q: [], [_answerable("a1")], {})


def _report(overall: dict[str, float]) -> RetrievalReport:
    return RetrievalReport(per_item=[], per_category={}, overall=overall)


class TestBaselineRatchet:
    def test_detects_regression_beyond_tolerance(self) -> None:
        report = _report({"recall@8": 0.80, "mrr": 0.70})
        regressions = compare_to_baseline(report, {"recall@8": 0.85, "mrr": 0.70})
        assert len(regressions) == 1
        assert "recall@8" in regressions[0]

    def test_within_tolerance_passes(self) -> None:
        report = _report({"recall@8": 0.84, "mrr": 0.69})
        assert compare_to_baseline(report, {"recall@8": 0.85, "mrr": 0.70}) == []

    def test_improvement_passes(self) -> None:
        report = _report({"recall@8": 0.95, "mrr": 0.90})
        assert compare_to_baseline(report, {"recall@8": 0.85, "mrr": 0.70}) == []

    def test_missing_metric_is_a_regression(self) -> None:
        regressions = compare_to_baseline(_report({}), {"recall@8": 0.85})
        assert regressions == ["recall@8: missing from report (baseline 0.8500)"]

    def test_ungated_metrics_ignored(self) -> None:
        report = _report({"recall@8": 0.85, "mrr": 0.70, "hit@1": 0.0})
        assert compare_to_baseline(report, {"recall@8": 0.85, "mrr": 0.70, "hit@1": 0.9}) == []

    def test_custom_tolerance(self) -> None:
        report = _report({"recall@8": 0.80, "mrr": 0.70})
        assert compare_to_baseline(report, {"recall@8": 0.85}, tolerance=0.05) == []
        assert len(compare_to_baseline(report, {"recall@8": 0.85}, tolerance=0.04)) == 1


ARTIFACT: dict[str, object] = {
    "git_sha": "abc1234",
    "config_hash": "deadbeef00000000",
    "metrics": {"overall": {"mrr": 0.75, "recall@8": 0.9}},
}


class TestReport:
    def test_write_artifact_sorted_keys_trailing_newline(self, tmp_path: Path) -> None:
        out = tmp_path / "results" / "2026-08-30-abc.json"
        write_artifact(out, {"zeta": 1, "alpha": {"b": 2, "a": 1}})
        text = out.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert text.index('"alpha"') < text.index('"zeta"')
        assert json.loads(text) == {"zeta": 1, "alpha": {"b": 2, "a": 1}}

    def test_render_readme_table(self) -> None:
        table = render_readme_table(ARTIFACT)
        assert "| mrr | 0.750 |" in table
        assert "| recall@8 | 0.900 |" in table
        assert "git_sha=abc1234" in table

    def test_render_requires_metrics(self) -> None:
        with pytest.raises(ReportError, match="metrics"):
            render_readme_table({"git_sha": "abc"})

    def test_sync_readme_replaces_between_markers_idempotently(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(
            f"# Title\n\n{EVAL_BEGIN}\nstale hand-typed numbers\n{EVAL_END}\n\ntail\n",
            encoding="utf-8",
        )
        first = sync_readme(readme, ARTIFACT)
        assert "stale hand-typed numbers" not in first
        assert "| mrr | 0.750 |" in first
        assert first.startswith("# Title\n") and first.endswith("tail\n")
        second = sync_readme(readme, ARTIFACT)
        assert second == first
        assert readme.read_text(encoding="utf-8") == first

    def test_sync_readme_missing_markers_raises(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Title with no markers\n", encoding="utf-8")
        with pytest.raises(ReportError, match="markers"):
            sync_readme(readme, ARTIFACT)
