"""Gold-set loading/validation + label → chunk resolution (fails loudly per SPEC §6)."""

import json
from pathlib import Path
from typing import Any

import pytest

from hedis_copilot.corpus.manifest import LicensePosture
from hedis_copilot.corpus.models import Chunk, SectionKind
from hedis_copilot.evals.dataset import (
    DatasetError,
    EvalItem,
    GoldLabel,
    LabelResolutionError,
    load_dataset,
    resolve_labels,
)


def _answerable_row(item_id: str = "elig-001", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "item_id": item_id,
        "question": "Which ages are eligible for colorectal cancer screening?",
        "category": "eligibility",
        "split": "dev",
        "gold": [
            {
                "doc_id": "cms-tn-2026",
                "measure_name_contains": "colorectal",
                "section": "denominator",
            }
        ],
        "reference_answer": "Adults 45-75.",
    }
    row.update(overrides)
    return row


def _refusal_row(item_id: str = "refuse-001", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "item_id": item_id,
        "question": "List every ICD-10 code in the COL exclusion value set.",
        "category": "refusal_licensed_only",
        "split": "test",
        "gold": [],
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class TestLoadDataset:
    def test_loads_valid_dataset(self, tmp_path: Path) -> None:
        path = _write_jsonl(tmp_path / "q.jsonl", [_answerable_row(), _refusal_row()])
        items = load_dataset(path)
        assert [item.item_id for item in items] == ["elig-001", "refuse-001"]
        assert not items[0].is_refusal
        assert items[1].is_refusal
        assert items[0].gold[0].measure_name_contains == "colorectal"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "q.jsonl"
        path.write_text(json.dumps(_answerable_row()) + "\n\n\n", encoding="utf-8")
        assert len(load_dataset(path)) == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nope.jsonl")

    def test_invalid_json_names_line(self, tmp_path: Path) -> None:
        path = tmp_path / "q.jsonl"
        path.write_text(json.dumps(_answerable_row()) + "\n{not json\n", encoding="utf-8")
        with pytest.raises(DatasetError, match=r"q\.jsonl:2"):
            load_dataset(path)

    def test_unknown_category_rejected(self, tmp_path: Path) -> None:
        path = _write_jsonl(tmp_path / "q.jsonl", [_answerable_row(category="vibes")])
        with pytest.raises(DatasetError, match=r"q\.jsonl:1"):
            load_dataset(path)

    def test_duplicate_item_ids_rejected(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "q.jsonl", [_answerable_row("dup-1"), _answerable_row("dup-1")]
        )
        with pytest.raises(DatasetError, match="dup-1: duplicate item_id"):
            load_dataset(path)

    def test_refusal_with_gold_rejected(self, tmp_path: Path) -> None:
        bad = _refusal_row(gold=_answerable_row()["gold"])
        path = _write_jsonl(tmp_path / "q.jsonl", [bad])
        with pytest.raises(DatasetError, match="refusal item must carry no gold"):
            load_dataset(path)

    def test_answerable_without_gold_rejected(self, tmp_path: Path) -> None:
        path = _write_jsonl(tmp_path / "q.jsonl", [_answerable_row(gold=[])])
        with pytest.raises(DatasetError, match="needs >= 1 gold label"):
            load_dataset(path)

    def test_answerable_without_reference_answer_rejected(self, tmp_path: Path) -> None:
        for missing_ref in (None, "  "):
            path = _write_jsonl(
                tmp_path / "q.jsonl", [_answerable_row(reference_answer=missing_ref)]
            )
            with pytest.raises(DatasetError, match="needs a reference_answer"):
                load_dataset(path)

    def test_all_violations_reported_at_once(self, tmp_path: Path) -> None:
        rows = [_answerable_row("a", gold=[]), _refusal_row("b", gold=_answerable_row()["gold"])]
        path = _write_jsonl(tmp_path / "q.jsonl", rows)
        with pytest.raises(DatasetError) as excinfo:
            load_dataset(path)
        message = str(excinfo.value)
        assert "a: answerable item needs" in message
        assert "b: refusal item must carry no gold" in message


def _chunk(
    chunk_id: str,
    *,
    doc_id: str = "cms-tn-2026",
    measure_id: str | None = "C14",
    measure_name: str | None = "Colorectal Cancer Screening",
    section: SectionKind = "denominator",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        measure_id=measure_id,
        measure_name=measure_name,
        section=section,
        header="[h]",
        text="body",
        plan_year=2026,
        license_posture=LicensePosture.US_GOV_PUBLIC_DOMAIN,
        source_url="https://example.gov/doc.pdf",
    )


CHUNKS = [
    _chunk("col-den-1"),
    _chunk("col-den-2"),  # a split section yields several chunks; all are gold
    _chunk("col-exc-1", section="exclusions"),
    _chunk("cbp-den-1", measure_id="C11", measure_name="Controlling High Blood Pressure"),
    _chunk("gen-1", measure_id=None, measure_name=None, section="general"),
    _chunk("2025-col-den", doc_id="cms-tn-2025"),
]


def _item(item_id: str, gold: list[GoldLabel]) -> EvalItem:
    return EvalItem(
        item_id=item_id,
        question="q",
        category="eligibility",
        split="dev",
        gold=gold,
        reference_answer="ref",
    )


class TestResolveLabels:
    def test_matches_all_chunks_of_doc_measure_section(self) -> None:
        label = GoldLabel(
            doc_id="cms-tn-2026", measure_name_contains="colorectal", section="denominator"
        )
        resolved = resolve_labels([_item("i1", [label])], CHUNKS)
        assert resolved == {"i1": {"col-den-1", "col-den-2"}}

    def test_substring_match_is_case_insensitive(self) -> None:
        label = GoldLabel(
            doc_id="cms-tn-2026", measure_name_contains="COLORECTAL CANCER", section="exclusions"
        )
        resolved = resolve_labels([_item("i1", [label])], CHUNKS)
        assert resolved == {"i1": {"col-exc-1"}}

    def test_doc_id_and_section_are_exact_filters(self) -> None:
        label = GoldLabel(
            doc_id="cms-tn-2025", measure_name_contains="colorectal", section="denominator"
        )
        resolved = resolve_labels([_item("i1", [label])], CHUNKS)
        assert resolved == {"i1": {"2025-col-den"}}

    def test_none_measure_targets_general_chunks(self) -> None:
        label = GoldLabel(doc_id="cms-tn-2026", measure_name_contains=None, section="general")
        resolved = resolve_labels([_item("i1", [label])], CHUNKS)
        assert resolved == {"i1": {"gen-1"}}

    def test_multiple_labels_union(self) -> None:
        labels = [
            GoldLabel(
                doc_id="cms-tn-2026", measure_name_contains="colorectal", section="exclusions"
            ),
            GoldLabel(
                doc_id="cms-tn-2026", measure_name_contains="blood pressure", section="denominator"
            ),
        ]
        resolved = resolve_labels([_item("i1", labels)], CHUNKS)
        assert resolved == {"i1": {"col-exc-1", "cbp-den-1"}}

    def test_refusal_item_resolves_to_empty_set(self) -> None:
        refusal = EvalItem(
            item_id="r1", question="q", category="refusal_out_of_corpus", split="test", gold=[]
        )
        assert resolve_labels([refusal], CHUNKS) == {"r1": set()}

    def test_unresolvable_label_fails_loudly_naming_item_and_label(self) -> None:
        good = GoldLabel(
            doc_id="cms-tn-2026", measure_name_contains="colorectal", section="denominator"
        )
        vanished = GoldLabel(
            doc_id="cms-tn-2026", measure_name_contains="colorectal", section="timeline"
        )
        with pytest.raises(LabelResolutionError) as excinfo:
            resolve_labels([_item("ok", [good]), _item("broken", [vanished])], CHUNKS)
        message = str(excinfo.value)
        assert "1 gold label(s) resolved to zero chunks" in message
        assert "broken" in message
        assert "section=timeline" in message
        assert excinfo.value.unresolved == [("broken", vanished)]
