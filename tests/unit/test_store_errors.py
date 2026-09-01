"""Corrupt/missing index artifacts must surface as :class:`StaleIndexError` with a
rebuild hint - never as raw pydantic/json tracebacks (and never leaking file content)."""

from pathlib import Path

import pytest

from hedis_copilot.corpus.manifest import LicensePosture
from hedis_copilot.corpus.models import Chunk
from hedis_copilot.index.store import (
    IndexStamp,
    StaleIndexError,
    chunks_path,
    load_chunks_jsonl,
    load_stamp,
    stamp_path,
    write_chunks_jsonl,
    write_stamp,
)


def _stamp() -> IndexStamp:
    return IndexStamp(
        embedding_model="fake-model",
        config_hash="abc123",
        manifest_version="1",
        chunk_count=1,
        built_at="2026-08-31T00:00:00+00:00",
    )


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="fake-tn-2026:C01:description:0123456789abcdef",
        doc_id="fake-tn-2026",
        measure_id="C01",
        measure_name="Breast Cancer Screening",
        section="description",
        header="Fake 2026 Technical Notes > C01 > Description",
        text="The percentage of women screened.",
        plan_year=2026,
        license_posture=LicensePosture.US_GOV_PUBLIC_DOMAIN,
        source_url="https://www.cms.gov/files/document/fake.pdf",
    )


class TestLoadStamp:
    def test_missing_stamp_raises_stale_with_rebuild_hint(self, tmp_path: Path) -> None:
        with pytest.raises(StaleIndexError, match="hedis build"):
            load_stamp(tmp_path)

    def test_corrupt_json_raises_stale_with_rebuild_hint(self, tmp_path: Path) -> None:
        stamp_path(tmp_path).write_text("{not valid json at all", encoding="utf-8")
        with pytest.raises(StaleIndexError, match="hedis build"):
            load_stamp(tmp_path)

    def test_wrong_schema_raises_stale_not_validation_error(self, tmp_path: Path) -> None:
        stamp_path(tmp_path).write_text('{"embedding_model": 42}', encoding="utf-8")
        with pytest.raises(StaleIndexError, match="hedis build"):
            load_stamp(tmp_path)

    def test_corrupt_message_names_cause_class_not_content(self, tmp_path: Path) -> None:
        secret = "SECRET-CONTENT-SHOULD-NOT-LEAK"
        stamp_path(tmp_path).write_text(secret, encoding="utf-8")
        with pytest.raises(StaleIndexError) as excinfo:
            load_stamp(tmp_path)
        assert secret not in str(excinfo.value)
        assert "ValidationError" in str(excinfo.value)

    def test_valid_stamp_round_trips(self, tmp_path: Path) -> None:
        write_stamp(tmp_path, _stamp())
        assert load_stamp(tmp_path) == _stamp()


class TestLoadChunksJsonl:
    def test_missing_sidecar_raises_stale_with_rebuild_hint(self, tmp_path: Path) -> None:
        with pytest.raises(StaleIndexError, match="hedis build"):
            load_chunks_jsonl(chunks_path(tmp_path))

    def test_truncated_line_raises_stale_with_rebuild_hint(self, tmp_path: Path) -> None:
        path = chunks_path(tmp_path)
        write_chunks_jsonl(path, [_chunk()])
        full = path.read_text(encoding="utf-8")
        path.write_text(full[: len(full) // 2], encoding="utf-8")
        with pytest.raises(StaleIndexError, match="hedis build"):
            load_chunks_jsonl(path)

    def test_garbage_line_raises_stale_with_rebuild_hint(self, tmp_path: Path) -> None:
        path = chunks_path(tmp_path)
        path.write_text("this is not json\n", encoding="utf-8")
        with pytest.raises(StaleIndexError, match="hedis build"):
            load_chunks_jsonl(path)

    def test_garbage_message_names_cause_class_not_content(self, tmp_path: Path) -> None:
        path = chunks_path(tmp_path)
        secret = "SECRET-LINE-SHOULD-NOT-LEAK"
        path.write_text(secret + "\n", encoding="utf-8")
        with pytest.raises(StaleIndexError) as excinfo:
            load_chunks_jsonl(path)
        assert secret not in str(excinfo.value)
        assert "ValidationError" in str(excinfo.value)

    def test_valid_sidecar_round_trips(self, tmp_path: Path) -> None:
        path = chunks_path(tmp_path)
        write_chunks_jsonl(path, [_chunk()])
        loaded = load_chunks_jsonl(path)
        assert loaded == {_chunk().chunk_id: _chunk()}
