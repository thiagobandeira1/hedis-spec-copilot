"""Index + retrieval integration: embedder asymmetry (mocked fastembed, keyless) and a
real end-to-end build/query over a 2-measure slice of the committed corpus (@embed)."""

import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from hedis_copilot.config import Settings
from hedis_copilot.corpus.models import NormalizedDoc
from hedis_copilot.index.embed import QUERY_PREFIX, FastEmbedEmbeddings
from hedis_copilot.index.store import StaleIndexError, load_stamp, verify_stamp
from hedis_copilot.retrieval.hybrid import HybridRetriever

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_DIR = REPO_ROOT / "corpus" / "committed"

# --------------------------------------------------------------------------------------
# Embedder asymmetry â€” mocked fastembed, no model download, no marker.
# --------------------------------------------------------------------------------------


class _FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding; records constructor + embed calls."""

    instances: ClassVar[list["_FakeTextEmbedding"]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[list[str]] = []
        _FakeTextEmbedding.instances.append(self)

    def embed(self, texts: list[str]) -> list[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        return [[0.1, 0.2, 0.3] for _ in batch]


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTextEmbedding]:
    module = types.ModuleType("fastembed")
    module.TextEmbedding = _FakeTextEmbedding  # type: ignore[attr-defined]
    _FakeTextEmbedding.instances = []
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return _FakeTextEmbedding


def test_model_init_is_lazy_and_reused(fake_fastembed: type[_FakeTextEmbedding]) -> None:
    embedder = FastEmbedEmbeddings("BAAI/bge-small-en-v1.5")
    assert fake_fastembed.instances == []  # constructing the adapter loads nothing
    embedder.embed_documents(["alpha"])
    assert len(fake_fastembed.instances) == 1
    assert fake_fastembed.instances[0].model_name == "BAAI/bge-small-en-v1.5"
    embedder.embed_query("beta")
    assert len(fake_fastembed.instances) == 1  # same model instance reused


def test_documents_are_never_prefixed(fake_fastembed: type[_FakeTextEmbedding]) -> None:
    embedder = FastEmbedEmbeddings("m")
    embedder.embed_documents(["alpha", "beta passage"])
    (call,) = fake_fastembed.instances[0].calls
    assert call == ["alpha", "beta passage"]
    assert all(QUERY_PREFIX not in text for text in call)


def test_query_is_prefixed_exactly_once(fake_fastembed: type[_FakeTextEmbedding]) -> None:
    embedder = FastEmbedEmbeddings("m")
    vector = embedder.embed_query("colorectal screening exclusions")
    (call,) = fake_fastembed.instances[0].calls
    assert call == [QUERY_PREFIX + "colorectal screening exclusions"]
    assert call[0].count(QUERY_PREFIX) == 1
    assert vector == [0.1, 0.2, 0.3]


# --------------------------------------------------------------------------------------
# Real index over a small fixture (2 measures from the committed corpus) â€” @embed.
# --------------------------------------------------------------------------------------


def _small_fixture_settings(root: Path) -> Settings:
    """Slice C01 (BCS) + C02 (COL) out of the real 2026 Technical Notes doc."""
    source = NormalizedDoc.model_validate_json(
        (COMMITTED_DIR / "cms-tn-2026.json").read_text(encoding="utf-8")
    )
    keep = {"C01", "C02"}
    small = source.model_copy(
        update={
            "measures": [m for m in source.measures if m.measure_id in keep],
            "general_sections": [],
        }
    )
    committed = root / "corpus" / "committed"
    committed.mkdir(parents=True)
    (committed / "cms-tn-2026.json").write_text(small.model_dump_json(), encoding="utf-8")
    return Settings(corpus_dir=root / "corpus", index_dir=root / "index")


@pytest.fixture(scope="module")
def built_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    from hedis_copilot.index.build import build_index

    root = tmp_path_factory.mktemp("small-index")
    settings = _small_fixture_settings(root)
    stamp = build_index(settings, settings.corpus_dir / "committed")
    assert stamp.chunk_count > 0
    return settings


@pytest.mark.embed
def test_stamp_written_and_verifies(built_settings: Settings) -> None:
    stamp = verify_stamp(built_settings)
    assert stamp.embedding_model == built_settings.embedding_model
    assert stamp.config_hash == built_settings.index_hash()
    assert stamp.manifest_version == "unversioned"  # fixture corpus has no manifest.yaml
    assert stamp.chunk_count > 0
    assert stamp == load_stamp(built_settings.index_dir)


@pytest.mark.embed
def test_colorectal_exclusions_retrieved_in_fused_top4(built_settings: Settings) -> None:
    retriever = HybridRetriever.from_settings(built_settings)
    result = retriever.retrieve(
        "What are the exclusions for colorectal cancer screening?", plan_year=2026
    )
    assert result.measure_filter == ["C02"]
    assert not result.used_fallback
    top4 = result.chunks[:4]
    assert any(
        scored.chunk.measure_id == "C02" and scored.chunk.section == "exclusions" for scored in top4
    ), [(s.chunk.measure_id, s.chunk.section) for s in top4]
    # Provenance: every fused chunk carries at least one leg rank and the rrf score.
    for scored in result.chunks:
        assert scored.dense_rank is not None or scored.bm25_rank is not None
        assert scored.score > 0


@pytest.mark.embed
def test_unfiltered_query_still_answers(built_settings: Settings) -> None:
    retriever = HybridRetriever.from_settings(built_settings)
    result = retriever.retrieve("what screening measures exist", plan_year=2026)
    assert result.measure_filter is None
    assert result.chunks  # no filter, no fallback â€” plain hybrid retrieval
    assert not result.used_fallback


@pytest.mark.embed
def test_starved_filter_falls_back_to_unfiltered_union(built_settings: Settings) -> None:
    retriever = HybridRetriever.from_settings(built_settings)
    # C14 exists in the full corpus but not in this 2-measure fixture: the literal-id
    # filter starves, and the retriever unions in the unfiltered pass.
    result = retriever.retrieve("C14 controlling blood pressure exclusions", plan_year=2026)
    assert result.measure_filter == ["C14"]
    assert result.used_fallback
    assert result.chunks


@pytest.mark.embed
def test_stale_stamp_raises_with_rebuild_hint(built_settings: Settings) -> None:
    stale = built_settings.model_copy(update={"chunk_max_tokens": 999})
    with pytest.raises(StaleIndexError, match="hedis build"):
        verify_stamp(stale)
    with pytest.raises(StaleIndexError, match="hedis build"):
        HybridRetriever.from_settings(stale)


@pytest.mark.embed
def test_missing_stamp_raises(tmp_path: Path, built_settings: Settings) -> None:
    missing = built_settings.model_copy(update={"index_dir": tmp_path / "empty"})
    with pytest.raises(StaleIndexError, match="hedis build"):
        verify_stamp(missing)
