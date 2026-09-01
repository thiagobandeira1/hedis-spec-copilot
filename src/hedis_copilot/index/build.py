"""`hedis build` core: committed corpus -> chunks -> embeddings -> Chroma + sidecar + stamp.

Deterministic by construction: chunks are sorted by content-hashed ``chunk_id`` before
embedding/insert, so identical corpus + config always produces the identical index.
"""

from datetime import UTC, datetime
from pathlib import Path

import yaml

from hedis_copilot.config import Settings
from hedis_copilot.corpus.chunk import chunk_document
from hedis_copilot.corpus.models import Chunk
from hedis_copilot.corpus.normalize import load_committed
from hedis_copilot.index.embed import FastEmbedEmbeddings
from hedis_copilot.index.store import (
    ChromaStore,
    IndexStamp,
    chunks_path,
    write_chunks_jsonl,
    write_stamp,
)

_EMBED_BATCH_SIZE = 128


def _manifest_version(settings: Settings) -> str:
    """Manifest version for the stamp; 'unversioned' when no manifest exists (fixtures)."""
    path = settings.corpus_dir / "manifest.yaml"
    if not path.is_file():
        return "unversioned"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    version = raw.get("version") if isinstance(raw, dict) else None
    return version if isinstance(version, str) else "unversioned"


def build_index(settings: Settings, committed_dir: Path) -> IndexStamp:
    """Build the full index under ``settings.index_dir`` and return its stamp."""
    docs = load_committed(committed_dir)
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                doc,
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
        )
    chunks.sort(key=lambda chunk: chunk.chunk_id)

    embedder = FastEmbedEmbeddings(settings.embedding_model)
    store = ChromaStore(settings.index_dir)
    store.reset()
    for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
        batch = chunks[start : start + _EMBED_BATCH_SIZE]
        embeddings = embedder.embed_documents([chunk.embed_text for chunk in batch])
        store.add(batch, embeddings)

    write_chunks_jsonl(chunks_path(settings.index_dir), chunks)
    stamp = IndexStamp(
        embedding_model=settings.embedding_model,
        config_hash=settings.index_hash(),
        manifest_version=_manifest_version(settings),
        chunk_count=len(chunks),
        built_at=datetime.now(UTC).isoformat(),
    )
    write_stamp(settings.index_dir, stamp)
    return stamp
