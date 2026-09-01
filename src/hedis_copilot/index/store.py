"""Embedded Chroma store + sidecar chunk persistence + index staleness stamp.

Chroma holds ``embed_text`` and thin filterable metadata; full :class:`Chunk` objects are
too rich for Chroma metadata, so they persist as ``chunks.jsonl`` next to the index and
load into a ``dict[chunk_id, Chunk]`` at retriever init. ``stamp.json`` records what built
the index; :func:`verify_stamp` refuses to serve a stale index (fail loudly, never silently
answer from an index built by different code/config).
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from hedis_copilot.config import Settings
from hedis_copilot.corpus.models import Chunk

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

COLLECTION_NAME = "hedis"
STAMP_FILENAME = "stamp.json"
CHUNKS_FILENAME = "chunks.jsonl"


class StaleIndexError(RuntimeError):
    """The on-disk index does not match the current code/config â€” rebuild required."""


class IndexStamp(BaseModel):
    """What built the index; mismatches against current settings mark the index stale."""

    model_config = ConfigDict(frozen=True)

    embedding_model: str
    config_hash: str
    manifest_version: str
    chunk_count: int
    built_at: str
    """ISO-8601 UTC timestamp of the build."""


def stamp_path(index_dir: Path) -> Path:
    return index_dir / STAMP_FILENAME


def chunks_path(index_dir: Path) -> Path:
    return index_dir / CHUNKS_FILENAME


def write_stamp(index_dir: Path, stamp: IndexStamp) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    stamp_path(index_dir).write_text(stamp.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_stamp(index_dir: Path) -> IndexStamp:
    path = stamp_path(index_dir)
    if not path.is_file():
        raise StaleIndexError(f"no index stamp at {path} â€” run `hedis build` to build the index")
    return IndexStamp.model_validate_json(path.read_text(encoding="utf-8"))


def verify_stamp(settings: Settings) -> IndexStamp:
    """Load the stamp under ``settings.index_dir`` and check it matches current settings.

    Raises :class:`StaleIndexError` with a rebuild hint on any mismatch.
    """
    stamp = load_stamp(settings.index_dir)
    problems: list[str] = []
    if stamp.embedding_model != settings.embedding_model:
        problems.append(
            f"embedding model {stamp.embedding_model!r} != configured {settings.embedding_model!r}"
        )
    if stamp.config_hash != settings.index_hash():
        problems.append(
            f"index hash {stamp.config_hash} != current {settings.index_hash()} "
            "(a retrieval knob changed)"
        )
    if problems:
        raise StaleIndexError(
            f"index at {settings.index_dir} is stale: " + "; ".join(problems) + " â€” "
            "rebuild with `hedis build`"
        )
    return stamp


def write_chunks_jsonl(path: Path, chunks: Sequence[Chunk]) -> None:
    """Persist full chunks (one JSON per line) next to the index for retriever init."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")


def load_chunks_jsonl(path: Path) -> dict[str, Chunk]:
    if not path.is_file():
        raise StaleIndexError(f"no chunk sidecar at {path} â€” run `hedis build`")
    chunks: dict[str, Chunk] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunk = Chunk.model_validate_json(line)
                chunks[chunk.chunk_id] = chunk
    return chunks


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    # Chroma rejects None metadata values: empty string / -1 stand in for "absent".
    return {
        "doc_id": chunk.doc_id,
        "measure_id": chunk.measure_id or "",
        "section": chunk.section,
        "plan_year": chunk.plan_year,
        "license_posture": chunk.license_posture.value,
        "source_url": chunk.source_url,
        "page": chunk.page if chunk.page is not None else -1,
        "header": chunk.header,
    }


class ChromaStore:
    """Embedded ``chromadb.PersistentClient`` collection over the chunk corpus."""

    def __init__(self, index_dir: Path) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(index_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        self._collection: Collection = self._client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        """Drop and recreate the collection â€” a build always starts empty."""
        # The collection may not exist yet; chroma's not-found error type varies by version.
        with contextlib.suppress(Exception):
            self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return int(self._collection.count())

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return
        vectors: list[Sequence[float] | Sequence[int]] = [list(e) for e in embeddings]
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.embed_text for chunk in chunks],
            embeddings=vectors,
            metadatas=[_chunk_metadata(chunk) for chunk in chunks],
        )

    def query(
        self,
        embedding: Sequence[float],
        k: int,
        measure_ids: list[str] | None = None,
        plan_year: int | None = None,
    ) -> list[tuple[str, float]]:
        """Dense top-k as ``(chunk_id, cosine distance)``, optionally metadata-filtered."""
        total = self.count()
        if total == 0 or k <= 0:
            return []
        conditions: list[dict[str, Any]] = []
        if measure_ids is not None:
            conditions.append({"measure_id": {"$in": list(measure_ids)}})
        if plan_year is not None:
            conditions.append({"plan_year": {"$eq": plan_year}})
        where: dict[str, Any] | None
        if not conditions:
            where = None
        elif len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}
        query_vectors: list[Sequence[float] | Sequence[int]] = [list(embedding)]
        result = self._collection.query(
            query_embeddings=query_vectors,
            n_results=min(k, total),
            where=where,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = (result.get("distances") or [[]])[0]
        return [
            (str(chunk_id), float(distance))
            for chunk_id, distance in zip(ids, distances, strict=True)
        ]

