"""Index layer: FastEmbed embeddings, embedded Chroma store, BM25, and the build step."""

from hedis_copilot.index.bm25 import Bm25Index
from hedis_copilot.index.build import build_index
from hedis_copilot.index.embed import FastEmbedEmbeddings
from hedis_copilot.index.store import (
    ChromaStore,
    IndexStamp,
    StaleIndexError,
    load_chunks_jsonl,
    load_stamp,
    verify_stamp,
    write_chunks_jsonl,
)

__all__ = [
    "Bm25Index",
    "ChromaStore",
    "FastEmbedEmbeddings",
    "IndexStamp",
    "StaleIndexError",
    "build_index",
    "load_chunks_jsonl",
    "load_stamp",
    "verify_stamp",
    "write_chunks_jsonl",
]
