"""FastEmbed adapter behind the LangChain :class:`Embeddings` interface (ADR-002).

BGE models are asymmetric: queries carry an instruction prefix, documents never do.
That asymmetry lives HERE and only here — every other module embeds through this adapter
and stays prefix-ignorant. The ONNX model is initialized lazily on first use, so importing
this module (or constructing the adapter) never triggers a model download.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from fastembed import TextEmbedding

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
"""BGE query instruction — applied to queries exactly once, never to documents."""


class FastEmbedEmbeddings(Embeddings):
    """Wraps ``fastembed.TextEmbedding`` with lazy init and BGE query/document asymmetry."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: TextEmbedding | None = None

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        return [[float(value) for value in vector] for vector in model.embed(texts)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages verbatim — no instruction prefix, ever."""
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with the BGE instruction prefix (applied exactly once)."""
        return self._embed([QUERY_PREFIX + text])[0]
