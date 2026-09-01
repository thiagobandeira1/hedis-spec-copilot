"""Service configuration. Env-only (``HEDIS_`` prefix + standard ANTHROPIC/LANGCHAIN vars)."""

import hashlib
import json
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Configuration missing for the requested operation (e.g. no API key for a real run)."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HEDIS_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Real-model paths only; CI never sets this and never needs it.
    anthropic_api_key: SecretStr | None = None

    answer_model: str = "claude-sonnet-5"
    judge_model: str = "claude-opus-5"

    corpus_dir: Path = Path("corpus")
    index_dir: Path = Path("data/index")

    # Retrieval knobs (serialized into config_hash; changing them stamps evals as different).
    dense_k: int = 20
    bm25_k: int = 20
    rrf_k: int = 60
    final_k: int = 8
    chunk_max_tokens: int = 480
    chunk_overlap_tokens: int = 64
    # Gate A floor on best dense cosine similarity. Calibration on the dev split showed
    # refusal traps do NOT separate from answerables in embedding space (both 0.64-0.79 —
    # plausible healthcare questions land near corpus content by construction), so this is
    # a coarse degenerate-query guard only; substantive refusals are the prompt contract's
    # job (gate B), measured by the full LLM eval. See ADR-007.
    refusal_score_floor: float = 0.35

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    def index_hash(self) -> str:
        """Hash of the knobs baked INTO the index (embedding + chunking) — the stamp key.

        Query-time knobs (k values, floors) deliberately excluded: changing them must not
        force a rebuild of an index they never touched.
        """
        payload = {
            "embedding_model": self.embedding_model,
            "chunk_max_tokens": self.chunk_max_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def config_hash(self) -> str:
        """Hash of every retrieval-affecting knob — stamped into eval artifacts."""
        payload = {
            "index": self.index_hash(),
            "dense_k": self.dense_k,
            "bm25_k": self.bm25_k,
            "rrf_k": self.rrf_k,
            "final_k": self.final_k,
            "refusal_score_floor": self.refusal_score_floor,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def get_settings() -> Settings:
    return Settings()
