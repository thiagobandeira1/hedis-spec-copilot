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
    refusal_score_floor: float = 0.02

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    def config_hash(self) -> str:
        """Hash of every retrieval-affecting knob — stamped into the index and eval artifacts."""
        payload = {
            "embedding_model": self.embedding_model,
            "dense_k": self.dense_k,
            "bm25_k": self.bm25_k,
            "rrf_k": self.rrf_k,
            "final_k": self.final_k,
            "chunk_max_tokens": self.chunk_max_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "refusal_score_floor": self.refusal_score_floor,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def get_settings() -> Settings:
    return Settings()
