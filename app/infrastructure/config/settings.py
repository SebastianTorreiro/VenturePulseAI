"""Typed application settings, loaded from the environment / .env.

One sub-settings model per external sub-domain (Qdrant, Embedding, LLM,
Scraper), composed into a single Settings root. Nested environment
variables use the "__" delimiter, e.g. QDRANT__URL, LLM__OLLAMA_MODEL.

This module lives in infrastructure and is the only place allowed to read
configuration. It never imports from app.domain.
"""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, HttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Sub-models validate their string defaults (so HttpUrl defaults are coerced)
# and reject unknown keys, mirroring the root's extra="forbid".
_SUBMODEL_CONFIG = ConfigDict(extra="forbid", validate_default=True)


class QdrantSettings(BaseModel):
    model_config = _SUBMODEL_CONFIG

    url: HttpUrl = "http://localhost:6333"
    collection_prefix: str = "signals"

    @field_validator("collection_prefix")
    @classmethod
    def _prefix_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("QDRANT__COLLECTION_PREFIX must not be empty")
        return value


class EmbeddingSettings(BaseModel):
    model_config = _SUBMODEL_CONFIG

    model_id: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = 384


class LLMSettings(BaseModel):
    model_config = _SUBMODEL_CONFIG

    provider: Literal["ollama", "claude"] = "ollama"
    ollama_host: HttpUrl = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    claude_api_key: str | None = None
    claude_model: str | None = "claude-sonnet-4-6"

    @model_validator(mode="after")
    def _claude_requires_api_key(self) -> "LLMSettings":
        if self.provider == "claude" and self.claude_api_key is None:
            raise ValueError(
                "LLM__PROVIDER=claude requires LLM__CLAUDE_API_KEY to be set"
            )
        return self


class ScraperSettings(BaseModel):
    model_config = _SUBMODEL_CONFIG

    rss_feed_url: HttpUrl = "https://techcrunch.com/category/venture/feed/"
    fetch_timeout_seconds: int = 30


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        env_nested_delimiter="__",
    )

    qdrant: QdrantSettings = QdrantSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    llm: LLMSettings = LLMSettings()
    scraper: ScraperSettings = ScraperSettings()


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings, built once and cached.

    Works with defaults when no .env is present, so tests need no file.
    The cache makes this the single entry point every adapter shares.
    """
    return Settings()
