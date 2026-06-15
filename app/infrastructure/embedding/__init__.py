"""Embedding adapters: implementations of IEmbeddingService."""

from app.infrastructure.embedding.fastembed_service import (
    FastembedEmbeddingService,
)

__all__ = ["FastembedEmbeddingService"]
