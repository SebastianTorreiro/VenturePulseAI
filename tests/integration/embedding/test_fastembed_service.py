"""Integration tests for FastembedEmbeddingService (real model, no mocks).

These download/load a real fastembed model, so they are marked
`integration` and excluded from the default `pytest` run. Run with
`pytest -m integration`. Async port methods are driven with
asyncio.run() to avoid depending on a pytest-asyncio plugin.
"""

import asyncio
import math

import pytest

from app.domain.exceptions import EmbeddingError
from app.domain.value_objects.embedding import Embedding
from app.infrastructure.config.settings import EmbeddingSettings
from app.infrastructure.embedding.fastembed_service import (
    FastembedEmbeddingService,
)

pytestmark = pytest.mark.integration


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


@pytest.fixture(scope="module")
def service() -> FastembedEmbeddingService:
    # Module-scoped: load the model once for the whole file.
    return FastembedEmbeddingService(EmbeddingSettings())


def test_embed_returns_embedding_with_correct_dimensions(service):
    result = asyncio.run(service.embed("fintech B2B SaaS startup"))

    assert isinstance(result, Embedding)
    assert result.dimensions == service.dimensions


def test_embed_batch_preserves_input_order(service):
    texts = [
        "fintech payments platform",
        "quantum computing hardware",
        "italian pasta recipes",
    ]
    individually = [asyncio.run(service.embed(text)) for text in texts]

    batch = asyncio.run(service.embed_batch(texts))

    assert len(batch) == len(texts)
    for got, expected in zip(batch, individually):
        assert _cosine(got.vector, expected.vector) > 0.99


def test_embed_batch_empty_list_returns_empty_list(service):
    result = asyncio.run(service.embed_batch([]))

    assert result == []


def test_model_id_matches_configured_id(service):
    assert service.model_id == EmbeddingSettings().model_id


def test_similar_texts_produce_similar_vectors(service):
    related_a = asyncio.run(service.embed("fintech B2B SaaS"))
    related_b = asyncio.run(service.embed("fintech enterprise SaaS"))
    unrelated = asyncio.run(service.embed("medieval gardening techniques"))

    related_similarity = _cosine(related_a.vector, related_b.vector)
    unrelated_similarity = _cosine(related_a.vector, unrelated.vector)

    assert related_similarity > 0.6
    assert unrelated_similarity < 0.5


def test_construction_fails_when_settings_dimensions_mismatch():
    with pytest.raises(EmbeddingError, match="dimensions"):
        FastembedEmbeddingService(EmbeddingSettings(dimensions=999))
