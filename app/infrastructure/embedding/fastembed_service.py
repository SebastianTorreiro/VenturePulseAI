"""FREE implementation of IEmbeddingService using local fastembed (ONNX)."""

import asyncio
import logging

from fastembed import TextEmbedding

from app.domain.exceptions import EmbeddingError
from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.value_objects.embedding import Embedding
from app.infrastructure.config.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


class FastembedEmbeddingService(IEmbeddingService):
    """Embeds text with a local fastembed model — no API, no network.

    fastembed is synchronous, so every call is offloaded to a worker
    thread (asyncio.to_thread) to honor the async port contract without
    blocking the event loop.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._model_id = settings.model_id

        logger.info("Loading fastembed model %s", settings.model_id)
        try:
            self._model = TextEmbedding(model_name=settings.model_id)
            probe = next(iter(self._model.embed(["dimension probe"])))
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load fastembed model {settings.model_id!r}"
            ) from e

        self._dimensions = len(probe)
        if self._dimensions != settings.dimensions:
            raise EmbeddingError(
                f"Configured EMBEDDING__DIMENSIONS ({settings.dimensions}) "
                f"does not match the real dimensions of model "
                f"{settings.model_id!r} ({self._dimensions})"
            )
        logger.info("Model loaded, dimensions=%d", self._dimensions)

    async def embed(self, text: str) -> Embedding:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        try:
            return await asyncio.to_thread(self._embed_sync, texts)
        except Exception as e:
            raise EmbeddingError("fastembed failed to embed batch") from e

    def _embed_sync(self, texts: list[str]) -> list[Embedding]:
        """Runs in a worker thread: embed, convert and validate."""
        return [
            Embedding(vector=tuple(vector.tolist()), model_id=self._model_id)
            for vector in self._model.embed(texts)
        ]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_id(self) -> str:
        return self._model_id
