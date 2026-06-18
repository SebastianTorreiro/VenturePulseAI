"""ISignalRepository backed by Qdrant (local or Cloud, same adapter)."""

import logging

from qdrant_client import AsyncQdrantClient, models

from app.domain.exceptions import RepositoryError
from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.signal_repository import (
    ISignalRepository,
    ScoredSignal,
    SignalFilter,
)
from app.domain.entities.signal import Signal
from app.domain.value_objects.embedding import Embedding
from app.domain.value_objects.identifiers import SignalId
from app.infrastructure.config.settings import QdrantSettings
from app.infrastructure.persistence._payload_codec import (
    payload_to_signal,
    signal_to_payload,
)

logger = logging.getLogger(__name__)


class QdrantSignalRepository(ISignalRepository):
    """Stores signals with their vectors in a single Qdrant collection.

    Construct with the async factory `create()`, which connects and
    ensures the collection exists with the right dimension — that work is
    async and cannot live in __init__.

    The embedding_service is injected only to name the collection
    (ADR-004) and to validate dimensions; this repository never embeds.
    """

    def __init__(
        self, settings: QdrantSettings, embedding_service: IEmbeddingService
    ) -> None:
        self._embedding_service = embedding_service
        model_slug = embedding_service.model_id.replace("/", "_")
        self._collection_name = f"{settings.collection_prefix}__{model_slug}"
        # HttpUrl renders a trailing slash; AsyncQdrantClient wants a bare URL.
        self._client = AsyncQdrantClient(url=str(settings.url).rstrip("/"))

    @classmethod
    async def create(
        cls, settings: QdrantSettings, embedding_service: IEmbeddingService
    ) -> "QdrantSignalRepository":
        repository = cls(settings, embedding_service)
        try:
            await repository._ensure_collection()
        except Exception:
            await repository._client.close()
            raise
        return repository

    @property
    def collection_name(self) -> str:
        return self._collection_name

    async def _ensure_collection(self) -> None:
        expected_dim = self._embedding_service.dimensions
        try:
            if not await self._client.collection_exists(self._collection_name):
                logger.info(
                    "Creating collection %s (dim=%d, cosine)",
                    self._collection_name,
                    expected_dim,
                )
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=expected_dim, distance=models.Distance.COSINE
                    ),
                )
                return
            info = await self._client.get_collection(self._collection_name)
            actual_dim = self._extract_dimension(info)
        except Exception as e:
            raise RepositoryError(
                f"Failed to ensure collection {self._collection_name!r}"
            ) from e

        if actual_dim != expected_dim:
            raise RepositoryError(
                f"Collection {self._collection_name!r} has dimension "
                f"{actual_dim}, but embedding model "
                f"{self._embedding_service.model_id!r} produces {expected_dim} "
                f"(ADR-004: re-index or use a matching collection)"
            )

    @staticmethod
    def _extract_dimension(info: models.CollectionInfo) -> int:
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):  # named vectors: expect exactly one
            if len(vectors) != 1:
                raise RepositoryError(
                    "Collection has multiple named vector configs; "
                    "expected exactly 1"
                )
            vectors = next(iter(vectors.values()))
        return vectors.size

    async def save(self, signal: Signal, embedding: Embedding) -> None:
        point = models.PointStruct(
            id=str(signal.id),
            vector=list(embedding.vector),
            payload=signal_to_payload(signal),
        )
        try:
            await self._client.upsert(
                collection_name=self._collection_name, points=[point]
            )
        except Exception as e:
            raise RepositoryError(f"Failed to save signal {signal.id}") from e

    async def search(
        self, query: Embedding, filters: SignalFilter, limit: int = 10
    ) -> list[ScoredSignal]:
        try:
            response = await self._client.query_points(
                collection_name=self._collection_name,
                query=list(query.vector),
                query_filter=self._build_filter(filters),
                limit=limit,
                with_payload=True,
            )
        except Exception as e:
            raise RepositoryError("Failed to search signals") from e
        return [
            ScoredSignal(
                signal=payload_to_signal(point.payload),
                semantic_score=point.score,
            )
            for point in response.points
        ]

    async def exists(self, content_hash: str) -> bool:
        condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="content_hash",
                    match=models.MatchValue(value=content_hash),
                )
            ]
        )
        try:
            points, _ = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=condition,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        except Exception as e:
            raise RepositoryError("Failed to check signal existence") from e
        return len(points) > 0

    async def get_by_id(self, signal_id: SignalId) -> Signal:
        try:
            results = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=[str(signal_id)],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            raise RepositoryError(
                f"Failed to retrieve signal {signal_id}"
            ) from e

        if not results:
            raise RepositoryError(
                f"Signal {signal_id} not found in collection "
                f"{self._collection_name!r}"
            )

        return payload_to_signal(results[0].payload)

    @staticmethod
    def _build_filter(filters: SignalFilter) -> models.Filter | None:
        must: list[models.FieldCondition] = []
        if filters.signal_type is not None:
            must.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=filters.signal_type),
                )
            )
        if filters.min_amount_usd is not None:
            must.append(
                models.FieldCondition(
                    key="amount_usd",
                    range=models.Range(gte=float(filters.min_amount_usd)),
                )
            )
        if filters.series is not None:
            must.append(
                models.FieldCondition(
                    key="series",
                    match=models.MatchValue(value=filters.series.value),
                )
            )
        if filters.seniority is not None:
            must.append(
                models.FieldCondition(
                    key="seniority",
                    match=models.MatchValue(value=filters.seniority.value),
                )
            )
        if filters.detected_after is not None:
            must.append(
                models.FieldCondition(
                    key="detected_at",
                    range=models.DatetimeRange(gt=filters.detected_after),
                )
            )
        return models.Filter(must=must) if must else None
