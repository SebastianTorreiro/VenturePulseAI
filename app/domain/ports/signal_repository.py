"""Port: persistence and hybrid vector search of signals."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from app.domain.entities.signal import Signal
from app.domain.value_objects.embedding import Embedding
from app.domain.value_objects.enums import FundingSeries, Seniority


@dataclass(frozen=True, slots=True)
class SignalFilter:
    """Payload-level constraints applied alongside the ANN search.

    Every field is optional; a field left as None imposes no constraint.
    An empty SignalFilter() matches every signal.
    """

    signal_type: Literal["funding_round", "job_offer"] | None = None
    min_amount_usd: Decimal | None = None
    series: FundingSeries | None = None
    seniority: Seniority | None = None
    detected_after: datetime | None = None


@dataclass(frozen=True, slots=True)
class ScoredSignal:
    """A signal returned by search together with its semantic similarity."""

    signal: Signal
    semantic_score: float  # cosine/dot similarity, normalized to [0, 1]


class ISignalRepository(ABC):
    """Stores signals with their vectors and serves hybrid search."""

    @abstractmethod
    async def save(self, signal: Signal, embedding: Embedding) -> None:
        """Persist a signal together with its vector, atomically.

        Guarantee: the signal is never stored without its embedding nor
        vice versa; on success both are queryable. Saving a signal whose
        content_hash already exists is an upsert (idempotent by id).

        Raises:
            RepositoryError: the vector store failed (connection,
                missing collection, dimension mismatch with the schema).
        """
        ...

    @abstractmethod
    async def search(
        self, query: Embedding, filters: SignalFilter, limit: int = 10
    ) -> list[ScoredSignal]:
        """Hybrid search: ANN over `query` constrained by `filters`.

        Returns at most `limit` results ordered by descending semantic
        score. Returns an empty list when nothing matches — absence of
        results is not an error.

        Raises:
            RepositoryError: the vector store failed during the query.
        """
        ...

    @abstractmethod
    async def exists(self, content_hash: str) -> bool:
        """Whether a signal with this content_hash is already persisted.

        Used for deduplication before spending embedding budget
        (see Signal.content_hash).

        Raises:
            RepositoryError: the vector store failed during the lookup.
        """
        ...
