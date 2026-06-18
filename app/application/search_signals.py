"""Use case: semantic search over stored signals.

Imports only from app.domain.* — no infrastructure.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from app.domain.ports.embedding_service import IEmbeddingService
from app.domain.ports.signal_repository import (
    ISignalRepository,
    ScoredSignal,
    SignalFilter,
)


@dataclass(frozen=True)
class SearchResult:
    signals: tuple[ScoredSignal, ...]
    query: str
    total: int


class SearchSignalsUseCase:
    def __init__(
        self, embedder: IEmbeddingService, repository: ISignalRepository
    ) -> None:
        self._embedder = embedder
        self._repo = repository

    async def execute(
        self,
        query: str,
        limit: int = 10,
        signal_type: Literal["funding_round", "job_offer"] | None = None,
        min_amount_usd: Decimal | None = None,
    ) -> SearchResult:
        embedding = await self._embedder.embed(query)
        filters = SignalFilter(
            signal_type=signal_type,
            min_amount_usd=min_amount_usd,
            series=None,
            seniority=None,
            detected_after=None,
        )
        results = await self._repo.search(embedding, filters, limit)
        return SearchResult(
            signals=tuple(results),
            query=query,
            total=len(results),
        )
